// Option-B scheduling laboratory: identical Q2_0 hot-GPU/cold-CPU FFNs,
// serialized and forked inside CUDA graphs. No expert transfers per token.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <sched.h>

extern "C" cudaGraph_t moestream_ggml_cuda_get_captured_graph(ggml_backend_t, const ggml_tensor *);
extern "C" int ggml_cpu_mul_mat_id_negative_skip(void);

namespace {
using Clock = std::chrono::steady_clock;
constexpr int D=2560, F=640, K=10, EXPERTS=64, HOT=32;
void check(cudaError_t status, const char *where) {
    if(status!=cudaSuccess)throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(status));
}
#define CUDA_CHECK_HERE(call) check((call),#call)

struct Ffn {
    ggml_context *ctx=nullptr;
    ggml_backend_buffer_t buffer=nullptr;
    ggml_cgraph *graph=nullptr;
    ggml_tensor *gate=nullptr,*up=nullptr,*down=nullptr,*input=nullptr,*ids=nullptr,*output=nullptr;
    ggml_tensor *shared_gate=nullptr,*shared_up=nullptr,*shared_down=nullptr,*shared_router=nullptr,*shared_output=nullptr;
    Ffn(ggml_backend_t backend,int experts,bool with_shared=false) {
        ctx=ggml_init({ggml_tensor_overhead()*64+ggml_graph_overhead_custom(64,false)+4096,nullptr,true});
        gate=ggml_new_tensor_3d(ctx,GGML_TYPE_Q2_0,D,F,experts);
        up=ggml_new_tensor_3d(ctx,GGML_TYPE_Q2_0,D,F,experts);
        down=ggml_new_tensor_3d(ctx,GGML_TYPE_Q2_0,F,D,experts);
        input=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,D,1,1);
        ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,K,1);
        auto *g=ggml_mul_mat_id(ctx,gate,input,ids);
        auto *u=ggml_mul_mat_id(ctx,up,input,ids);
        auto *act=ggml_swiglu_split(ctx,g,u);
        output=ggml_mul_mat_id(ctx,down,act,ids);
        ggml_set_output(output);
        graph=ggml_new_graph_custom(ctx,64,false);
        ggml_build_forward_expand(graph,output);
        if(with_shared) {
            // Real Q2 checkpoint Layer-13 shared-expert signature.
            shared_gate=ggml_new_tensor_2d(ctx,GGML_TYPE_Q4_K,D,F);
            shared_up=ggml_new_tensor_2d(ctx,GGML_TYPE_Q3_K,D,F);
            shared_down=ggml_new_tensor_2d(ctx,GGML_TYPE_Q2_0,F,D);
            shared_router=ggml_new_tensor_2d(ctx,GGML_TYPE_BF16,D,1);
            auto *x=ggml_reshape_2d(ctx,input,D,1);
            auto *sg=ggml_mul_mat(ctx,shared_gate,x);
            auto *su=ggml_mul_mat(ctx,shared_up,x);
            auto *sa=ggml_swiglu_split(ctx,sg,su);
            auto *sd=ggml_mul_mat(ctx,shared_down,sa);
            auto *weight=ggml_sigmoid(ctx,ggml_mul_mat(ctx,shared_router,x));
            shared_output=ggml_mul(ctx,sd,weight);
            ggml_set_output(shared_output);
            ggml_build_forward_expand(graph,shared_output);
        }
        buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
        if(!buffer)throw std::runtime_error("FFN allocation failed");
    }
    ~Ffn(){ggml_backend_buffer_free(buffer);ggml_free(ctx);}
    void fill() {
        std::mt19937 rng(20261004);
        std::normal_distribution<float> noise(0,0.02f);
        for(auto *w:{gate,up,down,shared_gate,shared_up,shared_down,shared_router}) {
            if(!w)continue;
            const int width=w->ne[0],rows=w->ne[1];
            std::vector<float> values(static_cast<size_t>(width)*rows);
            for(auto &v:values)v=noise(rng);
            std::vector<uint8_t> quant(ggml_row_size(w->type,width)*rows);
            std::vector<float> imatrix(width,1.0f);
            if(ggml_quantize_chunk(w->type,values.data(),quant.data(),0,rows,width,
                ggml_quantize_requires_imatrix(w->type)?imatrix.data():nullptr)!=quant.size())
                throw std::runtime_error("quantization size mismatch");
            for(int e=0;e<w->ne[2];++e)
                ggml_backend_tensor_set(w,quant.data(),static_cast<size_t>(e)*quant.size(),quant.size());
        }
    }
};

struct ColdContext {
    ggml_backend_t backend=nullptr;
    Ffn *ffn=nullptr;
    float *input=nullptr,*output=nullptr;
    int32_t *ids=nullptr;
    const float *expected_input=nullptr;
    const int32_t *expected_ids=nullptr;
    int cold_count=0;
    uint64_t sequence=0;
    std::atomic<uint64_t> completed{0};
    std::atomic<double> elapsed_us{0};
    std::atomic<bool> ok{true};
};

void CUDART_CB compute_cold(void *opaque) {
    auto *c=static_cast<ColdContext *>(opaque);
    const auto started=Clock::now();
    if(std::memcmp(c->input,c->expected_input,D*sizeof(float))!=0 ||
       std::memcmp(c->ids,c->expected_ids,K*sizeof(int32_t))!=0) {
        c->ok.store(false,std::memory_order_release);
        c->completed.store(c->sequence,std::memory_order_release);
        return;
    }
    bool ok=true;
    if(c->cold_count==0) {
        std::fill_n(c->output,D*K,0.0f);
    } else {
        std::memcpy(c->ffn->input->data,c->input,D*sizeof(float));
        std::memcpy(c->ffn->ids->data,c->ids,K*sizeof(int32_t));
        ok=ggml_backend_graph_compute(c->backend,c->ffn->graph)==GGML_STATUS_SUCCESS;
        if(ok)std::memcpy(c->output,c->ffn->output->data,D*K*sizeof(float));
    }
    c->ok.store(ok,std::memory_order_release);
    c->elapsed_us.store(std::chrono::duration<double,std::micro>(Clock::now()-started).count(),std::memory_order_release);
    c->completed.store(c->sequence,std::memory_order_release);
}

__global__ void combine(const float *hot,const float *cold,const float *weights,const float *shared,float *output) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=D)return;
    float sum=0;
    for(int k=0;k<K;++k)sum+=hot[k*D+i]*weights[k]+cold[k*D+i]*weights[k];
    output[i]=shared?sum+shared[i]:sum;
}

cudaGraphNode_t copy_node(cudaGraph_t graph,const cudaGraphNode_t *dep,size_t ndeps,
                          void *dst,const void *src,size_t bytes,cudaMemcpyKind kind) {
    cudaMemcpy3DParms p{};
    p.srcPtr=make_cudaPitchedPtr(const_cast<void *>(src),bytes,bytes,1);
    p.dstPtr=make_cudaPitchedPtr(dst,bytes,bytes,1);
    p.extent=make_cudaExtent(bytes,1,1);
    p.kind=kind;
    cudaGraphNode_t node=nullptr;
    CUDA_CHECK_HERE(cudaGraphAddMemcpyNode(&node,graph,dep,ndeps,&p));
    return node;
}

struct Stats {
    std::vector<double> wall_us,hot_us,cold_us;
    int mismatches=0;
    double mean()const{return std::accumulate(wall_us.begin(),wall_us.end(),0.0)/wall_us.size();}
};

class Experiment {
public:
    ggml_backend_t gpu=nullptr,cpu=nullptr;
    ggml_threadpool_t cpu_pool=nullptr;
    Ffn *hot=nullptr,*cold=nullptr;
    ColdContext callback;
    cudaGraph_t serial=nullptr,forked=nullptr;
    cudaGraphExec_t serial_exec=nullptr,forked_exec=nullptr;
    cudaGraph_t ready_serial=nullptr,ready_fork=nullptr,merge_graph=nullptr;
    cudaGraphExec_t ready_serial_exec=nullptr,ready_fork_exec=nullptr,merge_exec=nullptr;
    cudaStream_t stream=nullptr,copy_stream=nullptr;
    cudaEvent_t hot_start=nullptr,hot_done=nullptr;
    cudaEvent_t cpu_ready=nullptr,copy_done=nullptr;
    float *device_cold=nullptr,*device_weights=nullptr,*device_result=nullptr,*host_result=nullptr;
    int32_t *device_cold_ids=nullptr;
    int32_t expected_cold_ids[K]{};
    std::vector<float> input=std::vector<float>(D),reference;

    explicit Experiment(int threads,bool with_shared) {
        ggml_cpu_init();
        if(!ggml_cpu_mul_mat_id_negative_skip())throw std::runtime_error("patched GGML negative-ID skip required");
        gpu=ggml_backend_cuda_init(0);cpu=ggml_backend_cpu_init();
        if(!gpu||!cpu)throw std::runtime_error("backend initialization failed");
        ggml_backend_cpu_set_n_threads(cpu,threads);
        cpu_set_t original;
        if(sched_getaffinity(0,sizeof(original),&original)!=0)throw std::runtime_error("cannot read CPU affinity");
        auto pool_params=ggml_threadpool_params_default(threads);
        int selected=0;
        for(int i=0;i<GGML_MAX_N_THREADS&&i<CPU_SETSIZE&&selected<threads;++i) {
            if(CPU_ISSET(i,&original)){pool_params.cpumask[i]=true;++selected;}
        }
        if(selected!=threads)throw std::runtime_error("requested CPU threads exceed allowed CPU mask");
        pool_params.strict_cpu=true;
        cpu_pool=ggml_threadpool_new(&pool_params);
        if(!cpu_pool)throw std::runtime_error("CPU pool allocation failed");
        ggml_backend_cpu_set_threadpool(cpu,cpu_pool);
        if(sched_setaffinity(0,sizeof(original),&original)!=0)throw std::runtime_error("cannot restore main-thread affinity");
        hot=new Ffn(gpu,HOT,with_shared);cold=new Ffn(cpu,EXPERTS);
        hot->fill();cold->fill();
        callback.backend=cpu;callback.ffn=cold;
        callback.expected_input=input.data();callback.expected_ids=expected_cold_ids;
        CUDA_CHECK_HERE(cudaMallocHost(&callback.input,D*sizeof(float)));
        CUDA_CHECK_HERE(cudaMallocHost(&callback.ids,K*sizeof(int32_t)));
        CUDA_CHECK_HERE(cudaMallocHost(&callback.output,D*K*sizeof(float)));
        CUDA_CHECK_HERE(cudaMallocHost(&host_result,D*sizeof(float)));
        CUDA_CHECK_HERE(cudaMalloc(&device_cold,D*K*sizeof(float)));
        CUDA_CHECK_HERE(cudaMalloc(&device_cold_ids,K*sizeof(int32_t)));
        CUDA_CHECK_HERE(cudaMalloc(&device_weights,K*sizeof(float)));
        CUDA_CHECK_HERE(cudaMalloc(&device_result,D*sizeof(float)));
        CUDA_CHECK_HERE(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        CUDA_CHECK_HERE(cudaStreamCreateWithFlags(&copy_stream,cudaStreamNonBlocking));
        CUDA_CHECK_HERE(cudaEventCreate(&hot_start));
        CUDA_CHECK_HERE(cudaEventCreate(&hot_done));
        CUDA_CHECK_HERE(cudaEventCreateWithFlags(&cpu_ready,cudaEventDisableTiming|cudaEventBlockingSync));
        CUDA_CHECK_HERE(cudaEventCreateWithFlags(&copy_done,cudaEventDisableTiming));
        float weights[K];for(int k=0;k<K;++k)weights[k]=static_cast<float>(k+1)/55;
        CUDA_CHECK_HERE(cudaMemcpy(device_weights,weights,sizeof(weights),cudaMemcpyHostToDevice));
        update(0,0);
        for(int i=0;i<4;++i) {
            if(ggml_backend_graph_compute(gpu,hot->graph)!=GGML_STATUS_SUCCESS)
                throw std::runtime_error("GPU graph warmup failed");
        }
        const auto child=moestream_ggml_cuda_get_captured_graph(gpu,ggml_graph_node(hot->graph,0));
        if(!child)throw std::runtime_error("integration GGML graph-harvest bridge returned no graph");
        build(child,false,serial,serial_exec);
        build(child,true,forked,forked_exec);
        build_ready(child,false,ready_serial,ready_serial_exec);
        build_ready(child,true,ready_fork,ready_fork_exec);
        build_merge(merge_graph,merge_exec);
    }

    ~Experiment() {
        if(stream)cudaStreamSynchronize(stream);
        if(serial_exec)cudaGraphExecDestroy(serial_exec);
        if(forked_exec)cudaGraphExecDestroy(forked_exec);
        if(ready_serial_exec)cudaGraphExecDestroy(ready_serial_exec);
        if(ready_fork_exec)cudaGraphExecDestroy(ready_fork_exec);
        if(merge_exec)cudaGraphExecDestroy(merge_exec);
        if(serial)cudaGraphDestroy(serial);
        if(forked)cudaGraphDestroy(forked);
        if(ready_serial)cudaGraphDestroy(ready_serial);
        if(ready_fork)cudaGraphDestroy(ready_fork);
        if(merge_graph)cudaGraphDestroy(merge_graph);
        if(hot_start)cudaEventDestroy(hot_start);
        if(hot_done)cudaEventDestroy(hot_done);
        if(cpu_ready)cudaEventDestroy(cpu_ready);
        if(copy_done)cudaEventDestroy(copy_done);
        if(stream)cudaStreamDestroy(stream);
        if(copy_stream)cudaStreamDestroy(copy_stream);
        cudaFreeHost(callback.input);cudaFreeHost(callback.ids);cudaFreeHost(callback.output);cudaFreeHost(host_result);
        cudaFree(device_cold);cudaFree(device_cold_ids);cudaFree(device_weights);cudaFree(device_result);
        delete hot;delete cold;
        ggml_backend_free(gpu);ggml_backend_free(cpu);
        if(cpu_pool)ggml_threadpool_free(cpu_pool);
    }

    void update(int cold_count,int iteration) {
        callback.cold_count=cold_count;callback.sequence=static_cast<uint64_t>(iteration+1);
        callback.completed.store(0,std::memory_order_release);
        callback.ok.store(false,std::memory_order_release);
        std::fill_n(callback.output,D*K,NAN);
        for(int i=0;i<D;++i)input[i]=0.04f*static_cast<float>((i*31)%43-21)+0.001f*iteration;
        int32_t hg[K],cg[K];
        for(int k=0;k<K;++k) {
            const bool is_cold=k<cold_count;
            hg[k]=is_cold?-1:(iteration*13+k*3)%HOT;
            cg[k]=is_cold?HOT+(iteration*13+k*3)%(EXPERTS-HOT):-1;
        }
        ggml_backend_tensor_set(hot->input,input.data(),0,D*sizeof(float));
        ggml_backend_tensor_set(hot->ids,hg,0,sizeof(hg));
        std::memcpy(expected_cold_ids,cg,sizeof(cg));
        CUDA_CHECK_HERE(cudaMemcpy(device_cold_ids,cg,sizeof(cg),cudaMemcpyHostToDevice));
    }

    double replay(bool parallel,bool host_driven) {
        const auto start=Clock::now();
        if(host_driven) {
            CUDA_CHECK_HERE(cudaGraphLaunch(parallel?ready_fork_exec:ready_serial_exec,stream));
            CUDA_CHECK_HERE(cudaEventSynchronize(cpu_ready));
            compute_cold(&callback);
            CUDA_CHECK_HERE(cudaMemcpyAsync(device_cold,callback.output,D*K*sizeof(float),cudaMemcpyHostToDevice,copy_stream));
            CUDA_CHECK_HERE(cudaEventRecord(copy_done,copy_stream));
            CUDA_CHECK_HERE(cudaStreamWaitEvent(stream,copy_done,0));
            CUDA_CHECK_HERE(cudaGraphLaunch(merge_exec,stream));
        } else CUDA_CHECK_HERE(cudaGraphLaunch(parallel?forked_exec:serial_exec,stream));
        CUDA_CHECK_HERE(cudaStreamSynchronize(stream));
        const double us=std::chrono::duration<double,std::micro>(Clock::now()-start).count();
        if(!callback.ok.load(std::memory_order_acquire)||callback.completed.load(std::memory_order_acquire)!=callback.sequence)
            throw std::runtime_error("CPU callback failed or returned stale output");
        for(int i=0;i<D;++i)if(!std::isfinite(host_result[i]))throw std::runtime_error("nonfinite combined output");
        return us;
    }

    Stats run(bool parallel,int cold_count,int iterations,bool compare_output,bool host_driven) {
        for(int i=0;i<12;++i){update(cold_count,i);replay(parallel,host_driven);}
        Stats s;
        if(!compare_output)reference.resize(static_cast<size_t>(iterations)*D);
        for(int i=0;i<iterations;++i) {
            update(cold_count,i+100);
            s.wall_us.push_back(replay(parallel,host_driven));
            float ms=0;CUDA_CHECK_HERE(cudaEventElapsedTime(&ms,hot_start,hot_done));
            s.hot_us.push_back(ms*1000);
            s.cold_us.push_back(callback.elapsed_us.load(std::memory_order_acquire));
            auto *ref=reference.data()+static_cast<size_t>(i)*D;
            if(compare_output) {
                if(std::memcmp(ref,host_result,D*sizeof(float))!=0)++s.mismatches;
            } else std::memcpy(ref,host_result,D*sizeof(float));
        }
        return s;
    }

private:
    void add_merge(cudaGraph_t graph,const cudaGraphNode_t *deps,size_t count) {
        cudaKernelNodeParams kp{};kp.func=reinterpret_cast<void *>(combine);
        auto *hot_output=static_cast<float *>(hot->output->data);
        auto *shared_output=hot->shared_output?static_cast<float *>(hot->shared_output->data):nullptr;
        void *arguments[]={&hot_output,&device_cold,&device_weights,&shared_output,&device_result};
        kp.gridDim=dim3((D+255)/256);kp.blockDim=dim3(256);kp.kernelParams=arguments;
        cudaGraphNode_t merge=nullptr;
        CUDA_CHECK_HERE(cudaGraphAddKernelNode(&merge,graph,deps,count,&kp));
        copy_node(graph,&merge,1,host_result,device_result,D*sizeof(float),cudaMemcpyDeviceToHost);
    }
    void build_merge(cudaGraph_t &graph,cudaGraphExec_t &exec) {
        CUDA_CHECK_HERE(cudaGraphCreate(&graph,0));
        add_merge(graph,nullptr,0);
        CUDA_CHECK_HERE(cudaGraphInstantiate(&exec,graph,0));
    }
    void build_ready(cudaGraph_t child,bool parallel,cudaGraph_t &graph,cudaGraphExec_t &exec) {
        CUDA_CHECK_HERE(cudaGraphCreate(&graph,0));
        cudaGraphNode_t gs=nullptr,gchild=nullptr,ge=nullptr,ready=nullptr;
        CUDA_CHECK_HERE(cudaGraphAddEventRecordNode(&gs,graph,nullptr,0,hot_start));
        CUDA_CHECK_HERE(cudaGraphAddChildGraphNode(&gchild,graph,&gs,1,child));
        CUDA_CHECK_HERE(cudaGraphAddEventRecordNode(&ge,graph,&gchild,1,hot_done));
        const auto input=copy_node(graph,parallel?nullptr:&ge,parallel?0:1,callback.input,hot->input->data,D*sizeof(float),cudaMemcpyDeviceToHost);
        const auto ids=copy_node(graph,&input,1,callback.ids,device_cold_ids,K*sizeof(int32_t),cudaMemcpyDeviceToHost);
        CUDA_CHECK_HERE(cudaGraphAddEventRecordNode(&ready,graph,&ids,1,cpu_ready));
        CUDA_CHECK_HERE(cudaGraphInstantiate(&exec,graph,0));
    }
    void build(cudaGraph_t child,bool parallel,cudaGraph_t &graph,cudaGraphExec_t &exec) {
        CUDA_CHECK_HERE(cudaGraphCreate(&graph,0));
        cudaGraphNode_t gs=nullptr,gchild=nullptr,ge=nullptr;
        CUDA_CHECK_HERE(cudaGraphAddEventRecordNode(&gs,graph,nullptr,0,hot_start));
        CUDA_CHECK_HERE(cudaGraphAddChildGraphNode(&gchild,graph,&gs,1,child));
        CUDA_CHECK_HERE(cudaGraphAddEventRecordNode(&ge,graph,&gchild,1,hot_done));
        const auto input=copy_node(graph,parallel?nullptr:&ge,parallel?0:1,callback.input,hot->input->data,D*sizeof(float),cudaMemcpyDeviceToHost);
        const auto ids=copy_node(graph,&input,1,callback.ids,device_cold_ids,K*sizeof(int32_t),cudaMemcpyDeviceToHost);
        cudaHostNodeParams hp{};hp.fn=compute_cold;hp.userData=&callback;
        cudaGraphNode_t hn=nullptr;
        CUDA_CHECK_HERE(cudaGraphAddHostNode(&hn,graph,&ids,1,&hp));
        const auto copied=copy_node(graph,&hn,1,device_cold,callback.output,D*K*sizeof(float),cudaMemcpyHostToDevice);
        const cudaGraphNode_t deps[]={ge,copied};
        add_merge(graph,deps,2);
        CUDA_CHECK_HERE(cudaGraphInstantiate(&exec,graph,0));
    }
};

double average(const std::vector<double> &v){return std::accumulate(v.begin(),v.end(),0.0)/v.size();}
uint64_t hash(const std::vector<float> &values) {
    uint64_t h=1469598103934665603ULL;
    const auto *bytes=reinterpret_cast<const unsigned char *>(values.data());
    for(size_t i=0;i<values.size()*sizeof(float);++i){h^=bytes[i];h*=1099511628211ULL;}
    return h;
}
} // namespace

int main(int argc,char **argv) {
    std::setbuf(stdout,nullptr);
    const int iterations=argc>1?std::atoi(argv[1]):128;
    const int threads=argc>2?std::atoi(argv[2]):10;
    const int repetitions=argc>3?std::atoi(argv[3]):6;
    const bool with_shared=argc>4?std::atoi(argv[4])!=0:true;
    const bool host_driven=argc>5?std::atoi(argv[5])!=0:false;
    if(iterations<1||threads<1||repetitions<1)return 2;
    try {
        CUDA_CHECK_HERE(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
        Experiment e(threads,with_shared);
        std::printf("D=%d F=%d topk=%d threads=%d iterations=%d repetitions=%d shared=%d host_driven=%d; fixed Q2_0 placement, graph replay\n",D,F,K,threads,iterations,repetitions,with_shared,host_driven);
        bool ok=true;
        for(int cold=0;cold<=K;++cold) {
            std::vector<double> serial_means,fork_means,gpu_means,cpu_means,serial_gpu_means,serial_cpu_means;
            int mismatches=0,wins=0;
            for(int rep=0;rep<repetitions;++rep) {
                Stats serial,parallel;
                if(rep%2==0) {
                    serial=e.run(false,cold,iterations,false,host_driven);
                    parallel=e.run(true,cold,iterations,true,host_driven);
                } else {
                    parallel=e.run(true,cold,iterations,false,host_driven);
                    serial=e.run(false,cold,iterations,true,host_driven);
                }
                mismatches+=serial.mismatches+parallel.mismatches;
                serial_means.push_back(serial.mean());fork_means.push_back(parallel.mean());
                gpu_means.push_back(average(parallel.hot_us));cpu_means.push_back(average(parallel.cold_us));
                serial_gpu_means.push_back(average(serial.hot_us));serial_cpu_means.push_back(average(serial.cold_us));
                if(parallel.mean()<serial.mean())++wins;
            }
            const double serial=average(serial_means),parallel=average(fork_means);
            ok=ok&&mismatches==0;
            std::printf("cold=%d serial_us=%.2f fork_us=%.2f saving_pct=%.2f gpu_branch_us=%.2f cold_callback_us=%.2f serial_gpu_us=%.2f serial_cpu_us=%.2f wins=%d/%d mismatches=%d/%d reference_fnv64=%016llx\n",
                        cold,serial,parallel,100*(serial-parallel)/serial,
                        average(gpu_means),average(cpu_means),average(serial_gpu_means),average(serial_cpu_means),
                        wins,repetitions,mismatches,iterations*repetitions,(unsigned long long)hash(e.reference));
        }
        return ok?0:1;
    } catch(const std::exception &error) {
        std::fprintf(stderr,"overlap spike: %s\n",error.what());return 1;
    }
}

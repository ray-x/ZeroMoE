// Isolate decode FFN backend cost at the model's D=2560, F=640, top-k=10.
// Compare uniform Q2_0 with the two actual UD-IQ1_M routed signatures.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    const bool cuda = argc > 1 && std::string(argv[1]) == "cuda";
    const int nth = argc > 2 ? std::atoi(argv[2]) : 10;
    constexpr int D=2560, F=640, E=32, K=10, ITER=120;
    ggml_cpu_init();
    auto backend = cuda ? ggml_backend_cuda_init(0) : ggml_backend_cpu_init();
    if (!backend) return 2;
    if (!cuda) ggml_backend_cpu_set_n_threads(backend, nth);
    struct Signature { const char *name; ggml_type gate, down; };
    const Signature types[] = {{"Q2/Q2", GGML_TYPE_Q2_0, GGML_TYPE_Q2_0},
                               {"IQ1/IQ4", GGML_TYPE_IQ1_M, GGML_TYPE_IQ4_NL},
                               {"IQ2/IQ4", GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ4_NL}};
    for (const auto &type : types) {
        auto ctx=ggml_init({ggml_tensor_overhead()*64+ggml_graph_overhead_custom(64,false)+4096,nullptr,true});
        auto *wg=ggml_new_tensor_3d(ctx,type.gate,D,F,E);
        auto *wu=ggml_new_tensor_3d(ctx,type.gate,D,F,E);
        auto *wd=ggml_new_tensor_3d(ctx,type.down,F,D,E);
        auto *x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,D,1,1);
        auto *ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,K,1);
        auto *gate=ggml_mul_mat_id(ctx,wg,x,ids);
        auto *up=ggml_mul_mat_id(ctx,wu,x,ids);
        auto *act=ggml_swiglu_split(ctx,gate,up);
        auto *down=ggml_mul_mat_id(ctx,wd,act,ids);
        ggml_set_output(down);
        auto *graph=ggml_new_graph_custom(ctx,64,false);
        ggml_build_forward_expand(graph,down);
        auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
        if (!buffer) return 3;
        std::mt19937 rng(20261004);
        std::normal_distribution<float> noise(0,0.02f);
        for (auto *w : {wg,wu,wd}) {
            const int width=static_cast<int>(w->ne[0]), rows=static_cast<int>(w->ne[1]);
            std::vector<float> values(static_cast<size_t>(width)*rows), imatrix(width,1.0f);
            std::vector<uint8_t> quant(ggml_row_size(w->type,width)*rows);
            for(auto &v:values) v=noise(rng);
            const auto bytes=ggml_quantize_chunk(w->type,values.data(),quant.data(),0,rows,width,
                ggml_quantize_requires_imatrix(w->type)?imatrix.data():nullptr);
            if(bytes!=quant.size()) return 4;
            for(int e=0;e<E;++e) {
                ggml_backend_tensor_set(w,quant.data(),static_cast<size_t>(e)*bytes,bytes);
            }
        }
        std::vector<float> xv(D);
        for(auto &v:xv) v=noise(rng)*50;
        ggml_backend_tensor_set(x,xv.data(),0,xv.size()*sizeof(float));
        for(int valid : {1,2,5,8,10}) {
            std::vector<int32_t> route(K,-1);
            auto run=[&](int it){
                for(int i=0;i<valid;++i) route[i]=(it*13+i*3)%E;
                ggml_backend_tensor_set(ids,route.data(),0,route.size()*sizeof(int32_t));
                const auto start=std::chrono::steady_clock::now();
                if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) std::abort();
                ggml_backend_synchronize(backend);
                return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
            };
            for(int i=0;i<20;++i) run(i);
            std::vector<double> timing;
            for(int i=0;i<ITER;++i) timing.push_back(run(i+50));
            std::sort(timing.begin(),timing.end());
            std::vector<float> output(ggml_nelements(down));
            ggml_backend_tensor_get(down,output.data(),0,output.size()*sizeof(float));
            for(auto v:output) if(!std::isfinite(v)) return 5;
            std::printf("%s %-8s threads=%d valid=%d mean_us=%.2f median_us=%.2f p95_us=%.2f\n",
                        cuda?"CUDA":"CPU",type.name,nth,valid,
                        std::accumulate(timing.begin(),timing.end(),0.0)/ITER,
                        timing[ITER/2],timing[static_cast<size_t>(ITER*0.95)]);
            std::fflush(stdout);
        }
        ggml_backend_buffer_free(buffer); ggml_free(ctx);
    }
    ggml_backend_free(backend);
    return 0;
}

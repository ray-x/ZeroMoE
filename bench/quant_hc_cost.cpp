// CUDA HC down/up pair at the actual shapes: BF16 (Q2 checkpoint) vs Q8_0 (UD).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <random>
#include <vector>

int main() {
    const char *dump_path = std::getenv("ZEROMOE_HC_OUTPUT_DUMP");
    std::ofstream dump;
    if(dump_path && dump_path[0]) {
        dump.open(dump_path,std::ios::binary|std::ios::trunc);
        if(!dump)return 6;
    }
    auto backend=ggml_backend_cuda_init(0);
    if(!backend) return 2;
    constexpr int D=10240, R=320, E=8, ITER=160;
    for(auto type : {GGML_TYPE_BF16,GGML_TYPE_Q8_0}) {
        auto ctx=ggml_init({ggml_tensor_overhead()*128+ggml_graph_overhead_custom(128,false)+4096,nullptr,true});
        auto *wd=ggml_new_tensor_3d(ctx,type,D,R,E);
        auto *wu=ggml_new_tensor_3d(ctx,type,R,D,E);
        // Eight fixed pairs in one graph keep the working set larger than L2
        // without changing pointers and forcing graph recapture per iteration.
        auto *input=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,D,1);
        ggml_tensor *output=nullptr;
        for(int e=0;e<E;++e) {
            auto *down_w=ggml_view_2d(ctx,wd,D,R,wd->nb[1],e*wd->nb[2]);
            auto *up_w=ggml_view_2d(ctx,wu,R,D,wu->nb[1],e*wu->nb[2]);
            auto *down=ggml_mul_mat(ctx,down_w,input);
            auto *pair=ggml_mul_mat(ctx,up_w,down);
            output=output?ggml_concat(ctx,output,pair,1):pair;
        }
        ggml_set_output(output);
        auto *graph=ggml_new_graph_custom(ctx,128,false);
        ggml_build_forward_expand(graph,output);
        auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
        if(!buffer) return 3;
        std::mt19937 rng(20261004);
        std::normal_distribution<float> noise(0,0.02f);
        for(auto *w : {wd,wu}) {
            const int width=w->ne[0], rows=w->ne[1];
            std::vector<float> values(static_cast<size_t>(width)*rows);
            for(auto &v:values)v=noise(rng);
            std::vector<uint8_t> quant(ggml_row_size(type,width)*rows);
            if(ggml_quantize_chunk(type,values.data(),quant.data(),0,rows,width,nullptr)!=quant.size())return 4;
            for(int e=0;e<E;++e)ggml_backend_tensor_set(w,quant.data(),e*quant.size(),quant.size());
        }
        std::vector<float> x(D);
        for(auto &v:x)v=noise(rng)*50;
        ggml_backend_tensor_set(input,x.data(),0,x.size()*sizeof(float));
        auto run=[&](int i){
            (void)i;
            const auto started=std::chrono::steady_clock::now();
            if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)return -1.0;
            ggml_backend_synchronize(backend);
            return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-started).count();
        };
        for(int i=0;i<20;++i)run(i);
        std::vector<double> times;
        for(int i=0;i<ITER;++i)times.push_back(run(i+20));
        std::sort(times.begin(),times.end());
        std::vector<float> y(D*E);
        ggml_backend_tensor_get(output,y.data(),0,y.size()*sizeof(float));
        for(auto v:y)if(!std::isfinite(v))return 5;
        if(dump.is_open()) {
            const uint32_t type_id=static_cast<uint32_t>(type);
            const uint32_t count=static_cast<uint32_t>(y.size());
            dump.write(reinterpret_cast<const char *>(&type_id),sizeof(type_id));
            dump.write(reinterpret_cast<const char *>(&count),sizeof(count));
            dump.write(reinterpret_cast<const char *>(y.data()),y.size()*sizeof(float));
            if(!dump)return 7;
        }
        std::printf("HC %s mean_us=%.2f median_us=%.2f p95_us=%.2f pair_bytes=%zu\n",ggml_type_name(type),
                    std::accumulate(times.begin(),times.end(),0.0)/ITER/E,times[ITER/2]/E,times[152]/E,
                    ggml_nbytes(wd)/E+ggml_nbytes(wu)/E);
        ggml_backend_buffer_free(buffer);ggml_free(ctx);
    }
    ggml_backend_free(backend);
}

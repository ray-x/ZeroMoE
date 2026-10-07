// CPU cost of one routed-MoE layer (gate+up mmid, swiglu, down mmid) per quant type, decode shape (1 token, 10 slots, N valid).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
static double now_us(){return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();}
struct T{const char*n;ggml_type t;};
int main(int argc,char**argv){
  const int D=2560,F=640,NE=128,TOPK=10,iters=300;
  const int nth=argc>1?atoi(argv[1]):10;
  const char* only=argc>2?argv[2]:nullptr;
  T types[]={{"iq1_m",GGML_TYPE_IQ1_M},{"iq2_xxs",GGML_TYPE_IQ2_XXS},{"iq2_xs",GGML_TYPE_IQ2_XS},{"iq3_xxs",GGML_TYPE_IQ3_XXS},
             {"q3_k",GGML_TYPE_Q3_K},{"iq4_xs",GGML_TYPE_IQ4_XS},{"q4_0",GGML_TYPE_Q4_0},{"iq4_nl",GGML_TYPE_IQ4_NL},{"q4_k",GGML_TYPE_Q4_K},
             {"q5_k",GGML_TYPE_Q5_K},{"q6_k",GGML_TYPE_Q6_K},{"q8_0",GGML_TYPE_Q8_0}};
  ggml_backend_t be=ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(be,nth);
  printf("threads=%d  D=%d F=%d  (per expert: gate+up+down = 3*D*F weights)\n",nth,D,F);
  printf("%-8s %9s | %s\n","type","MB/expert","valid=1: us | valid=2 | valid=5 | valid=10   (us per layer; us/expert; eff GB/s at valid=10)");
  for(auto&ty:types){
    if(only&&strcmp(only,ty.n))continue;
    // F=640 is not a multiple of 256 (k-quants need K%256==0 for the down proj); use F=768 for k-quants/iq-xs types.
    int FF=F; if(ggml_blck_size(ty.t)==256) FF=768;
    ggml_init_params ip={ggml_tensor_overhead()*64+ggml_graph_overhead_custom(64,false)+4096,nullptr,true};
    ggml_context*ctx=ggml_init(ip);
    ggml_tensor*Wg=ggml_new_tensor_3d(ctx,ty.t,D,FF,NE),*Wu=ggml_new_tensor_3d(ctx,ty.t,D,FF,NE),*Wd=ggml_new_tensor_3d(ctx,ty.t,FF,D,NE);
    ggml_tensor*x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,D,1,1),*ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,TOPK,1);
    ggml_tensor*g=ggml_mul_mat_id(ctx,Wg,x,ids),*u=ggml_mul_mat_id(ctx,Wu,x,ids);
    ggml_tensor*a=ggml_swiglu_split(ctx,g,u),*d=ggml_mul_mat_id(ctx,Wd,a,ids);
    ggml_cgraph*gr=ggml_new_graph_custom(ctx,64,false); ggml_build_forward_expand(gr,d);
    ggml_backend_buffer_t buf=ggml_backend_alloc_ctx_tensors(ctx,be);
    if(!buf){printf("%-8s alloc failed\n",ty.n);ggml_free(ctx);continue;}
    // fill weights with a non-trivial but valid pattern (quant bytes are arbitrary-valid for these types except fp16 scales)
    { size_t nb[3]={ggml_nbytes(Wg),ggml_nbytes(Wu),ggml_nbytes(Wd)}; ggml_tensor*W[3]={Wg,Wu,Wd};
      for(int i=0;i<3;i++){ std::vector<float> row(W[i]->ne[0]); size_t rows=ggml_nrows(W[i]);
        std::vector<uint8_t> tmp(ggml_row_size(ty.t,W[i]->ne[0]));
        for(int k=0;k<(int)W[i]->ne[0];k++)row[k]=0.01f*((k*7919)%13-6);
        std::vector<float> imat(W[i]->ne[0],1.0f);
        ggml_quantize_chunk(ty.t,row.data(),tmp.data(),0,1,W[i]->ne[0],imat.data());
        for(size_t r=0;r<rows;r++) ggml_backend_tensor_set(W[i],tmp.data(),r*tmp.size(),tmp.size());
        (void)nb; } }
    std::vector<float> xv(D); for(int k=0;k<D;k++)xv[k]=0.02f*((k*31)%11-5); ggml_backend_tensor_set(x,xv.data(),0,D*4);
    double bytes_per_expert=(double)(ggml_nbytes(Wg)+ggml_nbytes(Wu)+ggml_nbytes(Wd))/NE;
    printf("%-8s %9.2f |",ty.n,bytes_per_expert/1e6);
    double us10=0;
    for(int nv:{1,2,5,10}){
      std::vector<int32_t> iv(TOPK,-1);
      auto setids=[&](int it){ for(int i=0;i<nv;i++) iv[i]=(it*17+i*13+3)%NE; ggml_backend_tensor_set(ids,iv.data(),0,TOPK*4); };
      for(int w=0;w<20;w++){setids(w);ggml_backend_graph_compute(be,gr);}
      double tot=0; for(int it=0;it<iters;it++){ setids(it+100); double t0=now_us(); ggml_backend_graph_compute(be,gr); tot+=now_us()-t0; }
      double us=tot/iters; if(nv==10)us10=us;
      printf(" %7.0f",us);
    }
    printf("  | %6.0f us/exp  %5.1f GB/s\n",us10/10,bytes_per_expert*10/us10/1e3);
    ggml_backend_buffer_free(buf); ggml_free(ctx);
  }
  return 0; }

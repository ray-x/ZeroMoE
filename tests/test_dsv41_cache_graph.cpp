// The production graph borrows GpuTopology slabs. Compare that layout against
// ordinary planar GGML weights, including tail padding and immutable bytes.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include "ggml-quants.h"
#include "moe_routed_ggml.hpp"
#include "ggml_f32_quant_backend.hpp"
#include <cstdlib>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

static void cuda_check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
static void require(bool ok, const char *msg) {
  if (!ok) throw std::runtime_error(msg);
}
extern "C" void launch_q2_K_matmul_stream(float *, const float *, const void *, int, int, int, cudaStream_t);
extern "C" void launch_q3_K_matmul_stream(float *, const float *, const void *, int, int, int, cudaStream_t);
static bool native_mode = false;
static void run(int device, int D, int I, int S, int N, int K) {
  cuda_check(cudaSetDevice(device));
  ggml_backend_t be = ggml_backend_cuda_init(device);
  require(be != nullptr, "CUDA backend");
  const size_t gb = ggml_row_size(GGML_TYPE_Q2_K, D) * I;
  const size_t db = ggml_row_size(GGML_TYPE_Q3_K, I) * D;
  const size_t raw = gb * 2 + db;
  const size_t alignment = std::lcm(ggml_row_size(GGML_TYPE_Q2_K, D),
                                     ggml_row_size(GGML_TYPE_Q3_K, I));
  const size_t stride = (raw + alignment - 1) / alignment * alignment;
  std::vector<unsigned char> bytes(stride * N + 256, 0xa5);
  std::vector<unsigned char> gate(gb*N), up(gb*N), down(db*N);
  std::mt19937 rng(803);
  std::uniform_real_distribution<float> random(-0.05f, 0.05f);
  std::vector<float> row(std::max(D,I));
  for (int expert=0; expert<N; ++expert) {
    for (int matrix=0; matrix<3; ++matrix) {
      const int width = matrix==2 ? I : D;
      const int rows = matrix==2 ? D : I;
      const size_t off = matrix==2 ? 2*gb : matrix*gb;
      const size_t rb = ggml_row_size(matrix==2 ? GGML_TYPE_Q3_K : GGML_TYPE_Q2_K, width);
      for (int r=0; r<rows; ++r) {
        for (int d=0; d<width; ++d) row[d]=random(rng);
        auto *dst=bytes.data()+expert*stride+off+r*rb;
        if (matrix==2) quantize_row_q3_K_ref(row.data(), reinterpret_cast<block_q3_K *>(dst), width);
        else quantize_row_q2_K_ref(row.data(), reinterpret_cast<block_q2_K *>(dst), width);
      }
      auto &plane=matrix==0 ? gate : matrix==1 ? up : down;
      const size_t nb=matrix==2 ? db : gb;
      std::memcpy(plane.data()+expert*nb, bytes.data()+expert*stride+off, nb);
    }
  }
  void *pool=nullptr;
  cuda_check(cudaMalloc(&pool,bytes.size()));
  cuda_check(cudaMemcpy(pool,bytes.data(),bytes.size(),cudaMemcpyHostToDevice));
  auto marker=ggml_backend_alloc_buffer(be,256);
  require(marker != nullptr, "external CUDA marker");
  ggml_backend_buffer_set_usage(marker,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  std::vector<float> input((size_t)D*S), weights((size_t)K*S);
  std::vector<int32_t> ids((size_t)K*S);
  for (auto &v:input) v=random(rng)*20;
  for (int t=0;t<S;++t) for(int k=0;k<K;++k) {
    ids[(size_t)t*K+k]=(t+k)%N;
    weights[(size_t)t*K+k]=((t+k)%4==0) ? 0.0f : float(k+1)/(K*(K+1)/2.0f);
    if (native_mode && weights[(size_t)t*K+k]==0) ids[(size_t)t*K+k]=-1;
  }
  std::vector<float> results[2];
  for (int layout=0;layout<2;++layout) {
    auto *ctx=ggml_init({16ull<<20,nullptr,true});
    require(ctx!=nullptr,"context");
    auto *wg=ggml_new_tensor_3d(ctx,GGML_TYPE_Q2_K,D,I,N);
    auto *wu=ggml_new_tensor_3d(ctx,GGML_TYPE_Q2_K,D,I,N);
    auto *wd=ggml_new_tensor_3d(ctx,GGML_TYPE_Q3_K,I,D,N);
    ggml_backend_buffer_t owned=nullptr;
    if (layout==0) {
      size_t offsets[]={0,gb,2*gb};
      ggml_tensor *planes[]={wg,wu,wd};
      for(int m=0;m<3;++m) {
        planes[m]->nb[2]=stride;
        planes[m]->nb[3]=stride*N;
        planes[m]->data=static_cast<unsigned char *>(pool)+offsets[m];
        planes[m]->buffer=marker;
      }
    } else {
      owned=ggml_backend_alloc_ctx_tensors(ctx,be);
      require(owned!=nullptr,"planar weights");
      ggml_backend_buffer_set_usage(owned,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
      ggml_backend_tensor_set(wg,gate.data(),0,gate.size());
      ggml_backend_tensor_set(wu,up.data(),0,up.size());
      ggml_backend_tensor_set(wd,down.data(),0,down.size());
    }
    auto *x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,D,1,S);
    auto *routes=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,K,S);
    auto *rw=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,1,K,S);
    ggml_set_input(x);ggml_set_input(routes);ggml_set_input(rw);
    moe_routed::ExpertWeights ew;ew.gate=wg;ew.up=wu;ew.down=wd;
    auto ffn=moe_routed::expert_ffn(ctx,ew,x,routes);
    const uint64_t before = zeromoe_f32_quant_graph_launches();
    if (native_mode) {
      // Keep the input alive for the standalone oracle after graph compute;
      // GGML may otherwise reuse its allocation for the final routed sum.
      ggml_set_output(x);
      for (auto *node : {ffn.gate, ffn.up, ffn.act, ffn.down}) ggml_set_output(node);
      for (auto *node : {ffn.gate, ffn.up, ffn.down}) ggml_prec_set_src(node,GGML_PREC_F32,1);
    }
    auto *out=moe_routed::sum_topk(ctx,ggml_mul(ctx,ffn.down,rw),D,S,K);
    ggml_set_output(out);
    auto *graph=ggml_new_graph_custom(ctx,256,false);
    ggml_build_forward_expand(graph,out);
    auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    require(ggml_gallocr_alloc_graph(alloc,graph),"activation graph allocation");
    ggml_backend_tensor_set(x,input.data(),0,input.size()*sizeof(float));
    ggml_backend_tensor_set(routes,ids.data(),0,ids.size()*sizeof(int32_t));
    ggml_backend_tensor_set(rw,weights.data(),0,weights.size()*sizeof(float));
    require(ggml_backend_graph_compute(be,graph)==GGML_STATUS_SUCCESS,"expert graph compute");
    results[layout].resize((size_t)D*S);
    ggml_backend_tensor_get(out,results[layout].data(),0,results[layout].size()*sizeof(float));
    if (native_mode) {
      require(zeromoe_f32_quant_graph_launches() >= before+3, "native hook was bypassed");
      // Compare each projection to the original standalone native kernel.
      // Inputs/weights are identical, so these comparisons must be bitwise.
      for (int matrix=0;matrix<3;++matrix) {
        const int width=matrix==2 ? I : D, rows=matrix==2 ? D : I;
        const size_t offset=matrix==2 ? 2*gb : matrix*gb;
        auto *actual=matrix==0 ? ffn.gate : matrix==1 ? ffn.up : ffn.down;
        const size_t count=(size_t)rows*K*S;
        float *reference=nullptr;
        cuda_check(cudaMalloc(&reference,count*sizeof(float)));
        for (int t=0;t<S;++t) for(int k=0;k<K;++k) {
          const int expert=ids[(size_t)t*K+k];
          if(expert<0) {
            cuda_check(cudaMemsetAsync(reference+((size_t)t*K+k)*rows,0,(size_t)rows*sizeof(float),nullptr));
            continue;
          }
          const float *source=matrix==2
              ? static_cast<const float *>(ffn.act->data)+((size_t)t*K+k)*I
              : static_cast<const float *>(x->data)+(size_t)t*D;
          const void *w=static_cast<const unsigned char *>(pool)+expert*stride+offset;
          float *dest=reference+((size_t)t*K+k)*rows;
          if(matrix==2)launch_q3_K_matmul_stream(dest,source,w,1,width,rows,nullptr);
          else launch_q2_K_matmul_stream(dest,source,w,1,width,rows,nullptr);
        }
        std::vector<float> expected(count),got(count);
        cuda_check(cudaMemcpy(expected.data(),reference,count*sizeof(float),cudaMemcpyDeviceToHost));
        ggml_backend_tensor_get(actual,got.data(),0,count*sizeof(float));
        if (std::memcmp(got.data(),expected.data(),count*sizeof(float))) {
          size_t first=count,bad=0;float maxabs=0;
          for(size_t j=0;j<count;++j) if(std::memcmp(&got[j],&expected[j],4)) {
            if(first==count)first=j; ++bad;
            maxabs=std::max(maxabs,std::fabs(got[j]-expected[j]));
          }
          std::vector<float> observed_input(input.size());
          ggml_backend_tensor_get(x,observed_input.data(),0,input.size()*sizeof(float));
          float input_max=0;
          for(size_t j=0;j<input.size();++j)input_max=std::max(input_max,std::fabs(observed_input[j]-input[j]));
          std::fprintf(stderr,"native mismatch dev=%d layout=%d matrix=%d D=%d I=%d S=%d bad=%zu/%zu first=%zu got=%.9g expected=%.9g maxabs=%g input_max=%g\n",
              device,layout,matrix,D,I,S,bad,count,first,got[first],expected[first],maxabs,input_max);
          throw std::runtime_error("native graph projection differs from standalone F32 kernel");
        }
        cuda_check(cudaFree(reference));
      }
    }
    ggml_gallocr_free(alloc);
    if(owned)ggml_backend_buffer_free(owned);
    ggml_free(ctx);
  }
  size_t bad=0;float maxabs=0;
  for(size_t i=0;i<results[0].size();++i) {
    require(std::isfinite(results[0][i]),"nonfinite output");
    if(std::memcmp(&results[0][i],&results[1][i],sizeof(float)))++bad;
    maxabs=std::max(maxabs,std::fabs(results[0][i]-results[1][i]));
  }
  std::vector<unsigned char> after(bytes.size());
  cuda_check(cudaMemcpy(after.data(),pool,after.size(),cudaMemcpyDeviceToHost));
  require(bytes==after,"borrowed expert bytes or tail guard changed");
  std::printf("gpu=%d D=%d I=%d S=%d N=%d K=%d stride=%zu bitwise_mismatches=%zu/%zu max_abs=%g immutable=1\n",
              device,D,I,S,N,K,stride,bad,results[0].size(),maxabs);
  std::fflush(stdout);
  require(bad==0,"strided cache graph differs from planar graph");
  ggml_backend_buffer_free(marker);cuda_check(cudaFree(pool));ggml_backend_free(be);
}

extern "C" void launch_q2_K_grouped_dot_f32_batch(float *,const float *,const void *,size_t,int,int,int,int,cudaStream_t);
static void grouped_graph(int device, int I, int O, int S, int G) {
  cuda_check(cudaSetDevice(device));
  auto be=ggml_backend_cuda_init(device);
  require(be!=nullptr,"grouped backend");
  auto ctx=ggml_init({16ull<<20,nullptr,true});
  require(ctx!=nullptr,"grouped context");
  const size_t rb=ggml_row_size(GGML_TYPE_Q2_K,I);
  const size_t stride=(rb*O+4095)/4096*4096;
  std::vector<unsigned char> bytes(stride*G+256,0x5a);
  std::vector<float> row(I),input((size_t)I*G*S);
  std::mt19937 rng(909);std::uniform_real_distribution<float> random(-1,1);
  for(auto &v:input)v=random(rng);
  for(int g=0;g<G;++g)for(int o=0;o<O;++o) {
    for(auto &v:row)v=random(rng);
    quantize_row_q2_K_ref(row.data(),reinterpret_cast<block_q2_K *>(bytes.data()+g*stride+o*rb),I);
  }
  void *pool=nullptr;float *reference=nullptr;
  cuda_check(cudaMalloc(&pool,bytes.size()));
  cuda_check(cudaMalloc(&reference,(size_t)S*G*O*sizeof(float)));
  cuda_check(cudaMemcpy(pool,bytes.data(),bytes.size(),cudaMemcpyHostToDevice));
  auto marker=ggml_backend_alloc_buffer(be,256);
  ggml_backend_buffer_set_usage(marker,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  auto w=ggml_new_tensor_3d(ctx,GGML_TYPE_Q2_K,I,O,G);
  w->nb[2]=stride;w->nb[3]=stride*G;w->buffer=marker;w->data=pool;
  auto raw=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,I,G,S);ggml_set_input(raw);ggml_set_output(raw);
  auto x=ggml_permute(ctx,raw,0,2,1,3);
  auto mat=ggml_mul_mat(ctx,w,x);ggml_prec_set_src(mat,GGML_PREC_F32,1);
  auto out=ggml_cont(ctx,ggml_permute(ctx,mat,0,2,1,3));ggml_set_output(out);
  auto graph=ggml_new_graph_custom(ctx,256,false);ggml_build_forward_expand(graph,out);
  auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
  require(ggml_gallocr_alloc_graph(alloc,graph),"grouped graph allocation");
  ggml_backend_tensor_set(raw,input.data(),0,input.size()*sizeof(float));
  const auto before=zeromoe_f32_quant_graph_launches();
  require(ggml_backend_graph_compute(be,graph)==GGML_STATUS_SUCCESS,"grouped graph compute");
  require(zeromoe_f32_quant_graph_launches()>before,"grouped native hook bypassed");
  launch_q2_K_grouped_dot_f32_batch(reference,static_cast<const float *>(raw->data),pool,stride,G,O,I,S,nullptr);
  std::vector<float> expected((size_t)S*G*O),actual(expected.size());
  cuda_check(cudaMemcpy(expected.data(),reference,expected.size()*sizeof(float),cudaMemcpyDeviceToHost));
  ggml_backend_tensor_get(out,actual.data(),0,actual.size()*sizeof(float));
  require(std::memcmp(expected.data(),actual.data(),actual.size()*sizeof(float))==0,"grouped graph stride/reduction mismatch");
  std::vector<unsigned char> after(bytes.size());
  cuda_check(cudaMemcpy(after.data(),pool,bytes.size(),cudaMemcpyDeviceToHost));
  require(after==bytes,"grouped weight storage changed");
  std::printf("gpu=%d grouped_graph I=%d O=%d S=%d G=%d checked=%zu bitwise=1 immutable=1\n",device,I,O,S,G,actual.size());
  std::fflush(stdout);
  ggml_gallocr_free(alloc);ggml_backend_buffer_free(marker);ggml_free(ctx);
  cuda_check(cudaFree(pool));cuda_check(cudaFree(reference));ggml_backend_free(be);
}

int main() {
  try {
    const char *mode=std::getenv("ZEROMOE_TEST_NATIVE_F32");
    native_mode=mode && mode[0]=='1';
    if(native_mode)require(zeromoe_install_f32_quant_graph_hook(),"GGML F32 hook unavailable");
    std::printf("native_f32=%d\n",int(native_mode));std::fflush(stdout);
    int count=0;cuda_check(cudaGetDeviceCount(&count));require(count>=2,"both GPUs required");
    for(int dev=0;dev<2;++dev) {
      run(dev,256,256,1,9,6);
      run(dev,256,256,19,9,6);
      run(dev,256,256,817,9,6);
      run(dev,5120,2304,1,7,6);
      if(native_mode) {
        grouped_graph(dev,256,7,3,3);
        grouped_graph(dev,4096,128,17,8);
      }
    }
  }catch(const std::exception &e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}

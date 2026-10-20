// cold-memory, multi-thread: stock vs AVX-512VL iq3_xxs dot over ~2 GiB of weights, plus a raw read baseline
#include "iq3_dot_bench.c"
#undef main
#include <omp.h>
static double secs(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(int argc,char**argv){
    int nt=argc>1?atoi(argv[1]):6; omp_set_num_threads(nt);
    const int K=4096, nb=K/QK_K; const size_t rows=(size_t)20000*8; // 160000 rows * 1568 B = 251 MB per "expert group"; x8 below
    const size_t total_rows=rows*8; // ~2.0 GB
    block_iq3_xxs*x=aligned_alloc(4096,sizeof(block_iq3_xxs)*nb*total_rows); block_q8_K*y=aligned_alloc(64,sizeof(block_q8_K)*nb);
    #pragma omp parallel for schedule(static)
    for(size_t i=0;i<sizeof(block_iq3_xxs)*nb*total_rows/8;i++){ ((uint64_t*)x)[i]=i*0x9E3779B97F4A7C15ull; }
    for(int r=0;r<nb*total_rows;r++) x[r].d=0x3400+(r&0xff);
    for(int b=0;b<nb;b++){ y[b].d=0.01f; for(int j=0;j<QK_K;j++) y[b].qs[j]=(int8_t)(rand()%255-127); }
    float*o=malloc(4*total_rows);
    double bytes=(double)sizeof(block_iq3_xxs)*nb*total_rows;
    for(int rep=0;rep<3;rep++){
        double t0=secs();
        #pragma omp parallel for schedule(static)
        for(size_t r=0;r<total_rows;r++) ref(K,&o[r],x+r*nb,y);
        double t1=secs();
        #pragma omp parallel for schedule(static)
        for(size_t r=0;r<total_rows;r++) opt(K,&o[r],x+r*nb,y);
        double t2=secs();
        uint64_t acc=0;
        #pragma omp parallel for schedule(static) reduction(+:acc)
        for(size_t i=0;i<(size_t)bytes/64;i++){ const uint64_t*p=(const uint64_t*)((const char*)x+i*64); acc+=p[0]+p[1]+p[2]+p[3]+p[4]+p[5]+p[6]+p[7]; }
        double t3=secs();
        printf("threads=%d  stock %.2f GB/s (%.0f ms)  avx512vl %.2f GB/s (%.0f ms, %.3fx)  raw-read %.2f GB/s  [acc %llu]\n",nt,bytes/(t1-t0)/1e9,(t1-t0)*1e3,bytes/(t2-t1)/1e9,(t2-t1)*1e3,(t1-t0)/(t2-t1),bytes/(t3-t2)/1e9,(unsigned long long)(acc&1));
    }
}

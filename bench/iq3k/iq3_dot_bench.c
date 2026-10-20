// standalone: current AVX2 iq3_xxs dot vs mask-expanded AVX-512VL variant; checks bit-exactness and times both
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "keven_signs.h"
#define GGML_CPU_FP16_TO_FP32(x) _cvtsh_ss(x)
static inline float hsum_float_8(const __m256 x) {
    __m128 r = _mm_add_ps(_mm256_extractf128_ps(x,1), _mm256_castps256_ps128(x));
    r = _mm_add_ps(r, _mm_movehl_ps(r,r)); r = _mm_add_ss(r, _mm_movehdup_ps(r)); return _mm_cvtss_f32(r);
}
void ref(int n, float*s, const block_iq3_xxs*x, const block_q8_K*y){
    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs; uint32_t aux32[2];
    const int nb=n/QK_K; __m256 accumf=_mm256_setzero_ps();
    for(int i=0;i<nb;++i){
        const float d=GGML_CPU_FP16_TO_FP32(x[i].d)*y[i].d; const uint8_t*q3=x[i].qs; const uint8_t*gas=x[i].qs+QK_K/4; const int8_t*q8=y[i].qs;
        __m256i sumi1=_mm256_setzero_si256(), sumi2=_mm256_setzero_si256();
        for(int ib32=0;ib32<QK_K/32;ib32+=2){
            const __m256i q8_1=_mm256_loadu_si256((const __m256i*)q8); q8+=32; const __m256i q8_2=_mm256_loadu_si256((const __m256i*)q8); q8+=32;
            const __m256i q2_1=_mm256_set_epi32(iq3xxs_grid[q3[7]],iq3xxs_grid[q3[6]],iq3xxs_grid[q3[5]],iq3xxs_grid[q3[4]],iq3xxs_grid[q3[3]],iq3xxs_grid[q3[2]],iq3xxs_grid[q3[1]],iq3xxs_grid[q3[0]]); q3+=8;
            const __m256i q2_2=_mm256_set_epi32(iq3xxs_grid[q3[7]],iq3xxs_grid[q3[6]],iq3xxs_grid[q3[5]],iq3xxs_grid[q3[4]],iq3xxs_grid[q3[3]],iq3xxs_grid[q3[2]],iq3xxs_grid[q3[1]],iq3xxs_grid[q3[0]]); q3+=8;
            memcpy(aux32,gas,8); gas+=8;
            const __m256i s2_1=_mm256_set_epi64x(signs64[(aux32[0]>>21)&127],signs64[(aux32[0]>>14)&127],signs64[(aux32[0]>>7)&127],signs64[(aux32[0]>>0)&127]);
            const __m256i s2_2=_mm256_set_epi64x(signs64[(aux32[1]>>21)&127],signs64[(aux32[1]>>14)&127],signs64[(aux32[1]>>7)&127],signs64[(aux32[1]>>0)&127]);
            const __m256i dot1=_mm256_maddubs_epi16(q2_1,_mm256_sign_epi8(q8_1,s2_1)); const __m256i dot2=_mm256_maddubs_epi16(q2_2,_mm256_sign_epi8(q8_2,s2_2));
            const uint16_t ls1=aux32[0]>>28, ls2=aux32[1]>>28;
            sumi1=_mm256_add_epi32(sumi1,_mm256_madd_epi16(dot1,_mm256_set1_epi16(2*ls1+1))); sumi2=_mm256_add_epi32(sumi2,_mm256_madd_epi16(dot2,_mm256_set1_epi16(2*ls2+1)));
        }
        accumf=_mm256_fmadd_ps(_mm256_set1_ps(d),_mm256_cvtepi32_ps(_mm256_add_epi32(sumi1,sumi2)),accumf);
    }
    *s=0.25f*hsum_float_8(accumf);
}
void opt(int n, float*s, const block_iq3_xxs*x, const block_q8_K*y){
    const __m256i zero=_mm256_setzero_si256(), m7=_mm256_set1_epi32(127), m0f=_mm256_set1_epi32(0x0F0F0F0F), m01=_mm256_set1_epi32(0x01010101), one=_mm256_set1_epi32(1), dup16=_mm256_set1_epi32(0x00010001);
    _Alignas(32) uint32_t mk[8]; _Alignas(32) uint32_t sc[8];
    const int nb=n/QK_K; __m256 accumf=_mm256_setzero_ps();
    for(int i=0;i<nb;++i){
        const float d=GGML_CPU_FP16_TO_FP32(x[i].d)*y[i].d; const uint8_t*q3=x[i].qs; const int8_t*q8=y[i].qs;
        const __m256i S=_mm256_loadu_si256((const __m256i*)(x[i].qs+QK_K/4));
        const __m256i ls=_mm256_srli_epi32(S,28);
        _mm256_store_si256((__m256i*)sc,_mm256_mullo_epi32(_mm256_add_epi32(_mm256_slli_epi32(ls,1),one),dup16));
        __m256i v=_mm256_and_si256(S,m7);
        v=_mm256_or_si256(v,_mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(S,7),m7),8));
        v=_mm256_or_si256(v,_mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(S,14),m7),16));
        v=_mm256_or_si256(v,_mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(S,21),m7),24));
        __m256i t=_mm256_xor_si256(v,_mm256_and_si256(_mm256_srli_epi32(v,4),m0f));
        t=_mm256_xor_si256(t,_mm256_srli_epi32(t,2)); t=_mm256_xor_si256(t,_mm256_srli_epi32(t,1));
        v=_mm256_or_si256(v,_mm256_slli_epi32(_mm256_and_si256(t,m01),7));
        _mm256_store_si256((__m256i*)mk,v);
        __m256i sumi1=zero,sumi2=zero;
        for(int ib=0;ib<QK_K/64;++ib){
            const int p1=2*ib,p2=2*ib+1; const uint8_t*a=q3+16*ib;
            const __m256i q8_1=_mm256_loadu_si256((const __m256i*)q8); const __m256i q8_2=_mm256_loadu_si256((const __m256i*)(q8+32)); q8+=64;
            const __m256i g1=_mm256_set_epi32(iq3xxs_grid[a[7]],iq3xxs_grid[a[6]],iq3xxs_grid[a[5]],iq3xxs_grid[a[4]],iq3xxs_grid[a[3]],iq3xxs_grid[a[2]],iq3xxs_grid[a[1]],iq3xxs_grid[a[0]]);
            const __m256i g2=_mm256_set_epi32(iq3xxs_grid[a[15]],iq3xxs_grid[a[14]],iq3xxs_grid[a[13]],iq3xxs_grid[a[12]],iq3xxs_grid[a[11]],iq3xxs_grid[a[10]],iq3xxs_grid[a[9]],iq3xxs_grid[a[8]]);
            const __m256i n1=_mm256_mask_sub_epi8(q8_1,(__mmask32)mk[p1],zero,q8_1); const __m256i n2=_mm256_mask_sub_epi8(q8_2,(__mmask32)mk[p2],zero,q8_2);
            sumi1=_mm256_add_epi32(sumi1,_mm256_madd_epi16(_mm256_maddubs_epi16(g1,n1),_mm256_set1_epi32((int)sc[p1])));
            sumi2=_mm256_add_epi32(sumi2,_mm256_madd_epi16(_mm256_maddubs_epi16(g2,n2),_mm256_set1_epi32((int)sc[p2])));
        }
        accumf=_mm256_fmadd_ps(_mm256_set1_ps(d),_mm256_cvtepi32_ps(_mm256_add_epi32(sumi1,sumi2)),accumf);
    }
    *s=0.25f*hsum_float_8(accumf);
}
void pr(int n, float*s, const block_iq3_xxs*x, const block_q8_K*y){
    const __m256i shifts=_mm256_setr_epi32(0,0,7,7,14,14,21,21);
    const __m256i spread=_mm256_setr_epi8(0,0,0,0,4,4,4,4,8,8,8,8,12,12,12,12,0,0,0,0,4,4,4,4,8,8,8,8,12,12,12,12);
    const __m256i bits=_mm256_set1_epi64x(0x8040201008040201);
    const __m256i parity=_mm256_broadcastsi128_si256(_mm_setr_epi8(0,-128,-128,0,-128,0,0,-128,-128,0,0,-128,0,-128,-128,0));
    uint32_t aux32[2];
    const int nb=n/QK_K; __m256 accumf=_mm256_setzero_ps();
    for(int i=0;i<nb;++i){
        const float d=GGML_CPU_FP16_TO_FP32(x[i].d)*y[i].d; const uint8_t*q3=x[i].qs; const uint8_t*gas=x[i].qs+QK_K/4; const int8_t*q8=y[i].qs;
        __m256i sumi1=_mm256_setzero_si256(), sumi2=_mm256_setzero_si256();
        #pragma GCC unroll 4
        for(int ib32=0;ib32<QK_K/32;ib32+=2){
            const __m256i q8_1=_mm256_loadu_si256((const __m256i*)q8); q8+=32; const __m256i q8_2=_mm256_loadu_si256((const __m256i*)q8); q8+=32;
            const __m256i q2_1=_mm256_set_epi32(iq3xxs_grid[q3[7]],iq3xxs_grid[q3[6]],iq3xxs_grid[q3[5]],iq3xxs_grid[q3[4]],iq3xxs_grid[q3[3]],iq3xxs_grid[q3[2]],iq3xxs_grid[q3[1]],iq3xxs_grid[q3[0]]); q3+=8;
            const __m256i q2_2=_mm256_set_epi32(iq3xxs_grid[q3[7]],iq3xxs_grid[q3[6]],iq3xxs_grid[q3[5]],iq3xxs_grid[q3[4]],iq3xxs_grid[q3[3]],iq3xxs_grid[q3[2]],iq3xxs_grid[q3[1]],iq3xxs_grid[q3[0]]); q3+=8;
            memcpy(aux32,gas,8); gas+=8;
            __m256i signs0=_mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(aux32[0]),shifts),_mm256_set1_epi32(127));
            const __m256i parity0=_mm256_shuffle_epi8(parity,_mm256_and_si256(_mm256_xor_si256(signs0,_mm256_srli_epi32(signs0,4)),_mm256_set1_epi8(15)));
            signs0=_mm256_or_si256(signs0,parity0); signs0=_mm256_shuffle_epi8(signs0,spread);
            const __m256i neg0=_mm256_cmpeq_epi8(_mm256_and_si256(signs0,bits),bits);
            const __m256i q8s_1=_mm256_sub_epi8(_mm256_xor_si256(q8_1,neg0),neg0);
            __m256i signs1=_mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(aux32[1]),shifts),_mm256_set1_epi32(127));
            const __m256i parity1=_mm256_shuffle_epi8(parity,_mm256_and_si256(_mm256_xor_si256(signs1,_mm256_srli_epi32(signs1,4)),_mm256_set1_epi8(15)));
            signs1=_mm256_or_si256(signs1,parity1); signs1=_mm256_shuffle_epi8(signs1,spread);
            const __m256i neg1=_mm256_cmpeq_epi8(_mm256_and_si256(signs1,bits),bits);
            const __m256i q8s_2=_mm256_sub_epi8(_mm256_xor_si256(q8_2,neg1),neg1);
            const __m256i dot1=_mm256_maddubs_epi16(q2_1,q8s_1); const __m256i dot2=_mm256_maddubs_epi16(q2_2,q8s_2);
            const uint16_t ls1=aux32[0]>>28, ls2=aux32[1]>>28;
            sumi1=_mm256_add_epi32(sumi1,_mm256_madd_epi16(dot1,_mm256_set1_epi16(2*ls1+1))); sumi2=_mm256_add_epi32(sumi2,_mm256_madd_epi16(dot2,_mm256_set1_epi16(2*ls2+1)));
        }
        accumf=_mm256_fmadd_ps(_mm256_set1_ps(d),_mm256_cvtepi32_ps(_mm256_add_epi32(sumi1,sumi2)),accumf);
    }
    *s=0.25f*hsum_float_8(accumf);
}
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e6+t.tv_nsec*1e-3;}
int main(){
    const int K=4096, nb=K/QK_K, rows=4096; // ~ one expert gate matrix worth, 4096 rows (9.6MB: exceeds L2)
    block_iq3_xxs*x=aligned_alloc(64,sizeof(block_iq3_xxs)*nb*rows); block_q8_K*y=aligned_alloc(64,sizeof(block_q8_K)*nb);
    srand(1); for(size_t i=0;i<sizeof(block_iq3_xxs)*nb*rows;i++) ((uint8_t*)x)[i]=rand();
    for(int r=0;r<rows*nb;r++) x[r].d=0x3400+(rand()&0xff);
    for(int b=0;b<nb;b++){ y[b].d=0.01f; for(int j=0;j<QK_K;j++) y[b].qs[j]=(int8_t)(rand()%255-127); }
    int bad=0; float*o1=malloc(4*rows),*o2=malloc(4*rows),*o3=malloc(4*rows); int bad3=0;
    for(int r=0;r<rows;r++){ ref(K,&o1[r],x+(size_t)r*nb,y); opt(K,&o2[r],x+(size_t)r*nb,y); if(memcmp(&o1[r],&o2[r],4)) bad++; pr(K,&o3[r],x+(size_t)r*nb,y); if(memcmp(&o1[r],&o3[r],4)) bad3++; }
    printf("mismatching rows opt: %d  pr30448: %d / %d\n",bad,bad3,rows);
    for(int rep=0;rep<3;rep++){
        double t0=now(); for(int it=0;it<20;it++) for(int r=0;r<rows;r++) ref(K,&o1[r],x+(size_t)r*nb,y); double t1=now();
        for(int it=0;it<20;it++) for(int r=0;r<rows;r++) opt(K,&o2[r],x+(size_t)r*nb,y); double t2=now();
        double t3=now(); for(int it=0;it<20;it++) for(int r=0;r<rows;r++) pr(K,&o3[r],x+(size_t)r*nb,y); double t4=now();
        printf("ref %.1f  opt %.1f (%.3fx)  pr30448 %.1f (%.3fx) us/matrix\n",(t1-t0)/20,(t2-t1)/20,(t1-t0)/(t2-t1),(t4-t3)/20,(t1-t0)/(t4-t3));
    }
}

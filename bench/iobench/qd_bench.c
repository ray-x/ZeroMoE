// O_DIRECT random-read throughput at a given queue depth (threads, each one synchronous preadv at a time)
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
static int fd; static size_t chunk, fsz; static int secs; static volatile int stop; static uint64_t bytes[64]; static uint64_t lat_us[64], nreq[64];
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void*work(void*a){ long id=(long)a; unsigned s=12345+id*7919; char*buf=aligned_alloc(4096,chunk);
  while(!stop){ s=s*1664525u+1013904223u; uint64_t r=((uint64_t)s<<20)^rand_r(&s); off_t off=(off_t)((r%((fsz-chunk)/4096))*4096);
    double t0=now(); ssize_t n=pread(fd,buf,chunk,off); double t1=now(); if(n<=0){perror("pread");break;} bytes[id]+=n; nreq[id]++; lat_us[id]+=(uint64_t)((t1-t0)*1e6); }
  free(buf); return 0; }
int main(int c,char**v){ if(c<5){fprintf(stderr,"usage: file chunk_kb qd secs\n");return 1;}
  fd=open(v[1],O_RDONLY|O_DIRECT); if(fd<0){perror("open");return 1;} chunk=(size_t)atol(v[2])*1024; int qd=atoi(v[3]); secs=atoi(v[4]);
  struct stat st; fstat(fd,&st); fsz=st.st_size; pthread_t th[64]; double t0=now();
  for(long i=0;i<qd;i++) pthread_create(&th[i],0,work,(void*)i); usleep(secs*1000000); stop=1; for(int i=0;i<qd;i++) pthread_join(th[i],0);
  double dt=now()-t0; uint64_t B=0,N=0,L=0; for(int i=0;i<qd;i++){B+=bytes[i];N+=nreq[i];L+=lat_us[i];}
  printf("%s chunk=%luKB qd=%d: %.2f GB/s  %.0f req/s  mean latency %.2f ms\n",v[1],chunk/1024,qd,B/dt/1e9,N/dt,N?L/1000.0/N:0); return 0; }

// Per-expert cost of reading into (a) freshly released pages (MADV_DONTNEED then O_DIRECT pread: kernel faults + zero-fills each page
// before the DMA), (b) pages that stayed resident (no release), (c) MADV_FREE'd pages not yet reclaimed. Single thread, one slot.
// build: cc -O2 -o dontneed_fault_bench bench/dontneed_fault_bench.c ; run: ./dontneed_fault_bench <file> [expert_bytes=7340032] [n=300]
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static double cpu(void){struct timespec t;clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(int argc,char**argv){
  const char*path=argv[1]; size_t sz=argc>2?strtoull(argv[2],0,10):7340032UL; int n=argc>3?atoi(argv[3]):300;
  sz=(sz+4095)/4096*4096;
  int fd=open(path,O_RDONLY|O_DIRECT); if(fd<0){perror("open");return 1;}
  struct stat st; fstat(fd,&st); size_t slots=(st.st_size/sz)-2; if(slots<(size_t)n) n=(int)slots;
  char*buf=mmap(0,sz,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
  const char*names[3]={"DONTNEED then read","resident (no release)","MADV_FREE then read"};
  for(int mode=0;mode<3;mode++){
    memset(buf,1,sz);
    double w=0,c=0;
    for(int i=0;i<n;i++){
      off_t off=(off_t)((i*2654435761u)%slots)*sz;
      if(mode==0) madvise(buf,sz,MADV_DONTNEED); else if(mode==2) madvise(buf,sz,MADV_FREE);
      double t0=now(),c0=cpu();
      ssize_t r=pread(fd,buf,sz,off); if(r!=(ssize_t)sz){perror("pread");return 1;}
      w+=now()-t0; c+=cpu()-c0;
    }
    printf("%-24s wall %.3f ms/expert, process CPU %.3f ms/expert (%.1f%% of wall)\n",names[mode],1e3*w/n,1e3*c/n,100*c/w);
  }
  return 0;
}

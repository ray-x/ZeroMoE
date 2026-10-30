/* LD_PRELOAD tracer for pinned host memory (cudaHostRegister) bugs: "Copy is larger than memobj size", HOST_MEMORY_ALREADY_REGISTERED.
 *   gcc -shared -fPIC -O1 bench/hostreg_trace.c -o hostreg_trace.so -ldl -lpthread
 *   HOSTREG_TRACE=/path/log LD_PRELOAD=$PWD/hostreg_trace.so ./build/zeromoe ...
 * Records every successful cudaHostRegister range with a short call stack, removes it on cudaHostUnregister, and logs
 *   FREE-OF-REGISTERED  free(p) of memory inside a range that was never unregistered (a stale registration over reused addresses)
 *   REGISTER-OVERLAP    a registration that failed with 712 and the registration it overlaps (both stacks)
 * Frames print as module(+offset): resolve with addr2line -Cfie <module> <offset>. Logging is capped. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdarg.h>
#define MAXR 400000
#define NBT 7
typedef struct { uintptr_t a, n; void *bt[NBT]; int nbt; int live; } Rec;
static Rec *recs; static size_t nrec;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static unsigned char bloom[1 << 22]; /* one bit per (addr >> 20) hashed: fast reject for free() */
static int fd = -2, nlog = 0;
static int (*real_reg)(void *, size_t, unsigned);
static int (*real_unreg)(void *);
extern void __libc_free(void *); /* glibc: avoids dlsym() (which allocates) inside free() before this library is initialised */
static __thread int in_hook;
static void logf_(const char *fmt, ...) {
  if (fd == -2) { const char *p = getenv("HOSTREG_TRACE"); fd = p ? open(p, O_WRONLY | O_CREAT | O_APPEND, 0644) : -1; }
  if (fd < 0 || nlog > 400) return;
  char b[1024]; va_list ap; (void)ap;
  __builtin_va_list v; __builtin_va_start(v, fmt); int n = vsnprintf(b, sizeof b, fmt, v); __builtin_va_end(v);
  if (n > 0) { (void)!write(fd, b, n > (int)sizeof b ? sizeof b : n); }
}
static void bt_(void **bt, int nbt) { if (fd < 0) return; backtrace_symbols_fd(bt, nbt, fd); }
static inline unsigned h_(uintptr_t a) { uintptr_t x = a >> 20; return (unsigned)((x * 0x9E3779B1u) >> 10) & ((1u << 25) - 1); }
static void setbits(uintptr_t a, size_t n) { for (uintptr_t x = a >> 20; x <= (a + n - 1) >> 20; ++x) { unsigned h = h_(x << 20); bloom[h >> 3] |= 1u << (h & 7); } }
static int maybe(uintptr_t a) { unsigned h = h_(a); return bloom[h >> 3] & (1u << (h & 7)); }
__attribute__((constructor)) static void init(void) {
  real_reg = dlsym(RTLD_NEXT, "cudaHostRegister"); real_unreg = dlsym(RTLD_NEXT, "cudaHostUnregister");
  recs = calloc(MAXR, sizeof(Rec)); void *w[2]; backtrace(w, 2); /* warm up libgcc */
}
int cudaHostRegister(void *ptr, size_t size, unsigned flags) {
  if (!real_reg) real_reg = dlsym(RTLD_NEXT, "cudaHostRegister");
  int r = real_reg(ptr, size, flags);
  if (in_hook || !recs) return r;
  in_hook = 1;
  void *bt[NBT + 1]; int nbt = backtrace(bt, NBT + 1);
  pthread_mutex_lock(&mu);
  if (r == 0) {
    if (nrec < MAXR) { Rec *x = &recs[nrec++]; x->a = (uintptr_t)ptr; x->n = size; x->nbt = nbt > 1 ? nbt - 1 : 0; memcpy(x->bt, bt + 1, sizeof(void *) * x->nbt); x->live = 1; setbits(x->a, size); }
  } else if (r == 712) {
    for (size_t i = 0; i < nrec; ++i) if (recs[i].live && (uintptr_t)ptr < recs[i].a + recs[i].n && recs[i].a < (uintptr_t)ptr + size) {
      logf_("REGISTER-OVERLAP new [%p,+%zu) overlaps live [%p,+%zu)\n new registered from:\n", ptr, size, (void *)recs[i].a, (size_t)recs[i].n); ++nlog;
      bt_(bt + 1, nbt - 1); logf_(" existing registered from:\n"); bt_(recs[i].bt, recs[i].nbt); break; }
  }
  pthread_mutex_unlock(&mu); in_hook = 0; return r;
}
int cudaHostUnregister(void *ptr) {
  if (!real_unreg) real_unreg = dlsym(RTLD_NEXT, "cudaHostUnregister");
  int r = real_unreg(ptr);
  if (!in_hook && recs && r == 0) { pthread_mutex_lock(&mu); for (size_t i = nrec; i-- > 0;) if (recs[i].live && recs[i].a == (uintptr_t)ptr) { recs[i].live = 0; break; } pthread_mutex_unlock(&mu); }
  return r;
}
void free(void *p) {
  if (p && !in_hook && recs && maybe((uintptr_t)p)) {
    in_hook = 1; pthread_mutex_lock(&mu);
    for (size_t i = 0; i < nrec; ++i) if (recs[i].live && (uintptr_t)p >= recs[i].a && (uintptr_t)p < recs[i].a + recs[i].n) {
      void *bt[NBT + 1]; int nbt = backtrace(bt, NBT + 1);
      logf_("FREE-OF-REGISTERED free(%p) inside live registration [%p,+%zu)\n freed from:\n", p, (void *)recs[i].a, (size_t)recs[i].n); ++nlog;
      bt_(bt + 1, nbt - 1); logf_(" registered from:\n"); bt_(recs[i].bt, recs[i].nbt); break; }
    pthread_mutex_unlock(&mu); in_hook = 0;
  }
  __libc_free(p);
}

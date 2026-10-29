/* LD_PRELOAD helper: print a backtrace on SIGABRT/SIGSEGV without ptrace (ggml's own abort backtrace needs gdb to attach).
 *   gcc -shared -fPIC -O1 bench/abort_backtrace.c -o abort_backtrace.so
 *   LD_PRELOAD=$PWD/abort_backtrace.so ./build/zeromoe ...      resolve addresses with: addr2line -Cfie ./build/zeromoe <addr>
 */
#define _GNU_SOURCE
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
static void handler(int sig) {
  void *frames[64];
  static const char hdr[] = "\n[abort_backtrace] fatal signal, frames (module(+offset)):\n";
  write(2, hdr, sizeof hdr - 1);
  int n = backtrace(frames, 64);
  backtrace_symbols_fd(frames, n, 2);
  signal(sig, SIG_DFL);
  raise(sig);
}
__attribute__((constructor)) static void init(void) {
  struct sigaction sa;
  memset(&sa, 0, sizeof sa);
  sa.sa_handler = handler;
  sa.sa_flags = SA_NODEFER | SA_RESETHAND;
  sigaction(SIGABRT, &sa, NULL);
  sigaction(SIGSEGV, &sa, NULL);
}

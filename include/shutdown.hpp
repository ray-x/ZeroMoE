// shutdown.hpp — cooperative graceful shutdown on SIGINT/SIGTERM.
//
// A signal handler can only safely touch async-signal-safe primitives
// (std::atomic, write(2), _exit(2)) — it must never allocate, lock a mutex,
// or call most C++/iostream machinery. So the handler does the bare minimum
// (set a flag, print a fixed notice via raw write(), and force-exit on a
// second signal) and the actual cleanup happens back on the main thread at
// the next safe checkpoint (top of the REPL loop / top of each decode-loop
// iteration). Reaching that checkpoint lets the function return normally,
// which lets every RAII destructor (StreamingEngine, CudaManager,
// GpuTopology, LockFreePrefetcher, ...) run and free host + GPU memory
// properly instead of relying on the OS/driver to reclaim everything at
// abrupt process death.
//
// Honest limitation: a decode step can take many seconds (CPU-bound expert
// FFN), and there's no cancellation point inside that computation — the
// first Ctrl-C is only checked at the next token/prompt boundary, not
// instantly. A second Ctrl-C (or SIGTERM) force-exits immediately for
// anyone who doesn't want to wait out the current step.
#pragma once
#include <atomic>
#include <csignal>
#include <unistd.h>

inline std::atomic<bool> g_shutdown_requested{false};
inline std::atomic<int> g_shutdown_signal_count{0};

inline void shutdown_signal_handler(int) {
    int count = g_shutdown_signal_count.fetch_add(1, std::memory_order_relaxed) + 1;
    g_shutdown_requested.store(true, std::memory_order_relaxed);
    if (count == 1) {
        static const char msg[] =
            "\n[Shutdown requested -- finishing current step, then cleaning up. "
            "Press Ctrl-C again to force quit immediately.]\n";
        ssize_t written = write(STDERR_FILENO, msg, sizeof(msg) - 1);
        (void)written;  // nothing sane to do with a failed write from a signal handler
    } else {
        // Impatient second signal — skip cleanup entirely rather than risk
        // hanging inside it. _exit() is async-signal-safe (unlike exit()).
        _exit(130);  // 128 + SIGINT, conventional shell exit code
    }
}

inline void install_shutdown_handlers() {
    std::signal(SIGINT, shutdown_signal_handler);
    std::signal(SIGTERM, shutdown_signal_handler);
}

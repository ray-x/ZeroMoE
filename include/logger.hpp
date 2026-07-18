// logger.hpp — leveled logging, optionally redirected to a file.
//
// Default behavior (no config) is unchanged from before: everything still
// goes to stderr. Setting `log_path` in the TOML config (or --log-path on
// the CLI) redirects it to a file instead; `log_level` filters by severity
// so routine per-token/per-expert diagnostics (cache hits/misses, VRAM
// promotions, CPU dequant fallback notices) don't flood the terminal during
// normal chat use but are still available on demand (log_level = "DEBUG").
// log_level = "VERBOSE" goes one step further: high-volume structural traces
// (e.g. per-layer hidden-state norms across the full model depth) that are
// too dense even for routine DEBUG troubleshooting, on by design only when
// actively bisecting a numerical bug.
#pragma once
#include <cstdio>
#include <cstdarg>
#include <string>
#include <mutex>
#include <sstream>

// VERBOSE sits below DEBUG: even more granular than the existing per-token/
// per-expert DEBUG diagnostics (cache hits/misses, VRAM promotions) --
// intended for high-volume structural traces (e.g. per-layer hidden-state
// norms across all ~79 layers) that would be unreadable noise during normal
// DEBUG-level troubleshooting but are exactly what's needed when actively
// bisecting a numerical bug layer-by-layer.
enum class LogLevel { VERBOSE = 0, DEBUG = 1, INFO = 2, WARN = 3, ERROR = 4 };

inline LogLevel log_level_from_string(const std::string& s) {
    if (s == "VERBOSE") return LogLevel::VERBOSE;
    if (s == "DEBUG") return LogLevel::DEBUG;
    if (s == "WARN")  return LogLevel::WARN;
    if (s == "ERROR") return LogLevel::ERROR;
    return LogLevel::INFO;
}

class Logger {
public:
    static Logger& instance() {
        static Logger inst;
        return inst;
    }

    // Called once at startup after config is parsed. `path` empty (the
    // default) keeps logging on stderr, matching prior behavior exactly.
    void init(const std::string& path, const std::string& level) {
        std::lock_guard<std::mutex> lock(mutex_);
        threshold_ = log_level_from_string(level);
        if (!path.empty()) {
            FILE* f = std::fopen(path.c_str(), "w");
            if (f) {
                file_ = f;
                owns_file_ = true;
            } else {
                std::fprintf(stderr, "[LOGGER] Could not open log_path '%s' — falling back to stderr\n",
                             path.c_str());
            }
        }
    }

    ~Logger() {
        if (owns_file_ && file_) std::fclose(file_);
    }

    void log(LogLevel lvl, const char* fmt, ...) {
        if (lvl < threshold_) return;
        std::lock_guard<std::mutex> lock(mutex_);
        FILE* out = file_ ? file_ : stderr;
        std::fputs(level_prefix(lvl), out);
        va_list args;
        va_start(args, fmt);
        std::vfprintf(out, fmt, args);
        va_end(args);
        std::fflush(out);
    }

    // Lets a hot call site skip building an expensive log message (string
    // concatenation, formatting) entirely when it wouldn't be emitted anyway
    // — log()'s own threshold check happens AFTER the caller already paid to
    // build the message, which matters for call sites that fire per-token /
    // per-expert (thousands of times a decode) rather than once at startup.
    bool would_log(LogLevel lvl) const { return lvl >= threshold_; }

private:
    Logger() = default;
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    static const char* level_prefix(LogLevel lvl) {
        switch (lvl) {
            case LogLevel::VERBOSE: return "[VERBOSE] ";
            case LogLevel::DEBUG: return "[DEBUG] ";
            case LogLevel::WARN:  return "[WARN] ";
            case LogLevel::ERROR: return "[ERROR] ";
            default:              return "";
        }
    }

    FILE* file_ = nullptr;
    bool owns_file_ = false;
    LogLevel threshold_ = LogLevel::INFO;
    std::mutex mutex_;
};

#define LOG_VERBOSE(...) Logger::instance().log(LogLevel::VERBOSE, __VA_ARGS__)
#define LOG_DEBUG(...) Logger::instance().log(LogLevel::DEBUG, __VA_ARGS__)
#define LOG_INFO(...)  Logger::instance().log(LogLevel::INFO,  __VA_ARGS__)
#define LOG_WARN(...)  Logger::instance().log(LogLevel::WARN,  __VA_ARGS__)
#define LOG_ERROR(...) Logger::instance().log(LogLevel::ERROR, __VA_ARGS__)

// Stream-style logging for call sites built around chained `<<` (the way
// most of this codebase's startup/diagnostic dumps are already written) —
// lets them route through the Logger (respecting log_path/log_level) without
// restructuring into printf-style format strings. Buffers into a string and
// flushes as one Logger::log() call when the temporary is destroyed at the
// end of the full expression, e.g.:
//   LOG_STREAM_INFO() << "[GRAPH] Hidden: " << c.hidden << "\n";
class LogStream {
public:
    explicit LogStream(LogLevel lvl) : level_(lvl) {}
    ~LogStream() {
        std::string s = buf_.str();
        if (!s.empty()) Logger::instance().log(level_, "%s", s.c_str());
    }
    LogStream(const LogStream&) = delete;
    LogStream& operator=(const LogStream&) = delete;

    template <typename T>
    LogStream& operator<<(const T& v) { buf_ << v; return *this; }
    // Support std::endl and other stream manipulators.
    LogStream& operator<<(std::ostream& (*manip)(std::ostream&)) { buf_ << manip; return *this; }

private:
    LogLevel level_;
    std::ostringstream buf_;
};

#define LOG_STREAM_VERBOSE() LogStream(LogLevel::VERBOSE)
#define LOG_STREAM_DEBUG() LogStream(LogLevel::DEBUG)
#define LOG_STREAM_INFO()  LogStream(LogLevel::INFO)
#define LOG_STREAM_WARN()  LogStream(LogLevel::WARN)
#define LOG_STREAM_ERROR() LogStream(LogLevel::ERROR)

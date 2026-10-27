// Per-request performance record.
//
// One RequestStats is filled in by generate_text() for every request and emitted as a single JSON line when the
// request ends (also on an early return, with status "error"). It is the one source of truth for per-request numbers:
// the bench mode, tools/perf_report.py and any future server read this instead of scraping log lines.
//
// Where it goes:
//   * the log, as one "[STATS] {json}" line (always);
//   * a file, one JSON object per line (JSONL), when --stats-file PATH is set;
//   * in memory, via Sink::last(), for in-process callers (the bench mode).
//
// Field meanings follow llama.cpp's llama_perf_context / server "timings" so numbers are comparable:
//   prompt_tokens / prompt_cached_tokens   prompt size and the part served from the prompt-state cache
//   prefill_ms, prefill_tok_s              time to process the uncached prompt tokens
//   decode_setup_ms                        decode-graph setup this request paid before its first token (Qwen graph;
//                                          absent for families that do not separate it)
//   ttft_ms                                request start to first generated token
//   generated_tokens                       tokens produced
//   decode_tok_s                           steady decode speed: 1000 / mean(step ms) over steps >= 3
//   decode_peak_tok_s                      best 32-step window
//   decode_p50_ms / decode_p95_ms          step-time percentiles over steps >= 3
//   decode_wall_tok_s                      generated_tokens / (end of prefill to end of request), setup included
//   total_ms                               whole request
// Anything model- or engine-specific goes into "extra" through note().
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace zeromoe::stats {

constexpr int kSchema = 1;
constexpr size_t kSkipSteps = 3;     // warm-up steps excluded from the steady decode figures
constexpr size_t kPeakWindow = 32;

struct RequestStats {
  std::string status = "error";      // "ok" once the request finished normally
  uint64_t seq = 0;                  // request number within this process, from 1
  std::string model;                 // file name of the model (set once by the runner)
  std::string family;                // graph/model family name, empty for the legacy path
  int prompt_tokens = 0;
  int prompt_cached_tokens = 0;
  double prefill_ms = -1;
  double decode_setup_ms = -1;       // < 0: not measured
  double ttft_ms = -1;
  int generated_tokens = 0;
  double decode_wall_ms = -1;
  double total_ms = -1;
  // Wall-clock instants (Unix epoch, ms): request start, end of prefill, request end. They let an outside monitor
  // attribute machine noise to the decode window of one request.
  double start_unix_ms = -1, decode_start_unix_ms = -1, end_unix_ms = -1;
  std::vector<float> step_ms;        // one entry per decode step (loop iteration time)
  std::map<std::string, double> extra;

  // ---- derived -----------------------------------------------------------------------------------------------
  double prefill_tok_s() const {
    const int fresh = prompt_tokens - prompt_cached_tokens;
    return (prefill_ms > 0 && fresh > 0) ? fresh * 1000.0 / prefill_ms : -1;
  }
  double decode_tok_s() const {
    if (step_ms.size() <= kSkipSteps) return -1;
    double sum = 0;
    for (size_t i = kSkipSteps; i < step_ms.size(); ++i) sum += step_ms[i];
    return sum > 0 ? (step_ms.size() - kSkipSteps) * 1000.0 / sum : -1;
  }
  double decode_peak_tok_s() const {
    if (step_ms.size() < kSkipSteps + kPeakWindow) return -1;
    double best = 0, window = 0;
    for (size_t i = kSkipSteps; i < step_ms.size(); ++i) {
      window += step_ms[i];
      if (i >= kSkipSteps + kPeakWindow) window -= step_ms[i - kPeakWindow];
      if (i >= kSkipSteps + kPeakWindow - 1 && window > 0) best = std::max(best, kPeakWindow * 1000.0 / window);
    }
    return best > 0 ? best : -1;
  }
  double decode_percentile_ms(double q) const {
    if (step_ms.size() <= kSkipSteps) return -1;
    std::vector<float> v(step_ms.begin() + kSkipSteps, step_ms.end());
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<size_t>(q * (v.size() - 1) + 0.5))];
  }
  // Decode speed in n equal consecutive parts of the steps after warm-up; empty unless each part has >= 20 steps.
  std::vector<double> decode_buckets(int n) const {
    std::vector<double> out;
    const size_t usable = step_ms.size() > kSkipSteps ? step_ms.size() - kSkipSteps : 0;
    const size_t per = n > 0 ? usable / static_cast<size_t>(n) : 0;
    if (per < 20) return out;
    for (int b = 0; b < n; ++b) {
      double sum = 0;
      for (size_t i = 0; i < per; ++i) sum += step_ms[kSkipSteps + b * per + i];
      out.push_back(sum > 0 ? per * 1000.0 / sum : -1);
    }
    return out;
  }
  double decode_wall_tok_s() const {
    return (decode_wall_ms > 0 && generated_tokens > 0) ? generated_tokens * 1000.0 / decode_wall_ms : -1;
  }

  std::string to_json() const;
};

namespace detail {
inline void put_num(std::string &s, const char *key, double v, int digits = 3) {
  char buf[64];
  if (v < 0 || !std::isfinite(v)) {
    std::snprintf(buf, sizeof buf, "\"%s\":null", key);
  } else {
    std::snprintf(buf, sizeof buf, "\"%s\":%.*f", key, digits, v);
  }
  if (!s.empty() && s.back() != '{') s += ',';
  s += buf;
}
inline void put_int(std::string &s, const char *key, long long v) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "\"%s\":%lld", key, v);
  if (!s.empty() && s.back() != '{') s += ',';
  s += buf;
}
inline std::string escape(const std::string &in) {
  std::string out;
  for (unsigned char c : in) {
    if (c == '"' || c == '\\') { out += '\\'; out += static_cast<char>(c); }
    else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); out += b; }
    else out += static_cast<char>(c);
  }
  return out;
}
inline void put_str(std::string &s, const char *key, const std::string &v) {
  if (!s.empty() && s.back() != '{') s += ',';
  s += "\"" + std::string(key) + "\":\"" + escape(v) + "\"";
}
}  // namespace detail

inline std::string RequestStats::to_json() const {
  std::string s = "{";
  detail::put_int(s, "schema", kSchema);
  detail::put_str(s, "status", status);
  detail::put_int(s, "seq", static_cast<long long>(seq));
  detail::put_str(s, "model", model);
  detail::put_str(s, "family", family);
  detail::put_int(s, "prompt_tokens", prompt_tokens);
  detail::put_int(s, "prompt_cached_tokens", prompt_cached_tokens);
  detail::put_num(s, "prefill_ms", prefill_ms);
  detail::put_num(s, "prefill_tok_s", prefill_tok_s(), 2);
  detail::put_num(s, "decode_setup_ms", decode_setup_ms);
  detail::put_num(s, "ttft_ms", ttft_ms);
  detail::put_int(s, "generated_tokens", generated_tokens);
  detail::put_num(s, "decode_tok_s", decode_tok_s(), 2);
  detail::put_num(s, "decode_peak_tok_s", decode_peak_tok_s(), 2);
  detail::put_num(s, "decode_p50_ms", decode_percentile_ms(0.50));
  detail::put_num(s, "decode_p95_ms", decode_percentile_ms(0.95));
  detail::put_num(s, "decode_wall_ms", decode_wall_ms);
  detail::put_num(s, "decode_wall_tok_s", decode_wall_tok_s(), 2);
  detail::put_num(s, "total_ms", total_ms);
  detail::put_num(s, "start_unix_ms", start_unix_ms, 0);
  detail::put_num(s, "decode_start_unix_ms", decode_start_unix_ms, 0);
  detail::put_num(s, "end_unix_ms", end_unix_ms, 0);
  const auto buckets = decode_buckets(6);
  if (!buckets.empty()) {
    s += ",\"decode_bucket_tok_s\":[";
    for (size_t i = 0; i < buckets.size(); ++i) {
      char b[32];
      std::snprintf(b, sizeof b, "%s%.2f", i ? "," : "", buckets[i]);
      s += b;
    }
    s += "]";
  }
  s += ",\"extra\":{";
  bool first = true;
  for (const auto &[k, v] : extra) {
    if (!first) s += ',';
    first = false;
    char buf[96];
    std::snprintf(buf, sizeof buf, ":%.6g", v);
    s += "\"" + detail::escape(k) + "\"" + buf;
  }
  s += "}}";
  return s;
}

inline double unix_ms_now() {
  using namespace std::chrono;
  return static_cast<double>(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());
}

// Process-wide destination for finished records and for notes made while a request runs.
class Sink {
 public:
  static Sink &instance() {
    static Sink s;
    return s;
  }
  void set_file(const std::string &path) {
    std::lock_guard<std::mutex> g(mu_);
    path_ = path;
  }
  void set_model(const std::string &model) {
    std::lock_guard<std::mutex> g(mu_);
    model_ = model;
  }
  std::string model() {
    std::lock_guard<std::mutex> g(mu_);
    return model_;
  }
  uint64_t next_seq() { return ++seq_; }

  // note() may be called from anywhere while a request runs; the values are attached to that request's record.
  void note(const std::string &key, double value) {
    std::lock_guard<std::mutex> g(mu_);
    notes_[key] = value;
  }
  std::map<std::string, double> take_notes() {
    std::lock_guard<std::mutex> g(mu_);
    return std::exchange(notes_, {});
  }

  // Writes the record to the file (if configured) and keeps it for last(). Returns the JSON line.
  std::string publish(const RequestStats &r) {
    const std::string json = r.to_json();
    std::lock_guard<std::mutex> g(mu_);
    last_ = r;
    have_last_ = true;
    const std::string path = path_;
    if (!path.empty()) {
      std::ofstream f(path, std::ios::app);
      if (f) f << json << '\n';
    }
    return json;
  }
  bool last(RequestStats &out) {
    std::lock_guard<std::mutex> g(mu_);
    if (!have_last_) return false;
    out = last_;
    return true;
  }

 private:
  std::mutex mu_;
  std::string path_, model_;
  std::map<std::string, double> notes_;
  RequestStats last_;
  bool have_last_ = false;
  std::atomic<uint64_t> seq_{0};
};

// Engine code that knows something about the current request (for example how much of an arena a prefill released)
// calls this; the value appears under "extra" in that request's record.
inline void note(const std::string &key, double value) { Sink::instance().note(key, value); }

}  // namespace zeromoe::stats

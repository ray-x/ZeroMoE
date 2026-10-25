// `zeromoe --bench`: run a fixed set of prompts through the normal request path in ONE process and summarise the
// per-request records (core/request_stats.hpp).
//
//   zeromoe --config C --bench --bench-prompt bench/prompts/short.txt --bench-prompt bench/prompts/mid.txt
//           --bench-prompt bench/prompts/long.txt [--bench-ngen 200] [--bench-reps 3] [--bench-warmup 1]
//           [--bench-long 1500] [--bench-out result.json]
//
// It loads the model once, then feeds the REPL a scripted queue instead of stdin, so chat templates, tokenization,
// prefill, decode-graph setup and teardown are exactly the code a user's request runs. Like llama-bench it
// reports prefill (pp) and decode (tg) speed with min/max over repetitions; unlike llama-bench it uses real text
// (a MoE model's speed depends on which experts the text routes to; random tokens would not reflect that) and also
// reports what a request pays around the model: time to first token and the decode-graph setup.
//
// Queue: `warmup` requests (not reported), then for each prompt `reps` requests of --bench-ngen tokens, then (with
// --bench-long) two requests of the first prompt generating --bench-long tokens, reported as per-bucket speeds so
// drift over a long generation is visible. Bench mode forces: full-length generation (ZEROMOE_BENCH_IGNORE_STOP),
// no per-token printing or VRAM sampling (ZEROMOE_BENCH_SILENT, as llama-bench does) and no prompt-state cache (a
// repeated prompt must be prefilled every time).
//
// Random-token mode (llama-bench's tests): --bench-random-pp N (repeatable) processes a prompt of N random token ids;
// --bench-random-tg N (repeatable) starts from one random token and generates N tokens, feeding a random token at every
// step instead of the sampled one (teacher forcing, no sampling). Ids come from a fixed seed (--bench-seed, default 1),
// so runs are reproducible and comparable with `llama-bench -p N -n N`. The two modes cannot be mixed in one process.
// Random ids route to random experts, which a hot-expert placement tuned on real text does not favour, so these
// numbers are far below real-text numbers for a MoE model; report both.
//
// The process does not watch the machine. A caller that needs a quiet-machine guarantee (tools/perf_report.py)
// checks the wall-clock instants in each request record against its own load samples.
#pragma once

#include "request_stats.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <unistd.h>
#include <sstream>
#include <string>
#include <vector>

namespace zeromoe::bench {

struct Plan {
  bool enabled = false;
  std::vector<std::string> prompt_files;
  int ngen = 200;
  int reps = 3;
  int warmup = 1;
  int long_ngen = 0;
  int long_reps = 2;
  std::vector<int> random_pp, random_tg;
  unsigned seed = 1;
  std::string out;
};

inline Plan &plan() {
  static Plan p;
  return p;
}

// Initialised during static initialisation, i.e. at process start.
inline const std::chrono::steady_clock::time_point g_process_start = std::chrono::steady_clock::now();

class Session {
 public:
  explicit Session(const Plan &p) : plan_(p) {}
  bool enabled() const { return plan_.enabled; }

  // Reads the prompt files and builds the queue. Returns an empty string on success, an error message otherwise.
  std::string prepare(bool &prompt_cache_flag, int vocab) {
    random_ = !plan_.random_pp.empty() || !plan_.random_tg.empty();
    if (random_ && !plan_.prompt_files.empty()) return "--bench-random-* cannot be combined with --bench-prompt";
    if (!random_ && plan_.prompt_files.empty())
      return "--bench needs --bench-prompt FILE (real text) or --bench-random-pp/-tg N (random tokens)";
    if (plan_.ngen < 8 || plan_.reps < 1) return "--bench-ngen must be at least 8 and --bench-reps at least 1";
    if (random_) {
      if (vocab < 2) return "the model vocabulary size is unknown; cannot draw random tokens";
      dir_ = (std::filesystem::temp_directory_path() / ("zeromoe_bench_" + std::to_string(getpid()))).string();
      std::filesystem::create_directories(dir_);
      std::mt19937 rng(plan_.seed);
      std::uniform_int_distribution<int> pick(0, vocab - 1);
      auto draw = [&](int n) {
        std::vector<int> v(n);
        for (auto &t : v) t = pick(rng);
        return v;
      };
      std::vector<Item> tests;
      for (int n : plan_.random_pp) {
        if (n < 8) return "--bench-random-pp N needs N >= 8";
        for (int r = 1; r <= plan_.reps; ++r) tests.push_back({"pp" + std::to_string(n), Kind::Test, r, "x", 2, draw(n), draw(2)});
      }
      for (int n : plan_.random_tg) {
        if (n < 8) return "--bench-random-tg N needs N >= 8";
        for (int r = 1; r <= plan_.reps; ++r) tests.push_back({"tg" + std::to_string(n), Kind::Test, r, "x", n, draw(1), draw(n)});
      }
      for (int i = 0; i < plan_.warmup; ++i) {
        Item w = tests.front();
        w.kind = Kind::Warmup;
        w.rep = 0;
        queue_.push_back(w);
      }
      for (auto &t : tests) {
        if (labels_.empty() || std::find(labels_.begin(), labels_.end(), t.label) == labels_.end()) labels_.push_back(t.label);
        queue_.push_back(std::move(t));
      }
      // The engine reads a per-request token stream from <dir>/request-<k>.tokens (k counts requests from 0) and
      // skips sampling for forced steps.
      setenv("ZEROMOE_TEACHER_FORCE_REQUEST_DIR", dir_.c_str(), 1);
      setenv("ZEROMOE_TEACHER_FORCE_SKIP_SAMPLING", "1", 1);
      setenv("ZEROMOE_BENCH_IGNORE_STOP", "1", 1);
      setenv("ZEROMOE_BENCH_SILENT", "1", 1);
      prompt_cache_flag = false;
      startup_s_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - g_process_start).count();
      total_ = queue_.size();
      return "";
    }
    for (const auto &file : plan_.prompt_files) {
      std::ifstream f(file);
      if (!f) return "cannot read bench prompt " + file;
      std::stringstream ss;
      ss << f.rdbuf();
      std::string text;
      for (char c : ss.str())
        if (c != '\n' && c != '\r') text += c;  // one turn per line, as the REPL reads them
      if (text.empty()) return "bench prompt is empty: " + file;
      prompts_.push_back({std::filesystem::path(file).stem().string(), text});
      labels_.push_back(prompts_.back().label);
    }
    setenv("ZEROMOE_BENCH_IGNORE_STOP", "1", 1);
    setenv("ZEROMOE_BENCH_SILENT", "1", 1);
    prompt_cache_flag = false;
    for (int i = 0; i < plan_.warmup; ++i)
      queue_.push_back({"warmup", Kind::Warmup, 0, prompts_[0].text, std::min(plan_.ngen, 64)});
    for (const auto &p : prompts_)
      for (int r = 1; r <= plan_.reps; ++r) queue_.push_back({p.label, Kind::Test, r, p.text, plan_.ngen});
    if (plan_.long_ngen > 0)
      for (int r = 1; r <= plan_.long_reps; ++r)
        queue_.push_back({"long_decode", Kind::Long, r, prompts_[0].text, plan_.long_ngen});
    startup_s_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - g_process_start).count();
    total_ = queue_.size();
    return "";
  }

  bool next(std::string &text, int &ngen) {
    if (queue_.empty()) return false;
    current_ = queue_.front();
    queue_.pop_front();
    text = current_.text;
    ngen = current_.ngen;
    if (random_) {
      write_ids(dir_ + "/prompt.txt", current_.prompt_ids);
      write_ids(dir_ + "/request-" + std::to_string(handed_out_) + ".tokens", current_.force_ids);
      setenv("ZEROMOE_PROMPT_TOKEN_INPUT_FILE", (dir_ + "/prompt.txt").c_str(), 1);
    }
    ++handed_out_;
    std::cout << "\n[BENCH] request " << (total_ - queue_.size()) << "/" << total_ << ": " << current_.label
              << " rep " << current_.rep << ", " << current_.ngen << " tokens\n";
    return true;
  }

  // Call right after generate_text() returned for the item handed out by next().
  void record() {
    zeromoe::stats::RequestStats r;
    Done d{current_, {}, false};
    if (zeromoe::stats::Sink::instance().last(r)) {
      d.stats = r;
      const bool prompt_ok = current_.prompt_ids.empty() || r.prompt_tokens == static_cast<int>(current_.prompt_ids.size());
      d.valid = r.status == "ok" && r.generated_tokens == current_.ngen && r.prompt_cached_tokens == 0 && prompt_ok &&
                (current_.ngen <= 8 || r.step_ms.size() > zeromoe::stats::kSkipSteps);
    }
    done_.push_back(std::move(d));
  }

  // Writes the summary (plan.out, if set) and prints a short table.
  void finish() {
    const std::string json = summary_json();
    if (!plan_.out.empty()) {
      std::ofstream f(plan_.out);
      if (f) f << json << '\n';
      else std::cerr << "[BENCH] cannot write " << plan_.out << "\n";
    }
    print_table();
  }

 private:
  enum class Kind { Warmup, Test, Long };
  struct Item {
    std::string label;
    Kind kind;
    int rep;
    std::string text;
    int ngen;
    std::vector<int> prompt_ids = {};  // random mode: exact prompt token ids
    std::vector<int> force_ids = {};   // random mode: token fed at each decode step
  };
  struct Prompt {
    std::string label, text;
  };
  struct Done {
    Item item;
    zeromoe::stats::RequestStats stats;
    bool valid;
  };

  struct Spread {
    double median = -1, min = -1, max = -1;
  };
  static Spread spread(std::vector<double> v) {
    v.erase(std::remove_if(v.begin(), v.end(), [](double x) { return x < 0; }), v.end());
    if (v.empty()) return {};
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return {n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]), v.front(), v.back()};
  }
  static void put(std::string &s, const char *key, const Spread &sp, int digits) {
    char b[160];
    if (sp.median < 0) std::snprintf(b, sizeof b, "\"%s\":null", key);
    else std::snprintf(b, sizeof b, "\"%s\":{\"median\":%.*f,\"min\":%.*f,\"max\":%.*f}", key, digits, sp.median, digits, sp.min, digits, sp.max);
    if (s.back() != '{') s += ',';
    s += b;
  }
  static void write_ids(const std::string &path, const std::vector<int> &ids) {
    std::ofstream f(path);
    for (size_t i = 0; i < ids.size(); ++i) f << (i ? " " : "") << ids[i];
    f << '\n';
  }
  static std::string quote(const std::string &in) { return "\"" + zeromoe::stats::detail::escape(in) + "\""; }

  // Per-bucket decode speed over the steps after the warm-up steps.
  static std::vector<double> buckets(const zeromoe::stats::RequestStats &r, int n_buckets) {
    std::vector<double> out;
    const size_t usable = r.step_ms.size() > zeromoe::stats::kSkipSteps ? r.step_ms.size() - zeromoe::stats::kSkipSteps : 0;
    const size_t per = usable / n_buckets;
    if (per < 20) return out;
    for (int b = 0; b < n_buckets; ++b) {
      double sum = 0;
      for (size_t i = 0; i < per; ++i) sum += r.step_ms[zeromoe::stats::kSkipSteps + b * per + i];
      out.push_back(sum > 0 ? per * 1000.0 / sum : -1);
    }
    return out;
  }

  std::string summary_json() const {
    std::string s = "{";
    zeromoe::stats::detail::put_int(s, "schema", zeromoe::stats::kSchema);
    zeromoe::stats::detail::put_str(s, "kind", "bench");
    zeromoe::stats::detail::put_str(s, "mode", random_ ? "random" : "text");
    if (random_) zeromoe::stats::detail::put_int(s, "seed", plan_.seed);
    zeromoe::stats::detail::put_num(s, "startup_s", startup_s_, 1);
    s += ",\"bench\":{";
    zeromoe::stats::detail::put_int(s, "ngen", plan_.ngen);
    zeromoe::stats::detail::put_int(s, "reps", plan_.reps);
    zeromoe::stats::detail::put_int(s, "warmup", plan_.warmup);
    zeromoe::stats::detail::put_int(s, "long_ngen", plan_.long_ngen);
    s += "},\"tests\":[";
    std::vector<double> medians;
    bool first_test = true;
    for (const auto &label : labels_) {
      std::vector<double> pre, dec, peak, ttft, setup, wall, p50, p95;
      int n = 0, invalid = 0, prompt_tokens = 0;
      for (const auto &d : done_) {
        if (d.item.kind != Kind::Test || d.item.label != label) continue;
        if (!d.valid) { ++invalid; continue; }
        ++n;
        prompt_tokens = d.stats.prompt_tokens;
        pre.push_back(d.stats.prefill_tok_s());
        dec.push_back(d.stats.decode_tok_s());
        peak.push_back(d.stats.decode_peak_tok_s());
        ttft.push_back(d.stats.ttft_ms);
        setup.push_back(d.stats.decode_setup_ms);
        wall.push_back(d.stats.decode_wall_tok_s());
        p50.push_back(d.stats.decode_percentile_ms(0.5));
        p95.push_back(d.stats.decode_percentile_ms(0.95));
      }
      if (!first_test) s += ',';
      first_test = false;
      std::string t = "{";
      zeromoe::stats::detail::put_str(t, "label", label);
      zeromoe::stats::detail::put_int(t, "prompt_tokens", prompt_tokens);
      zeromoe::stats::detail::put_int(t, "n", n);
      zeromoe::stats::detail::put_int(t, "invalid", invalid);
      put(t, "prefill_tok_s", spread(pre), 2);
      put(t, "decode_tok_s", spread(dec), 2);
      put(t, "decode_peak_tok_s", spread(peak), 2);
      put(t, "decode_p50_ms", spread(p50), 3);
      put(t, "decode_p95_ms", spread(p95), 3);
      put(t, "ttft_ms", spread(ttft), 1);
      put(t, "decode_setup_ms", spread(setup), 1);
      put(t, "decode_wall_tok_s", spread(wall), 2);
      t += "}";
      s += t;
      const Spread sp = spread(pre);
      if (sp.median >= 0 && prompt_tokens >= 8) medians.push_back(sp.median);
    }
    s += "]";
    if (!medians.empty()) {
      double sum = 0;
      for (double m : medians) sum += m;
      zeromoe::stats::detail::put_num(s, "prefill_avg_tok_s", sum / medians.size(), 2);
    }
    // long decode
    std::vector<double> ldec;
    std::vector<std::vector<double>> lb;
    int ln = 0, linvalid = 0, ltokens = 0;
    for (const auto &d : done_) {
      if (d.item.kind != Kind::Long) continue;
      if (!d.valid) { ++linvalid; continue; }
      ++ln;
      ltokens = d.stats.generated_tokens;
      ldec.push_back(d.stats.decode_tok_s());
      auto b = buckets(d.stats, 6);
      if (!b.empty()) lb.push_back(b);
    }
    if (ln + linvalid > 0) {
      s += ",\"long_decode\":{";
      zeromoe::stats::detail::put_int(s, "tokens", ltokens);
      zeromoe::stats::detail::put_int(s, "n", ln);
      zeromoe::stats::detail::put_int(s, "invalid", linvalid);
      put(s, "decode_tok_s", spread(ldec), 2);
      if (!lb.empty()) {
        s += ",\"bucket_tok_s\":[";
        std::vector<double> med;
        for (size_t i = 0; i < lb[0].size(); ++i) {
          std::vector<double> col;
          for (const auto &b : lb) col.push_back(b[i]);
          med.push_back(spread(col).median);
          char buf[32];
          std::snprintf(buf, sizeof buf, "%s%.2f", i ? "," : "", med.back());
          s += buf;
        }
        s += "]";
        zeromoe::stats::detail::put_num(s, "bucket_tokens", static_cast<double>(ltokens - 3) / 6.0, 0);
        if (med.front() > 0 && med.back() > 0) {
          char buf[48];
          std::snprintf(buf, sizeof buf, ",\"last_vs_first_pct\":%.1f", (med.back() / med.front() - 1.0) * 100.0);
          s += buf;
        }
      }
      s += "}";
    }
    s += ",\"requests\":[";
    bool first = true;
    for (const auto &d : done_) {
      if (!first) s += ',';
      first = false;
      std::string j = d.stats.to_json();
      s += "{\"label\":" + quote(d.item.label) + ",\"kind\":" +
           quote(d.item.kind == Kind::Warmup ? "warmup" : d.item.kind == Kind::Long ? "long" : "test") +
           ",\"rep\":" + std::to_string(d.item.rep) + ",\"valid\":" + (d.valid ? "true" : "false") + "," + j.substr(1);
    }
    s += "]}";
    return s;
  }

  void print_table() const {
    std::cout << "\n[BENCH] label        prompt  prefill tok/s   decode tok/s (min-max)   peak   ttft ms   setup ms   n/invalid\n";
    for (const auto &label : labels_) {
      std::vector<double> pre, dec, peak, ttft, setup;
      int n = 0, invalid = 0, tokens = 0;
      for (const auto &d : done_) {
        if (d.item.kind != Kind::Test || d.item.label != label) continue;
        if (!d.valid) { ++invalid; continue; }
        ++n;
        tokens = d.stats.prompt_tokens;
        pre.push_back(d.stats.prefill_tok_s());
        dec.push_back(d.stats.decode_tok_s());
        peak.push_back(d.stats.decode_peak_tok_s());
        ttft.push_back(d.stats.ttft_ms);
        setup.push_back(d.stats.decode_setup_ms);
      }
      const Spread a = spread(pre), b = spread(dec), c = spread(peak), t = spread(ttft), u = spread(setup);
      // "-" where a figure does not apply (decode of a 2-token pp run, prefill of a 1-token tg prompt)
      auto f = [](double v, const char *fmt) {
        char buf[32];
        if (v < 0) return std::string("-");
        std::snprintf(buf, sizeof buf, fmt, v);
        return std::string(buf);
      };
      const std::string pre_s = tokens >= 8 ? f(a.median, "%.1f") : "-";
      const std::string dec_s = b.median < 0 ? "-" : f(b.median, "%.2f") + " (" + f(b.min, "%.2f") + "-" + f(b.max, "%.2f") + ")";
      char line[256];
      std::snprintf(line, sizeof line, "[BENCH] %-12s %6d  %13s   %-24s   %5s   %7s   %8s   %d/%d\n", label.c_str(), tokens,
                    pre_s.c_str(), dec_s.c_str(), f(c.median, "%.1f").c_str(), f(t.median, "%.0f").c_str(),
                    f(u.median, "%.0f").c_str(), n, invalid);
      std::cout << line;
    }
    std::cout << std::flush;
  }

  Plan plan_;
  std::vector<Prompt> prompts_;
  std::vector<std::string> labels_;
  bool random_ = false;
  std::string dir_;
  size_t handed_out_ = 0;
  std::deque<Item> queue_;
  Item current_{};
  std::vector<Done> done_;
  size_t total_ = 0;
  double startup_s_ = 0;
};

}  // namespace zeromoe::bench

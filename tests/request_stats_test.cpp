// Tests for core/request_stats.hpp and core/bench_mode.hpp (header-only, no GPU, no model).
//   c++ -std=c++23 -O1 -Wall -Wextra -Icore tests/request_stats_test.cpp -o /tmp/request_stats_test
//   /tmp/request_stats_test [summary.json]     (writes the bench summary there so a caller can validate the JSON)
#include "bench_mode.hpp"
#include "request_stats.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>

using namespace zeromoe;

static int g_fail = 0;
#define CHECK(cond)                                                              \
  do {                                                                           \
    if (!(cond)) {                                                               \
      std::printf("FAIL line %d: %s\n", __LINE__, #cond);                        \
      ++g_fail;                                                                  \
    }                                                                            \
  } while (0)
static bool near(double a, double b, double tol = 1e-6) { return std::fabs(a - b) <= tol; }

static stats::RequestStats make(int prompt, double prefill_ms, int steps, float step_ms, float warm_ms = 100.f) {
  stats::RequestStats r;
  r.status = "ok";
  r.prompt_tokens = prompt;
  r.prefill_ms = prefill_ms;
  r.ttft_ms = prefill_ms + 50;
  r.generated_tokens = steps;
  r.decode_setup_ms = 400;
  r.decode_wall_ms = steps * step_ms + 400;
  r.total_ms = prefill_ms + r.decode_wall_ms;
  for (int i = 0; i < steps; ++i) r.step_ms.push_back(i < 3 ? warm_ms : step_ms);
  return r;
}

int main(int argc, char **argv) {
  // --- derived figures with known answers
  {
    auto r = make(1000, 2000.0, 203, 20.f);  // 203 steps: 3 warm-up + 200 at 20 ms
    CHECK(near(r.prefill_tok_s(), 500.0));
    CHECK(near(r.decode_tok_s(), 50.0));           // 1000 / 20
    CHECK(near(r.decode_peak_tok_s(), 50.0));
    CHECK(near(r.decode_percentile_ms(0.5), 20.0));
    CHECK(near(r.decode_wall_tok_s(), 203 * 1000.0 / (203 * 20.0 + 400.0)));
    r.prompt_cached_tokens = 400;                  // only the uncached part counts toward prefill speed
    CHECK(near(r.prefill_tok_s(), 600 * 1000.0 / 2000.0));
  }
  // --- peak is the best 32-step window, average is not
  {
    auto r = make(10, 100.0, 103, 20.f);
    for (int i = 60; i < 92; ++i) r.step_ms[i] = 10.f;   // one fast stretch of exactly 32 steps
    CHECK(near(r.decode_peak_tok_s(), 100.0));
    CHECK(r.decode_tok_s() > 50.0 && r.decode_tok_s() < 60.0);
    CHECK(near(r.decode_percentile_ms(0.0), 10.0));
    CHECK(near(r.decode_percentile_ms(1.0), 20.0));
  }
  // --- too few steps: figures are unavailable, not zero
  {
    auto r = make(10, 100.0, 3, 20.f);
    CHECK(r.decode_tok_s() < 0);
    CHECK(r.decode_peak_tok_s() < 0);
    auto r2 = make(10, 100.0, 20, 20.f);
    CHECK(r2.decode_tok_s() > 0);
    CHECK(r2.decode_peak_tok_s() < 0);               // fewer than 3 + 32 steps
    const std::string j = r.to_json();
    CHECK(j.find("\"decode_tok_s\":null") != std::string::npos);
  }
  // --- JSON shape and escaping
  {
    auto r = make(28, 1200.0, 203, 22.f);
    r.model = "weird \"name\"\\x";
    r.extra["soft_released_gib"] = 2.49;
    const std::string j = r.to_json();
    CHECK(j.front() == '{' && j.back() == '}');
    CHECK(j.find("\"schema\":1") != std::string::npos);
    CHECK(j.find("\"model\":\"weird \\\"name\\\"\\\\x\"") != std::string::npos);
    CHECK(j.find("\"soft_released_gib\":2.49") != std::string::npos);
    CHECK(j.find("\"decode_setup_ms\":400.000") != std::string::npos);
    stats::RequestStats none;
    CHECK(none.to_json().find("\"decode_setup_ms\":null") != std::string::npos);
    CHECK(none.status == "error");
  }
  // --- sink: notes are attached once, last() returns the published record, the file gets one line per record
  {
    const std::string path = "/tmp/request_stats_test.jsonl";
    std::remove(path.c_str());
    auto &sink = stats::Sink::instance();
    sink.set_file(path);
    stats::note("a", 1.0);
    CHECK(sink.take_notes().size() == 1);
    CHECK(sink.take_notes().empty());
    sink.publish(make(5, 50.0, 10, 10.f));
    sink.publish(make(6, 60.0, 10, 10.f));
    stats::RequestStats last;
    CHECK(sink.last(last) && last.prompt_tokens == 6);
    std::ifstream f(path);
    int lines = 0;
    for (std::string l; std::getline(f, l);) ++lines;
    CHECK(lines == 2);
    sink.set_file("");
  }
  // --- bench session end to end with synthetic requests
  {
    std::ofstream("/tmp/bench_prompt_a.txt") << "first line\nsecond line\n";
    std::ofstream("/tmp/bench_prompt_b.txt") << "another prompt\n";
    bench::Plan p;
    p.enabled = true;
    p.prompt_files = {"/tmp/bench_prompt_a.txt", "/tmp/bench_prompt_b.txt"};
    p.ngen = 100;
    p.reps = 3;
    p.warmup = 1;
    p.long_ngen = 400;
    p.long_reps = 2;
    p.out = argc > 1 ? argv[1] : "/tmp/bench_summary.json";
    bench::Session s(p);
    bool prompt_cache = true;
    CHECK(s.prepare(prompt_cache, 1000).empty());
    CHECK(!prompt_cache);                             // bench mode must switch the prompt-state cache off
    CHECK(std::string(std::getenv("ZEROMOE_BENCH_SILENT")) == "1");
    std::string text;
    int ngen = 0, n = 0;
    auto &sink = stats::Sink::instance();
    while (s.next(text, ngen)) {
      ++n;
      CHECK(text.find('\n') == std::string::npos);    // one turn per line
      // 100-token requests run at 50 tok/s, 400-token requests slow down by 20% over the run
      stats::RequestStats r = make(text == "another prompt" ? 300 : 28, 1000.0, ngen, 20.f);
      if (ngen == 400)
        for (size_t i = 200; i < r.step_ms.size(); ++i) r.step_ms[i] = 25.f;
      sink.publish(r);
      s.record();
    }
    CHECK(n == 1 + 2 * 3 + 2);                        // warmup + 2 prompts x 3 reps + 2 long
    s.finish();
    std::ifstream f(p.out);
    std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    CHECK(json.find("\"kind\":\"bench\"") != std::string::npos);
    CHECK(json.find("\"label\":\"bench_prompt_a\"") != std::string::npos);
    CHECK(json.find("\"prefill_avg_tok_s\"") != std::string::npos);
    CHECK(json.find("\"bucket_tok_s\":[") != std::string::npos);
    CHECK(json.find("\"last_vs_first_pct\":-") != std::string::npos);   // the long run slows down
    CHECK(json.find("\"kind\":\"warmup\"") != std::string::npos);
  }
  // --- random-token mode (llama-bench's tests)
  {
    bench::Plan p;
    p.enabled = true;
    p.random_pp = {64};
    p.random_tg = {32};
    p.reps = 2;
    p.warmup = 1;
    p.seed = 7;
    p.out = "/tmp/bench_random_summary.json";
    bench::Session s(p);
    bool prompt_cache = true;
    CHECK(s.prepare(prompt_cache, 1000).empty());
    CHECK(!prompt_cache);
    CHECK(std::string(std::getenv("ZEROMOE_TEACHER_FORCE_SKIP_SAMPLING")) == "1");
    const std::string dir = std::getenv("ZEROMOE_TEACHER_FORCE_REQUEST_DIR");
    std::string text;
    int ngen = 0, k = 0;
    std::vector<int> prompt_sizes, ngens;
    auto &sink = stats::Sink::instance();
    auto count_ids = [](const std::string &path) {
      std::ifstream f(path);
      int n = 0, v = 0, mx = -1;
      while (f >> v) { ++n; mx = std::max(mx, v); }
      return std::pair<int, int>(n, mx);
    };
    while (s.next(text, ngen)) {
      const auto prompt = count_ids(std::string(std::getenv("ZEROMOE_PROMPT_TOKEN_INPUT_FILE")));
      const auto forced = count_ids(dir + "/request-" + std::to_string(k) + ".tokens");
      CHECK(forced.first == ngen);                     // one forced token per decode step
      CHECK(prompt.second < 1000 && forced.second < 1000);   // ids stay inside the vocabulary
      prompt_sizes.push_back(prompt.first);
      ngens.push_back(ngen);
      stats::RequestStats r = make(prompt.first, 100.0, ngen, 20.f);
      sink.publish(r);
      s.record();
      ++k;
    }
    CHECK(k == 1 + 2 + 2);                             // warmup + pp64 x2 + tg32 x2
    CHECK((prompt_sizes == std::vector<int>{64, 64, 64, 1, 1}));
    CHECK((ngens == std::vector<int>{2, 2, 2, 32, 32}));
    s.finish();
    std::ifstream f(p.out);
    std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    CHECK(json.find("\"mode\":\"random\"") != std::string::npos);
    CHECK(json.find("\"seed\":7") != std::string::npos);
    CHECK(json.find("\"label\":\"pp64\"") != std::string::npos);
    CHECK(json.find("\"label\":\"tg32\"") != std::string::npos);
    CHECK(json.find("\"valid\":false") == std::string::npos);
    // the same seed gives the same ids
    bench::Session s2(p);
    bool pc2 = true;
    CHECK(s2.prepare(pc2, 1000).empty());
    s2.next(text, ngen);
    const auto again = count_ids(std::string(std::getenv("ZEROMOE_PROMPT_TOKEN_INPUT_FILE")));
    CHECK(again.first == 64);
  }
  std::printf(g_fail ? "request_stats_test: %d FAILED\n" : "request_stats_test: all passed\n", g_fail);
  return g_fail ? 1 : 0;
}

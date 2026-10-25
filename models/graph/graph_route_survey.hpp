#pragma once

// Route-usage survey shared by every in-graph MoE family (MiMo, GLM/Ling, new families).
//
// The in-graph MoE path bypasses the engine's own expert-usage statistics, so a surveyed profile (the pin file
// that picks the VRAM-hot experts) has to be recorded from the graph's own route ids. This is a SIDE TAP: a node
// that reads the layer's route ids and counts them, whose output nothing consumes, so the ids and the nodes that
// use them keep their backend and values. It does add one small copy and one CPU node per layer, which is why it
// is off unless a record file is set (docs: README "Validating", docs/models/mimo26-flash.md).
//
// Decode and prefill are counted separately and written to two files, because one 512-token prefill chunk holds
// far more route slots than a decode step and would otherwise dominate a decode profile:
//   <file>          decode counts   ("layer expert count" lines, the pin-file format)
//   <file>.prefill  prefill counts
// Files are rewritten every kFlushEvery taps and at shutdown, so a teardown abort loses little.

#include "logger.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"

namespace zeromoe::graph {

class GraphRouteSurvey {
 public:
  static GraphRouteSurvey &get() {
    static GraphRouteSurvey r;
    return r;
  }
  bool enabled() const { return !path_.empty(); }

  // First call sizes the tables; later calls (graph rebuilds) are no-ops. An empty path keeps the survey off.
  void configure(const std::string &path, int n_layers, int first_layer, int n_experts) {
    std::lock_guard<std::mutex> lock(mu_);
    if (path.empty() || !path_.empty()) return;
    path_ = path;
    n_layers_ = n_layers; first_ = first_layer; n_experts_ = n_experts;
    decode_ = std::vector<std::atomic<uint64_t>>(size_t(n_layers) * size_t(n_experts));
    prefill_ = std::vector<std::atomic<uint64_t>>(size_t(n_layers) * size_t(n_experts));
    LOG_INFO("[ROUTE-SURVEY] recording expert routes to %s (+ .prefill); survey mode adds a CPU node per layer\n",
             path_.c_str());
  }

  // Adds the tap for `ids` ([topk, S] I32). Returns false (and records nothing) for another id type.
  bool tap(ggml_context *ctx, ggml_cgraph *graph, ggml_backend_sched_t sched, ggml_backend_t cpu,
           ggml_tensor *ids, int layer, bool decode) {
    if (!enabled() || !ids) return false;
    if (ids->type != GGML_TYPE_I32) {
      LOG_ERROR("[ROUTE-SURVEY] layer %d route ids are not I32; not recorded\n", layer);
      return false;
    }
    std::lock_guard<std::mutex> lock(mu_);
    sites_.push_back({this, layer, decode});
    ggml_tensor *sink = ggml_map_custom1(ctx, ids, &GraphRouteSurvey::count_cb, 1, &sites_.back());
    if (!sink) return false;
    ggml_backend_sched_set_tensor_backend(sched, sink, cpu);
    ggml_build_forward_expand(graph, sink);  // runs although nothing reads it
    return true;
  }

  // Graph destruction invalidates the tap sites; counts stay.
  void release_sites() {
    std::lock_guard<std::mutex> lock(mu_);
    sites_.clear();
  }

  void flush() const {
    if (path_.empty()) return;
    write(path_, decode_);
    write(path_ + ".prefill", prefill_);
  }

 private:
  struct Site { GraphRouteSurvey *self; int layer; bool decode; };

  static void count_cb(ggml_tensor *dst, const ggml_tensor *a, int ith, int, void *ud) {
    (void)dst;
    if (ith != 0) return;
    const Site &s = *static_cast<Site *>(ud);
    GraphRouteSurvey &r = *s.self;
    auto &table = s.decode ? r.decode_ : r.prefill_;
    for (int64_t i1 = 0; i1 < a->ne[1]; ++i1)
      for (int64_t i0 = 0; i0 < a->ne[0]; ++i0) {
        const int32_t id = *reinterpret_cast<const int32_t *>(static_cast<const char *>(a->data) + i1 * a->nb[1] + i0 * a->nb[0]);
        if (id >= 0 && id < r.n_experts_)
          table[size_t(s.layer) * size_t(r.n_experts_) + size_t(id)].fetch_add(1, std::memory_order_relaxed);
      }
    if (r.taps_.fetch_add(1, std::memory_order_relaxed) % kFlushEvery == kFlushEvery - 1) r.flush();
  }

  void write(const std::string &path, const std::vector<std::atomic<uint64_t>> &t) const {
    std::FILE *f = std::fopen(path.c_str(), "w");
    if (!f) {
      LOG_ERROR("[ROUTE-SURVEY] cannot write %s\n", path.c_str());
      return;
    }
    for (int l = first_; l < n_layers_; ++l)
      for (int e = 0; e < n_experts_; ++e)
        if (const uint64_t n = t[size_t(l) * size_t(n_experts_) + size_t(e)].load(std::memory_order_relaxed))
          std::fprintf(f, "%d %d %llu\n", l, e, static_cast<unsigned long long>(n));
    std::fclose(f);
  }

  static constexpr uint64_t kFlushEvery = 4096;
  std::string path_;
  int n_layers_ = 0, first_ = 0, n_experts_ = 0;
  std::vector<std::atomic<uint64_t>> decode_, prefill_;
  std::deque<Site> sites_;
  std::atomic<uint64_t> taps_{0};
  std::mutex mu_;
};

}  // namespace zeromoe::graph

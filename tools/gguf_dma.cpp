// tools/gguf_dma.cpp
//
// gguf_tool dma: A/B benchmark of the ACTUAL production read mechanism --
// io_uring-batched, O_DIRECT, concurrent submission via io_uring_backend.hpp
// (the SAME Ring class RAMSafetyManager::get_experts_batch's Phase B uses in
// the real engine) -- comparing, for a fixed number of experts per
// submission N:
//   1-combined : N ReadRequests, one per expert (whole gate+up+down slab
//                in a single pread), matching build_expert_read_requests'
//                unpacked_expert_format branch.
//   3-separate : 3N ReadRequests, gate/up/down as independent reads AT THE
//                SAME FILE OFFSETS (same contiguous per-expert block, just
//                split into 3 requests), matching what build_expert_read_
//                requests' packed branch submits (minus the "scattered
//                across 3 unrelated tensors" locality difference -- this
//                isolates REQUEST COUNT / io_uring queue depth as the only
//                variable, which the real-hardware investigation this tool
//                was built for narrowed the packed-vs-unpacked throughput
//                gap down to; see CLAUDE.md's "why is 3 DMA faster than 1
//                DMA" entry).
//
// Only meaningful against a genuinely unpacked-format model (gguf_tool pack
// --unpack + unpacked_expert_format=true) -- 1-combined is only valid where
// gate/up/down are physically contiguous, which per-expert packed tensors
// (blk.N.ffn_{gate,up,down}_exps.weight) generally are NOT (they're 3
// separate, far-apart tensors). If a given model has no contiguous experts
// at all, this tool still reports 3-separate throughput but skips the
// 1-combined arm and says why.
//
// Random access, spanning every layer the model has experts for -- NOT
// sequential -- so results reflect real decode-time access, not readahead-
// friendly scanning. Requires O_DIRECT (this tool's whole point is testing
// the production disk path, not a page-cache benchmark) -- exits early with
// a clear message if O_DIRECT can't be enabled on a given model's shard(s).
//
// Build (no CMake / no CUDA -- Linux only for a REAL run: io_uring_backend.hpp
// is a no-op stub on non-Linux, so this compiles everywhere but only
// exercises the real ring on Linux):
//   c++ -std=c++23 -O2 -Iinclude tools/gguf_tool.cpp tools/gguf_pack.cpp \
//       tools/gguf_io.cpp tools/gguf_verify.cpp tools/gguf_plan.cpp \
//       tools/gguf_dma.cpp -o build/gguf_tool
// Run:
//   ./build/gguf_tool dma <model.gguf|dir> [<model2> ...] \
//       [--experts N[,N2,...]] [--submissions M] [--seed S] [--workers W]

#include "gguf_common.hpp"
#include "io_uring_backend.hpp"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

// "blk.<L>.ffn_<proj>.<E>.weight" -> (L, E). Mirrors gguf_common.hpp's
// parse_blk_layer for the layer piece, then parses the trailing per-expert
// index the same way. `proj` is "ffn_gate" / "ffn_up" / "ffn_down".
bool parse_unpacked_tensor(const std::string &name, const char *proj, int &L,
                            int &E) {
  int layer;
  if (!gguf_tool::parse_blk_layer(name, layer))
    return false;
  // Skip past "blk.<L>." to the rest of the name.
  std::string prefix = "blk." + std::to_string(layer) + ".";
  if (name.rfind(prefix, 0) != 0)
    return false;
  std::string_view rest(name.c_str() + prefix.size(), name.size() - prefix.size());
  std::string want = std::string(proj) + ".";
  if (rest.rfind(want, 0) != 0)
    return false;
  size_t q = want.size();
  int ev = 0;
  bool any = false;
  while (q < rest.size() && std::isdigit((unsigned char)rest[q])) {
    ev = ev * 10 + (rest[q] - '0');
    any = true;
    ++q;
  }
  if (!any)
    return false;
  std::string_view suffix = rest.substr(q);
  if (suffix != ".weight")
    return false;
  L = layer;
  E = ev;
  return true;
}

// One candidate expert's on-disk geometry, resolved once up front (never
// re-looked-up inside a timed loop -- same discipline as gguf_io.cpp's
// existing A/B).
struct ExpertGeom {
  int layer = -1, expert = -1;
  int fd = -1;
  uint64_t g_off = 0, u_off = 0, d_off = 0;
  uint64_t gb = 0, ub = 0, db = 0;
  bool tight = false; // gate/up/down contiguous, same fd -- 1-combined eligible
};

std::vector<std::string> split_csv(const std::string &s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ','))
    if (!tok.empty())
      out.push_back(tok);
  return out;
}

// io_uring_backend::Ring::CAPACITY is only a member of the real (Linux)
// Ring -- the non-Linux stub (io_uring_backend.hpp's `#else` branch) has no
// such member, since it never sizes anything. Mirrored here as a plain
// constant so this file compiles identically on both platforms; keep in
// sync with io_uring_backend.hpp's own CAPACITY if that ever changes.
constexpr unsigned kRingCapacity = 64;

int usage(const char *prog) {
  fprintf(stderr,
          "usage: %s <model.gguf|dir> [<model2> ...] [--experts "
          "N[,N2,...]] [--submissions M] [--seed S] [--workers W]\n"
          "\n"
          "A/B of the REAL production read mechanism (io_uring, O_DIRECT, "
          "concurrent\n"
          "submission -- the same Ring class RAMSafetyManager::"
          "get_experts_batch's\n"
          "Phase B uses) at a fixed number of experts per submission N:\n"
          "  1-combined : N reads (whole gate+up+down slab each)\n"
          "  3-separate : 3N reads (gate/up/down as independent reads at "
          "the SAME\n"
          "               file offsets -- isolates request count / queue "
          "depth from\n"
          "               locality, which a genuinely scattered packed-"
          "format model\n"
          "               would confound)\n"
          "Random (layer, expert) picks spanning every layer with experts -- "
          "not\n"
          "sequential. Requires O_DIRECT; Linux only for a real io_uring "
          "run (this\n"
          "tool builds everywhere but reports 'ring unavailable' on non-"
          "Linux).\n"
          "\n"
          "  --experts N[,N2,...]  experts per submission to sweep "
          "(default 1,2,3,4,8,16;\n"
          "                        clamped to <=21 so 3-separate's 3N stays "
          "under\n"
          "                        io_uring_backend::Ring::CAPACITY=64)\n"
          "  --submissions M       submit_and_wait calls per (mode, N), "
          "interleaved\n"
          "                        combined/separate to cancel out drift "
          "(default 200)\n"
          "  --seed S              RNG seed (default 12345)\n"
          "  --workers W           io-wq worker pool size, register_iowq_"
          "workers\n"
          "                        (default 24, matching the decode "
          "thread's own ring)\n",
          prog);
  return 2;
}

} // namespace

namespace gguf_tool {

int cmd_dma(int argc, char **argv) {
  std::vector<std::string> models;
  std::vector<int> sweep = {1, 2, 3, 4, 8, 16};
  int submissions = 200, seed = 12345, workers = 24;
  bool sweep_overridden = false;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--experts" && i + 1 < argc) {
      sweep.clear();
      for (auto &tok : split_csv(argv[++i]))
        sweep.push_back(std::atoi(tok.c_str()));
      sweep_overridden = true;
    } else if (a == "--submissions" && i + 1 < argc) {
      submissions = std::atoi(argv[++i]);
    } else if (a == "--seed" && i + 1 < argc) {
      seed = std::atoi(argv[++i]);
    } else if (a == "--workers" && i + 1 < argc) {
      workers = std::atoi(argv[++i]);
    } else if (!a.empty() && a[0] == '-') {
      fprintf(stderr, "unknown arg: %s\n", a.c_str());
      return 2;
    } else {
      models.push_back(a);
    }
  }
  if (models.empty())
    return usage(argv[0]);

  // Ring::CAPACITY=64: 3-separate needs 3N<=64 -> N<=21.
  for (auto &n : sweep) {
    if (n > 21) {
      std::cerr << "[dma] --experts " << n
                << " > 21 would exceed Ring::CAPACITY=64 for the "
                   "3-separate arm (3x) -- clamping to 21\n";
      n = 21;
    }
    if (n < 1) n = 1;
  }
  std::sort(sweep.begin(), sweep.end());
  sweep.erase(std::unique(sweep.begin(), sweep.end()), sweep.end());

  io_uring_backend::Ring ring;
  bool ring_ok = ring.init();
  if (!ring_ok) {
    std::cerr << "[dma] io_uring Ring::init() failed -- old kernel, "
                 "non-Linux, or sandboxed environment. This tool's whole "
                 "point is measuring io_uring queue-depth behavior, so "
                 "there is nothing meaningful to run here. Verify on the "
                 "real Linux/CUDA box.\n";
    return 1;
  }
  ring.register_iowq_workers((unsigned)std::max(1, workers));
  std::cout << "[dma] io_uring ring ready, CAPACITY=" << kRingCapacity
            << ", io-wq workers=" << workers << "\n";
  if (!sweep_overridden) {
    std::cout << "[dma] experts-per-submission sweep (default): ";
    for (auto n : sweep) std::cout << n << ' ';
    std::cout << "\n";
  }

  int overall_rc = 0;

  for (auto &model_path : models) {
    std::cout << "\n=== model: " << model_path << " ===\n";
    GGUFFile model;
    if (!model.load_header(model_path)) {
      std::cerr << "[dma] FAIL: cannot load " << model_path << "\n";
      overall_rc = 1;
      continue;
    }
    int n_direct = model.enable_direct_io();
    if (n_direct == 0) {
      std::cerr << "[dma] FAIL: O_DIRECT could not be enabled on any shard "
                   "of "
                << model_path
                << " -- this tool measures the production O_DIRECT/io_uring "
                   "path specifically and has nothing meaningful to report "
                   "without it (check alignment / filesystem support).\n";
      overall_rc = 1;
      continue;
    }

    // Discover every (layer, expert) with a complete, resolvable
    // gate+up+down triple -- unpacked format first (this investigation's
    // actual model shape), falling back to packed if no unpacked tensors
    // exist at all so the tool still works against an ordinary packed
    // model (3-separate only, reported as such; see header comment).
    std::map<std::pair<int, int>, ExpertGeom> found; // (L,E) -> partial geom
    bool any_unpacked = false;
    for (auto &kv : model.tensors) {
      int L, E;
      if (parse_unpacked_tensor(kv.first, "ffn_gate", L, E)) {
        any_unpacked = true;
        auto &g = found[{L, E}];
        g.layer = L; g.expert = E;
        g.fd = model.fd_for_tensor(kv.second);
        g.g_off = kv.second.offset; g.gb = kv.second.size_bytes;
      } else if (parse_unpacked_tensor(kv.first, "ffn_up", L, E)) {
        any_unpacked = true;
        auto &g = found[{L, E}];
        g.layer = L; g.expert = E;
        g.u_off = kv.second.offset; g.ub = kv.second.size_bytes;
      } else if (parse_unpacked_tensor(kv.first, "ffn_down", L, E)) {
        any_unpacked = true;
        auto &g = found[{L, E}];
        g.layer = L; g.expert = E;
        g.d_off = kv.second.offset; g.db = kv.second.size_bytes;
      }
    }

    std::vector<ExpertGeom> candidates, combined_eligible;
    if (any_unpacked) {
      for (auto &kv : found) {
        ExpertGeom &g = kv.second;
        if (g.gb == 0 || g.ub == 0 || g.db == 0)
          continue; // one of the three projections missing -- incomplete
        int fd_u = model.fd_for_tensor(model.tensors.at(
            "blk." + std::to_string(g.layer) + ".ffn_up." +
            std::to_string(g.expert) + ".weight"));
        int fd_d = model.fd_for_tensor(model.tensors.at(
            "blk." + std::to_string(g.layer) + ".ffn_down." +
            std::to_string(g.expert) + ".weight"));
        g.tight = (fd_u == g.fd && fd_d == g.fd) &&
                  (g.u_off == g.g_off + g.gb) && (g.d_off == g.u_off + g.ub);
        candidates.push_back(g);
        if (g.tight) combined_eligible.push_back(g);
      }
      std::cout << "format: unpacked (per-expert). experts discovered: "
                << candidates.size() << ", 1-combined-eligible (tight): "
                << combined_eligible.size() << "\n";
    } else {
      // Packed fallback: blk.<L>.ffn_{gate,up,down}_exps.weight, per-expert
      // slice = tensor sliced by expert index. 1-combined only valid if the
      // three tensors happen to be physically interleaved (rare for a
      // standard Unsloth-style packed layout) -- checked per expert exactly
      // like engine.hpp's load_expert_slab does.
      std::map<int, std::tuple<const GGUFTensor*, const GGUFTensor*, const GGUFTensor*>> layers;
      for (auto &kv : model.tensors) {
        int L;
        if (!gguf_tool::parse_blk_layer(kv.first, L)) continue;
        if (kv.first.find(".ffn_gate_exps.weight") != std::string::npos)
          std::get<0>(layers[L]) = &kv.second;
        else if (kv.first.find(".ffn_up_exps.weight") != std::string::npos)
          std::get<1>(layers[L]) = &kv.second;
        else if (kv.first.find(".ffn_down_exps.weight") != std::string::npos)
          std::get<2>(layers[L]) = &kv.second;
      }
      // architecture_config.n_experts is NEVER populated by GGUFFile::
      // load_header() (that parsing only happens in the full engine's
      // GGUFGraphInitializer, which this lightweight tool doesn't run) --
      // infer expert count from the packed tensor's OWN shape instead,
      // same approach gguf_io.cpp's cmd_io already uses for the identical
      // reason. Computed per-layer (not once globally) since nothing
      // guarantees every layer shares the same expert count.
      for (auto &kv : layers) {
        auto [gM, uM, dM] = kv.second;
        if (!gM || !uM || !dM || gM->shape.empty()) continue;
        int n_exp = (int)gM->shape.back();
        if (n_exp <= 0) continue;
        uint64_t gb = gM->size_bytes / (uint64_t)n_exp;
        uint64_t ub = uM->size_bytes / (uint64_t)n_exp;
        uint64_t db = dM->size_bytes / (uint64_t)n_exp;
        int fd_g = model.fd_for_tensor(*gM), fd_u = model.fd_for_tensor(*uM),
            fd_d = model.fd_for_tensor(*dM);
        for (int E = 0; E < n_exp; ++E) {
          ExpertGeom g;
          g.layer = kv.first; g.expert = E; g.fd = fd_g;
          g.g_off = gM->offset + (uint64_t)E * gb; g.gb = gb;
          g.u_off = uM->offset + (uint64_t)E * ub; g.ub = ub;
          g.d_off = dM->offset + (uint64_t)E * db; g.db = db;
          g.tight = (fd_u == fd_g && fd_d == fd_g) &&
                    (g.u_off == g.g_off + gb) && (g.d_off == g.u_off + ub);
          candidates.push_back(g);
          if (g.tight) combined_eligible.push_back(g);
        }
      }
      std::cout << "format: packed (blk.N.ffn_*_exps.weight). experts "
                   "discovered: "
                << candidates.size() << ", 1-combined-eligible (interleaved "
                   "on disk): " << combined_eligible.size() << "\n";
    }

    if (candidates.empty()) {
      std::cerr << "[dma] FAIL: no complete gate+up+down expert found in "
                << model_path << "\n";
      overall_rc = 1;
      continue;
    }
    {
      std::vector<int> layer_set;
      for (auto &g : candidates) layer_set.push_back(g.layer);
      std::sort(layer_set.begin(), layer_set.end());
      layer_set.erase(std::unique(layer_set.begin(), layer_set.end()), layer_set.end());
      std::cout << "layers spanned: " << layer_set.size() << " (L"
                << layer_set.front() << "..L" << layer_set.back() << ")\n";
    }
    bool have_combined = !combined_eligible.empty();
    if (!have_combined) {
      std::cout << "[dma] no tight (contiguous) experts found -- skipping "
                   "1-combined arm entirely, reporting 3-separate only. "
                   "This model's gate/up/down are not adjacent on disk for "
                   "any expert this tool found.\n";
    }

    std::mt19937 rng((unsigned)seed);

    // ---- correctness check: combined bytes == concatenated separate bytes,
    // for the SAME on-disk data, before trusting any throughput number ----
    if (have_combined) {
      std::uniform_int_distribution<size_t> pick(0, combined_eligible.size() - 1);
      int check_n = std::min<int>(20, (int)combined_eligible.size());
      uint64_t max_total = 0;
      for (auto &g : combined_eligible) max_total = std::max(max_total, g.gb + g.ub + g.db);
      uint8_t *buf_combined = nullptr, *buf_separate = nullptr;
      if (posix_memalign((void **)&buf_combined, 4096, max_total) != 0 ||
          posix_memalign((void **)&buf_separate, 4096, max_total) != 0) {
        std::cerr << "[dma] FAIL: posix_memalign scratch\n";
        overall_rc = 1;
        continue;
      }
      int mismatches = 0;
      for (int i = 0; i < check_n; ++i) {
        const ExpertGeom &g = combined_eligible[pick(rng)];
        uint64_t total = g.gb + g.ub + g.db;
        ssize_t r1 = direct_io::pread_direct_aware(g.fd, true, model.direct_io_alignment,
                                                     buf_combined, total, g.g_off);
        ssize_t r2 = direct_io::pread_direct_aware(g.fd, true, model.direct_io_alignment,
                                                     buf_separate, g.gb, g.g_off);
        ssize_t r3 = direct_io::pread_direct_aware(g.fd, true, model.direct_io_alignment,
                                                     buf_separate + g.gb, g.ub, g.u_off);
        ssize_t r4 = direct_io::pread_direct_aware(g.fd, true, model.direct_io_alignment,
                                                     buf_separate + g.gb + g.ub, g.db, g.d_off);
        if (r1 != (ssize_t)total || r2 != (ssize_t)g.gb || r3 != (ssize_t)g.ub ||
            r4 != (ssize_t)g.db || std::memcmp(buf_combined, buf_separate, total) != 0) {
          mismatches++;
        }
      }
      free(buf_combined);
      free(buf_separate);
      std::cout << "correctness: " << (check_n - mismatches) << "/" << check_n
                << " combined-vs-separate reads byte-identical";
      if (mismatches) {
        std::cerr << "\n[dma] CRITICAL FAIL: " << mismatches
                  << " mismatch(es) -- something is wrong with the geometry "
                     "this tool computed; throughput numbers below would "
                     "not be comparing equivalent data. Skipping model.\n";
        overall_rc = 1;
        continue;
      }
      std::cout << "\n";
    }

    // ---- disjoint pools: the combined arm and the separate arm must NEVER
    // read the same on-disk byte range over the course of this run. O_DIRECT
    // bypasses the OS page cache, but NOT an SSD's own onboard controller
    // cache (many NVMe drives keep a small DRAM/HMB-backed read cache
    // independent of the host page cache) -- if both arms independently
    // sampled from the same pool, they would eventually both touch the same
    // expert, and whichever arm's read happened to land second could get an
    // unearned speed boost from the drive's own cache that has nothing to do
    // with 1-vs-3 requests. Splitting the eligible pool in half up front and
    // giving each arm exclusive ownership of its half removes the
    // possibility entirely, rather than arguing the odds are probably low.
    std::vector<ExpertGeom> pool_combined_arm, pool_separate_arm;
    if (have_combined) {
      std::vector<ExpertGeom> shuffled = combined_eligible;
      std::shuffle(shuffled.begin(), shuffled.end(), rng);
      size_t half = shuffled.size() / 2;
      if (half == 0) {
        std::cout << "[dma] only " << shuffled.size()
                  << " tight expert(s) found -- too few to split into "
                     "disjoint arms, disabling 1-combined for this model.\n";
        have_combined = false;
        pool_separate_arm = candidates;
      } else {
        pool_combined_arm.assign(shuffled.begin(), shuffled.begin() + (long)half);
        pool_separate_arm.assign(shuffled.begin() + (long)half, shuffled.end());
        std::cout << "disjoint pools: " << pool_combined_arm.size()
                  << " expert(s) reserved for 1-combined, "
                  << pool_separate_arm.size()
                  << " reserved for 3-separate (zero overlap between arms)\n";
      }
    } else {
      pool_separate_arm = candidates; // no combined arm running -- no overlap risk
    }

    std::cout << std::left << std::setw(8) << "experts" << std::setw(14)
              << "1-combined" << std::setw(14) << "3-separate" << std::setw(10)
              << "speedup" << "\n";

    for (int N : sweep) {
      std::uniform_int_distribution<size_t> pick_c(0, have_combined ? pool_combined_arm.size() - 1 : 0);
      std::uniform_int_distribution<size_t> pick_a(0, pool_separate_arm.size() - 1);

      double combined_bytes = 0, combined_ns = 0;
      double separate_bytes = 0, separate_ns = 0;
      bool combined_ok_all = true, separate_ok_all = true;

      std::vector<io_uring_backend::ReadRequest> reqs_c(N);
      std::vector<ssize_t> res_c(N);
      std::vector<io_uring_backend::ReadRequest> reqs_s(3 * N);
      std::vector<ssize_t> res_s(3 * N);

      // Scratch: sized to worst case (largest N * largest expert slab); one
      // shared buffer reused across submissions (this benchmark only cares
      // about read throughput, not the destination bytes afterward).
      uint64_t max_slab = 0;
      for (auto &g : candidates) max_slab = std::max(max_slab, g.gb + g.ub + g.db);
      uint8_t *scratch = nullptr;
      if (posix_memalign((void **)&scratch, 4096, (size_t)max_slab * (size_t)N + 4096) != 0) {
        std::cerr << "[dma] FAIL: posix_memalign scratch for N=" << N << "\n";
        overall_rc = 1;
        continue;
      }

      for (int s = 0; s < submissions; ++s) {
        // ---- interleaved: 1-combined submission first, then 3-separate,
        // each round -- so drift (thermal, NVMe queue state, background
        // activity) affects both arms roughly equally instead of biasing
        // whichever mode happens to run first/last overall. ----
        if (have_combined) {
          uint64_t off = 0;
          for (int i = 0; i < N; ++i) {
            const ExpertGeom &g = pool_combined_arm[pick_c(rng)];
            uint64_t total = g.gb + g.ub + g.db;
            reqs_c[i] = {g.fd, true, model.direct_io_alignment,
                         scratch + off, (size_t)total, g.g_off};
            off += total;
          }
          auto t0 = std::chrono::high_resolution_clock::now();
          bool ok = ring.submit_and_wait(reqs_c.data(), N, res_c.data());
          double ns = std::chrono::duration<double, std::nano>(
                          std::chrono::high_resolution_clock::now() - t0)
                          .count();
          if (!ok) { combined_ok_all = false; }
          else {
            for (int i = 0; i < N; ++i) {
              if (res_c[i] != (ssize_t)reqs_c[i].len) { combined_ok_all = false; break; }
              combined_bytes += (double)reqs_c[i].len;
            }
            combined_ns += ns;
          }
        }

        {
          uint64_t off = 0;
          for (int i = 0; i < N; ++i) {
            const ExpertGeom &g = pool_separate_arm[pick_a(rng)];
            reqs_s[3 * i + 0] = {g.fd, true, model.direct_io_alignment,
                                 scratch + off, (size_t)g.gb, g.g_off};
            off += g.gb;
            reqs_s[3 * i + 1] = {g.fd, true, model.direct_io_alignment,
                                 scratch + off, (size_t)g.ub, g.u_off};
            off += g.ub;
            reqs_s[3 * i + 2] = {g.fd, true, model.direct_io_alignment,
                                 scratch + off, (size_t)g.db, g.d_off};
            off += g.db;
          }
          auto t0 = std::chrono::high_resolution_clock::now();
          bool ok = ring.submit_and_wait(reqs_s.data(), 3 * N, res_s.data());
          double ns = std::chrono::duration<double, std::nano>(
                          std::chrono::high_resolution_clock::now() - t0)
                          .count();
          if (!ok) { separate_ok_all = false; }
          else {
            for (int i = 0; i < 3 * N; ++i) {
              if (res_s[i] != (ssize_t)reqs_s[i].len) { separate_ok_all = false; break; }
              separate_bytes += (double)reqs_s[i].len;
            }
            separate_ns += ns;
          }
        }
      }
      free(scratch);

      double c_gbps = combined_ns > 0 ? (combined_bytes / 1e9) / (combined_ns / 1e9) : 0;
      double s_gbps = separate_ns > 0 ? (separate_bytes / 1e9) / (separate_ns / 1e9) : 0;

      std::cout << std::left << std::setw(8) << N;
      if (have_combined && combined_ok_all)
        std::cout << std::setw(14) << (std::ostringstream() << std::fixed << std::setprecision(3) << c_gbps << " GB/s").str();
      else
        std::cout << std::setw(14) << (have_combined ? "SUBMIT-FAIL" : "n/a (no tight)");
      if (separate_ok_all)
        std::cout << std::setw(14) << (std::ostringstream() << std::fixed << std::setprecision(3) << s_gbps << " GB/s").str();
      else
        std::cout << std::setw(14) << "SUBMIT-FAIL";
      if (have_combined && combined_ok_all && separate_ok_all && s_gbps > 0)
        std::cout << std::setw(10) << (std::ostringstream() << std::fixed << std::setprecision(2) << (c_gbps / s_gbps) << "x").str();
      std::cout << "\n";
    }
  }

  return overall_rc;
}

} // namespace gguf_tool

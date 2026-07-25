// tools/gguf_io.cpp
//
// gguf_tool io: cross-platform (macOS + Linux) A/B benchmark of the
// per-expert READ pattern:
//   OLD = 3 scattered preads per expert  (original Unsloth packed *_exps
//   layout) NEW = 1 tight pread per expert
//
// Loads both files with the REAL GGUFFile loader, then simulates the
// decode-time expert-access pattern (random layer+expert "misses") and
// measures aggregate disk-read throughput for each layout. Same I/O mode is
// used for BOTH paths -- buffered reads with readahead disabled and the page
// cache bypassed -- so the comparison isolates the READ-PATTERN improvement
// (1 syscall + 1 contiguous request vs 3 syscalls + 3 scattered requests),
// which is exactly what the repack + single-pread engine change delivers.
// The additional O_DIRECT page-cache-bypass benefit is production-Linux-only
// and measured by the engine itself, not here.
//
// Cache bypass (so a large file's reads actually hit disk, not RAM):
//   macOS : fcntl(fd, F_NOCACHE)           -- pages aren't cached
//   Linux : posix_fadvise(FADV_RANDOM)     -- readahead off (whole file)
// gguf_loader.hpp provides a posix_fadvise no-op stub on macOS + the
// POSIX_FADV_* constants, so the SAME source compiles and runs on both.
//
// Build (no CMake / no CUDA -- runs on macOS AND Linux): see gguf_common.hpp
// (compiled together with the other gguf_*.cpp into build/gguf_tool).
//   # or via the wrapper: make bench-io   (or tools/run_gguf_tool.sh io)
// Run:
//   ./build/gguf_tool io <orig_packed.gguf|dir> <repacked.gguf|dir> [--reads N]
//   [--seed S] [--direct]
//
// IMPORTANT: use a LARGE real model (multi-GB). On a small fixture the whole
// file sits in RAM/disk cache and both paths look like memory speed (the A/B
// ratio is meaningless there even though the harness still runs correctly).

#include "gguf_common.hpp" // cmd_io decl + shared helpers (pulls gguf_loader.hpp + direct_io.hpp)
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <system_error>
#include <unistd.h>
#include <vector>

// Disable readahead (Linux) / mark pages non-cacheable (macOS) on every shard
// fd, so reads of a large file go to disk instead of being served from RAM.
static void bypass_page_cache(GGUFFile &f) {
  for (int fd : f.shard_fds) {
    if (fd < 0)
      continue;
#ifdef __APPLE__
    fcntl(fd, F_NOCACHE, 1);
#endif
    posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM); // no-op stub on macOS
  }
}

// Evict the just-read region from the page cache so neither A/B path
// benefits from pages the OTHER path brought in. FADV_RANDOM (set once in
// bypass_page_cache) only disables readahead; it does NOT stop the kernel
// from caching pages that were actually read, so without this the NEW path
// (run first) warms the cache for the exact same expert slabs the OLD path
// then reads -- flattening the A/B ratio toward 1 and making "10% faster"
// look like the whole story when really both paths read mostly from RAM.
// FADV_DONTNEED after the read is the standard way to keep the working set
// out of cache; no-op stub on macOS (see gguf_loader.hpp).
static void evict_after_read(int fd, uint64_t off, uint64_t len) {
  if (fd < 0 || len == 0)
    return;
  posix_fadvise(fd, (off_t)off, (off_t)len, POSIX_FADV_DONTNEED);
}

// "blk.<L>.ffn_*_exps.weight" -> L
static bool parse_packed_layer(const std::string &name, int &L) {
  if (name.rfind("blk.", 0) != 0)
    return false;
  size_t p = 4;
  int v = 0;
  bool any = false;
  while (p < name.size() && name[p] >= '0' && name[p] <= '9') {
    v = v * 10 + (name[p] - '0');
    any = true;
    ++p;
  }
  if (!any || p >= name.size() || name[p] != '.')
    return false;
  L = v;
  return true;
}

namespace gguf_tool {
int cmd_io(int argc, char **argv) {
  if (argc < 3) {
    fprintf(
        stderr,
        "usage: %s <orig_packed.gguf|dir> <repacked.gguf|dir> [--reads N] "
        "[--seed S] [--direct]\n"
        "\nA/B of per-expert read pattern: OLD=3 scattered preads (orig "
        "packed\n"
        "*_exps layout) vs NEW=1 tight pread (repacked layout). Both paths "
        "use\n"
        "buffered, readahead-off, cache-bypassed reads so the comparison\n"
        "isolates read-pattern. --direct adds a third line that re-reads "
        "the\n"
        "repacked shard through O_DIRECT/DMA (Linux only; the engine's "
        "production\n"
        "--direct-io path) -- the original shard can't use O_DIRECT (only "
        "32-byte\n"
        "aligned), so this is where the real DMA benefit shows up. Use a "
        "large\n"
        "real model file.\n",
        argv[0]);
    return 2;
  }
  std::string orig = argv[1], rep = argv[2];
  int reads = 4000, seed = 12345;
  bool use_direct = false; // --direct: add a third timed line that reads the
                           // REPACKED shard via O_DIRECT/DMA (the engine's
                           // production --direct-io path). The original shard
                           // can never use O_DIRECT (its tensors are only
                           // 32-byte aligned -- the whole reason gguf_tool pack
                           // exists), so OLD stays buffered regardless.
  for (int i = 3; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--reads" && i + 1 < argc)
      reads = std::atoi(argv[++i]);
    else if (a == "--seed" && i + 1 < argc)
      seed = std::atoi(argv[++i]);
    else if (a == "--direct")
      use_direct = true;
    else {
      fprintf(stderr, "unknown arg: %s\n", a.c_str());
      return 2;
    }
  }

  GGUFFile O, R;
  if (!O.load_header(orig)) {
    fprintf(stderr, "FAIL: cannot load orig %s\n", orig.c_str());
    return 1;
  }
  if (!R.load_header(rep)) {
    fprintf(stderr, "FAIL: cannot load repacked %s\n", rep.c_str());
    return 1;
  }
  bypass_page_cache(O);
  bypass_page_cache(R);

  // discover packed layers + n_experts from orig
  std::vector<int> layers;
  int n_exp = 0;
  for (auto &kv : O.tensors) {
    if (kv.first.find(".ffn_gate_exps.weight") != std::string::npos) {
      int L;
      if (parse_packed_layer(kv.first, L)) {
        layers.push_back(L);
        if (!n_exp && !kv.second.shape.empty())
          n_exp = (int)kv.second.shape.back();
      }
    }
  }
  if (layers.empty()) {
    std::cerr << "FAIL: no *_exps.weight tensors in orig. Is it already "
                 "unpacked, or not Unsloth-packed?\n";
    return 1;
  }
  std::sort(layers.begin(), layers.end());
  std::cout << "orig: " << layers.size()
            << " packed layer(s), n_experts=" << n_exp << ", "
            << O.tensors.size() << " tensors\n";

  auto find_exps = [&](int L, std::string_view proj) -> const GGUFTensor * {
    auto it = O.tensors.find("blk." + std::to_string(L) + "." +
                             std::string(proj) + "_exps.weight");
    return it == O.tensors.end() ? nullptr : &it->second;
  };
  int L0 = layers[0];
  auto gE = find_exps(L0, "ffn_gate"), uE = find_exps(L0, "ffn_up"),
       dE = find_exps(L0, "ffn_down");
  if (!gE || !uE || !dE) {
    std::cerr << "FAIL: layer 0 missing gate/up/down_exps\n";
    return 1;
  }
  uint64_t gb = gE->size_bytes / static_cast<uint64_t>(n_exp);
  uint64_t ub = uE->size_bytes / static_cast<uint64_t>(n_exp);
  uint64_t db = dE->size_bytes / static_cast<uint64_t>(n_exp);
  uint64_t per_expert = gb + ub + db;
  std::cout << "per-expert slab: gate=" << gb << " up=" << ub << " down=" << db
            << " total=" << per_expert << " B (" << std::fixed
            << std::setprecision(3) << (per_expert / 1e6) << " MB)\n";

  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> pl(0, (int)layers.size() - 1),
      pe(0, n_exp - 1);
  std::vector<std::pair<int, int>> picks;
  picks.reserve(reads);
  for (int i = 0; i < reads; ++i)
    picks.push_back({layers[pl(rng)], pe(rng)});

  // Pre-resolve every read to (fd, offset, len) OUTSIDE the timed loops, so
  // the measurement is pure pread I/O -- not unordered_map lookups or string
  // construction, which would swamp disk time on a small file and add
  // unequal per-read overhead (OLD needs 3 lookups/expert, NEW needs 1).
  struct Read {
    int fd;
    uint64_t off;
    uint64_t len;
  };
  std::vector<Read> old_plan, new_plan;
  old_plan.reserve(reads * 3);
  new_plan.reserve(reads);
  bool tight = true;
  // Capture the FIRST non-contiguous expert so the "NOT tight" message can
  // say WHY (out-of-order projections = stale repack; split fds = expert
  // straddled an output-shard boundary; ordered-but-gappy = padding bug)
  // instead of leaving the user to guess. All the diagnostics below come
  // from these.
  int fail_L = -1, fail_E = -1;
  uint64_t f_g = 0, f_u = 0, f_d = 0, f_wug = 0, f_wdu = 0;
  int f_gfd = -1, f_ufd = -1, f_dfd = -1;
  for (auto &p : picks) {
    int L = p.first, E = p.second;
    auto *g = find_exps(L, "ffn_gate");
    auto *u = find_exps(L, "ffn_up");
    auto *d = find_exps(L, "ffn_down");
    if (!g || !u || !d) {
      std::cerr << "FAIL: orig missing blk." << L << ".*_exps\n";
      return 1;
    }
    old_plan.push_back({static_cast<int>(O.fd_for_tensor(*g)),
                        g->offset + static_cast<uint64_t>(E) * gb, gb});
    old_plan.push_back({static_cast<int>(O.fd_for_tensor(*u)),
                        u->offset + static_cast<uint64_t>(E) * ub, ub});
    old_plan.push_back({static_cast<int>(O.fd_for_tensor(*d)),
                        d->offset + static_cast<uint64_t>(E) * db, db});

    auto git = R.tensors.find("blk." + std::to_string(L) + ".ffn_gate." +
                              std::to_string(E) + ".weight");
    auto uit = R.tensors.find("blk." + std::to_string(L) + ".ffn_up." +
                              std::to_string(E) + ".weight");
    auto dit = R.tensors.find("blk." + std::to_string(L) + ".ffn_down." +
                              std::to_string(E) + ".weight");
    if (git == R.tensors.end() || uit == R.tensors.end() ||
        dit == R.tensors.end()) {
      std::cerr << "FAIL: repacked missing blk." << L << ".ffn_*." << E
                << ".weight\n";
      return 1;
    }
    // tight-contiguity check on EVERY entry (cheap, untimed): a faithful
    // mirror of engine.hpp load_expert_slab's per-expert `tight` guard --
    // up must follow gate by EXACTLY gate's own size_bytes, down must
    // follow up by up's own size_bytes, and all three must share one shard
    // fd. Uses each repacked tensor's OWN size (not gb/ub derived from the
    // orig packed tensor) so the check stays correct even for arches whose
    // per-expert numel isn't a clean multiple of the quant block size.
    uint64_t want_ug = git->second.size_bytes;
    uint64_t want_du = uit->second.size_bytes;
    int g_fd = static_cast<int>(R.fd_for_tensor(git->second));
    int u_fd = static_cast<int>(R.fd_for_tensor(uit->second));
    int d_fd = static_cast<int>(R.fd_for_tensor(dit->second));
    bool this_tight = u_fd == g_fd && d_fd == g_fd &&
                      uit->second.offset == git->second.offset + want_ug &&
                      dit->second.offset == uit->second.offset + want_du;
    if (!this_tight && tight) {
      tight = false;
      fail_L = L;
      fail_E = E;
      f_g = git->second.offset;
      f_u = uit->second.offset;
      f_d = dit->second.offset;
      f_wug = want_ug;
      f_wdu = want_du;
      f_gfd = g_fd;
      f_ufd = u_fd;
      f_dfd = d_fd;
    }
    new_plan.push_back({g_fd, git->second.offset, per_expert});
  }
  std::cout << "tight layout: "
            << (tight ? "OK (single-read path eligible)"
                      : "NOT tight -- engine would use the 3-read fallback")
            << '\n';
  if (!tight) {
    std::cout << "  first non-contiguous expert: blk." << fail_L << " expert "
              << fail_E << '\n';
    std::cout << "    gate.off=" << f_g << " up.off=" << f_u
              << " down.off=" << f_d << '\n';
    std::cout << "    up-gate delta=" << static_cast<int64_t>(f_u - f_g)
              << " (want " << f_wug << "), down-up delta="
              << static_cast<int64_t>(f_d - f_u) << " (want " << f_wdu << ")\n";
    std::cout << "    fds gate/up/down=" << f_gfd << '/' << f_ufd << '/'
              << f_dfd
              << (f_gfd == f_ufd && f_ufd == f_dfd ? " (same shard)"
                                                    : " (SPLIT across shards)")
              << '\n';
    bool ordered = f_g < f_u && f_u < f_d;
    std::cout << "    likely cause: "
              << (!ordered
                      ? "projections NOT in gate<up<down order -- this file "
                        "was repacked by an OLDER gguf_tool pack that sorted "
                        "tensors by NAME (down<gate<up). Re-run the CURRENT "
                        "tools/gguf_tool pack over the source shard."
                      : (f_gfd != f_ufd || f_ufd != f_dfd
                             ? "this expert's projections landed in different "
                               "OUTPUT shards -- re-repack with a larger "
                               "--max-shard-gb."
                             : "offsets ordered but gappy -- unexpected "
                               "padding; re-run the current gguf_tool pack."))
              << '\n';
  }

  // 4096-aligned scratch so the --direct (O_DIRECT) run can read straight
  // into it zero-copy when the repacked slab's offset+len are block-aligned
  // (they are by construction: gate starts a 4096-aligned block and the
  // slab length is a whole number of 4K pages for IQ2_M). posix_memalign
  // instead of std::vector because O_DIRECT requires a block-aligned dst.
  uint8_t *buf = nullptr;
  if (posix_memalign((void **)&buf, 4096, per_expert) != 0 || !buf) {
    std::cerr << "FAIL: posix_memalign scratch\n";
    return 1;
  }
  auto now = [] { return std::chrono::high_resolution_clock::now(); };

  // --direct: open an O_DIRECT fd on the REPACKED shard (Linux only). This
  // is the only file in the comparison that CAN use O_DIRECT -- the
  // original Unsloth shard is 32-byte aligned, so its expert offsets aren't
  // block-aligned and the kernel rejects O_DIRECT reads with EINVAL (the
  // exact gap gguf_tool pack closes). Validate with a real test read: some
  // filesystems accept O_DIRECT at open() but fail the actual read.
  int r_direct_fd = -1;
  if (use_direct) {
#ifdef __linux__
    if (!R.shard_paths.empty()) {
      r_direct_fd = open(R.shard_paths[0].c_str(), O_RDONLY | O_DIRECT);
      bool ok = r_direct_fd >= 0;
      if (ok) {
        void *tb = nullptr;
        if (posix_memalign(&tb, 4096, 4096) == 0 && tb) {
          ok = pread(r_direct_fd, tb, 4096, 0) >= 0;
          free(tb);
        } else {
          ok = false;
        }
      }
      if (!ok) {
        if (r_direct_fd >= 0)
          close(r_direct_fd);
        r_direct_fd = -1;
        std::cerr << "[io] --direct: O_DIRECT not usable on repacked "
                     "shard on this filesystem -- skipping DMA run\n";
      } else {
        posix_fadvise(r_direct_fd, 0, 0,
                      POSIX_FADV_RANDOM); // no-op for O_DIRECT, harmless
      }
    }
#else
    std::cerr << "[io] --direct: O_DIRECT is Linux-only -- skipping "
                 "DMA run\n";
#endif
  }
  // --- Data Integrity Check ---
  // Verify the actual tensor bytes match before running the timed benchmark.
  // Checks EVERY picked expert (not just the first) so a per-expert offset or
  // stride bug in the repack can't hide behind expert #0 happening to read
  // correctly -- a real failure mode (expert 0 is always at offset 0, so a
  // wrong per-expert stride still reads expert 0 right while corrupting 1+).
  {
    std::vector<uint8_t> buf_old(per_expert);
    std::vector<uint8_t> buf_new(per_expert);
    int matched = 0, bad = 0;
    int first_bad_pick = -1;
    for (size_t i = 0; i < picks.size(); ++i) {
      // OLD: 3 scattered reads of this expert's gate/up/down (buffered).
      const auto &o_gate = old_plan[3 * i + 0];
      const auto &o_up = old_plan[3 * i + 1];
      const auto &o_down = old_plan[3 * i + 2];
      pread(o_gate.fd, buf_old.data(), o_gate.len, o_gate.off);
      pread(o_up.fd, buf_old.data() + o_gate.len, o_up.len, o_up.off);
      pread(o_down.fd, buf_old.data() + o_gate.len + o_up.len, o_down.len,
            o_down.off);
      // NEW: 1 tight read of the same expert (buffered).
      const auto &n_tight = new_plan[i];
      pread(n_tight.fd, buf_new.data(), n_tight.len, n_tight.off);
      if (std::memcmp(buf_old.data(), buf_new.data(), per_expert) != 0) {
        if (first_bad_pick < 0)
          first_bad_pick = (int)i;
        bad++;
      } else {
        matched++;
      }
    }
    std::cout << "verification: " << matched << "/" << picks.size()
              << " picked experts byte-match between original and repacked";
    if (bad) {
      std::cerr << "\nCRITICAL FAIL: " << bad << " picked expert(s) differ "
                << "(first at pick #" << first_bad_pick << ": layer "
                << picks[first_bad_pick].first << " expert "
                << picks[first_bad_pick].second
                << ") -- repacked bytes do NOT match original.\n";
      return 1;
    }
    std::cout << " (repacked bytes exactly match original)\n";
  }

  // Warm-up: touch the same number of BYTES for both paths so neither gets
  // an NVMe / driver-state advantage from the other's run.  OLD issues 3
  // preads per expert so it needs 3x fewer warm experts to match NEW's bytes.
  int warm_experts = std::min(50, reads);
  double old_mbps = 0, new_mbps = 0;

  // ---- NEW: 1 tight pread per expert (run FIRST to avoid OLD warming orig)
  // ----
  {
    for (int i = 0; i < warm_experts; ++i) {
      pread(new_plan[i].fd, buf, new_plan[i].len, new_plan[i].off);
      evict_after_read(new_plan[i].fd, new_plan[i].off, new_plan[i].len);
    }
    auto t0 = now();
    uint64_t bytes = 0;
    for (auto &r : new_plan) {
      pread(r.fd, buf, r.len, r.off);
      evict_after_read(r.fd, r.off, r.len);
      bytes += r.len;
    }
    double s = std::chrono::duration<double>(now() - t0).count();
    new_mbps = s > 0 ? bytes / 1e6 / s : 0;
    std::cout << "NEW 1-tight : " << std::fixed << std::setprecision(1)
              << std::setw(8) << new_mbps << " MB/s  (" << reads
              << " pread calls)\n";
  }

  // ---- OLD: 3 scattered preads per expert ----
  {
    // warm_experts * 3 old-plan entries ≈ same byte count as warm_experts new
    // entries
    for (int i = 0; i < warm_experts * 3; ++i) {
      pread(old_plan[i].fd, buf, old_plan[i].len, old_plan[i].off);
      evict_after_read(old_plan[i].fd, old_plan[i].off, old_plan[i].len);
    }
    auto t0 = now();
    uint64_t bytes = 0;
    for (auto &r : old_plan) {
      pread(r.fd, buf, r.len, r.off);
      evict_after_read(r.fd, r.off, r.len);
      bytes += r.len;
    }
    double s = std::chrono::duration<double>(now() - t0).count();
    old_mbps = s > 0 ? bytes / 1e6 / s : 0;
    std::cout << "OLD 3-scatter: " << std::fixed << std::setprecision(1)
              << std::setw(8) << old_mbps << " MB/s  (" << reads * 3
              << " pread calls)\n";
  }

  // ---- NEW (O_DIRECT/DMA): same 1-tight pread plan as NEW-buffered, but
  // against an O_DIRECT fd on the repacked shard. This is the engine's
  // production --direct-io path and the ONLY line here that bypasses the
  // page cache -- so it's the one that actually shows the DMA benefit.
  // OLD can't join it (original shard isn't block-aligned), so this isn't
  // a clean A/B against OLD; compare it to NEW-buffered to isolate DMA,
  // and to OLD-buffered for the full repacked+O_DIRECT vs original delta.
  double dir_mbps = 0;
  if (r_direct_fd >= 0) {
    // Sanity: every repacked slab offset+len must be 4K-aligned for a raw
    // O_DIRECT pread; if any isn't, route through the engine's alignment-
    // staging helper instead of failing. (For a 4096-aligned repack this
    // branch is never taken -- the aligned raw pread below is zero-copy.)
    bool raw_ok = true;
    for (auto &r : new_plan) {
      if ((r.off % 4096) != 0 || (r.len % 4096) != 0) {
        raw_ok = false;
        break;
      }
    }
    for (int i = 0; i < warm_experts; ++i) {
      auto &r = new_plan[i];
      if (raw_ok)
        pread(r_direct_fd, buf, r.len, r.off);
      else
        direct_io::pread_direct_aware(r_direct_fd, true, 4096, buf, r.len,
                                      r.off);
    }
    auto t0 = now();
    uint64_t bytes = 0;
    for (auto &r : new_plan) {
      ssize_t got = raw_ok
                        ? pread(r_direct_fd, buf, r.len, r.off)
                        : direct_io::pread_direct_aware(r_direct_fd, true, 4096,
                                                        buf, r.len, r.off);
      if (got < 0 || (size_t)got != r.len) {
        std::cerr << "[io] --direct short read at off=" << r.off
                  << " len=" << r.len << " got=" << got << " -- aborting DMA run\n";
        dir_mbps = 0;
        break;
      }
      bytes += r.len;
    }
    double s = std::chrono::duration<double>(now() - t0).count();
    dir_mbps = s > 0 ? bytes / 1e6 / s : 0;
    std::cout << "NEW 1-tight (O_DIRECT): " << std::fixed
              << std::setprecision(1) << std::setw(8) << dir_mbps
              << " MB/s  (" << reads << " DMA pread calls)\n";
  }

  if (old_mbps > 0)
    std::cout << "speedup: " << std::fixed << std::setprecision(2)
              << (new_mbps / old_mbps) << "x  (NEW-buffered / OLD-buffered)\n";
  if (dir_mbps > 0 && new_mbps > 0)
    std::cout << "DMA vs buffered (NEW): " << std::fixed
              << std::setprecision(2) << (dir_mbps / new_mbps)
              << "x  -- isolates the O_DIRECT / page-cache-bypass benefit\n";
  if (dir_mbps > 0 && old_mbps > 0)
    std::cout << "production delta: " << std::fixed << std::setprecision(2)
              << (dir_mbps / old_mbps)
              << "x  (NEW+O_DIRECT / OLD-buffered) -- repacked+DMA vs original\n";

  if (r_direct_fd >= 0)
    close(r_direct_fd);
  free(buf);
  return 0;
}
} // namespace gguf_tool

// tools/gguf_mirror.cpp
//
// gguf_tool mirror-subset: build a SPARSE second copy of a model on another drive that holds only a chosen share of
// the routed experts, so reads can be spread over two NVMe drives without a second full copy.
//
//   gguf_tool mirror-subset <model_file_or_dir> <mirror_dir> --usage-file <decode usage>
//       [--prefill-usage-file <p>] [--prefill-weight W=0.25] --hot-gib G [--fraction F=0.5] [--dry-run]
//   gguf_tool mirror-subset --verify <mirror_dir> --against <model_file_or_dir> [--sample N]
//
// Selection. Every routed expert (packed blk.L.ffn_{gate,up,down}_exps) is ranked by traffic (decode count +
// W * prefill count). The hottest experts, up to G GiB, are what DDR and VRAM are expected to hold; they are never
// the ones read from disk, so they are left out of the mirror. The remaining (cold) experts are walked from hot to
// cold and split between the two drives by error diffusion on BYTES, so the mirror receives fraction F of the cold
// bytes and also of the expected read traffic, instead of one drive getting all the busy ones.
//
// Layout. For each shard a file with the SAME base name, SAME size and SAME tensor offsets as the primary is created
// (ftruncate, i.e. sparse), and only the byte ranges of the chosen experts are written into it. The engine's size
// check passes and every offset stays valid; unchosen ranges are holes and are never read, because the loader
// consults the sidecar index <mirror_dir>/mirror_subset.tsv ("layer<TAB>expert" per line) and sends any expert not
// listed to the primary. Pure byte copy, no requantization.
#include "gguf_common.hpp"
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Proj {
  const GGUFTensor *t = nullptr;
  uint64_t per = 0, n = 0;
};

bool find_proj(const GGUFFile &f, int layer, const char *kind, Proj &out) {
  auto it = f.tensors.find("blk." + std::to_string(layer) + ".ffn_" + kind + "_exps.weight");
  if (it == f.tensors.end() || it->second.shape.size() < 3) return false;
  uint64_t n = it->second.shape.back();
  if (n == 0 || it->second.size_bytes % n != 0) return false;
  out = {&it->second, it->second.size_bytes / n, n};
  return true;
}

bool load_usage(const std::string &path, std::map<std::pair<int, int>, uint64_t> &m, double w) {
  std::ifstream in(path);
  if (!in) return false;
  int l, e;
  uint64_t c;
  while (in >> l >> e >> c) m[{l, e}] += (uint64_t)(c * w * 1000.0);
  return true;
}

struct Expert {
  int layer, expert;
  uint64_t bytes, score;
};

int verify(const std::string &mirror_dir, const std::string &model, int sample) {
  GGUFFile base;
  if (!base.load_header(model)) { fprintf(stderr, "[mirror-verify] FATAL: cannot load model %s\n", model.c_str()); return 1; }
  std::ifstream idx(mirror_dir + "/mirror_subset.tsv");
  if (!idx) { fprintf(stderr, "[mirror-verify] FATAL: no mirror_subset.tsv in %s\n", mirror_dir.c_str()); return 1; }
  std::vector<std::pair<int, int>> keys;
  int l, e;
  while (idx >> l >> e) keys.push_back({l, e});
  if (sample > 0 && (size_t)sample < keys.size()) {
    std::vector<std::pair<int, int>> s2;
    for (size_t i = 0; i < keys.size(); i += keys.size() / sample) s2.push_back(keys[i]);
    keys.swap(s2);
  }
  std::vector<int> mfd(base.shard_paths.size(), -1);
  for (size_t i = 0; i < base.shard_paths.size(); ++i) {
    std::string b = base.shard_paths[i];
    auto sl = b.find_last_of('/');
    if (sl != std::string::npos) b = b.substr(sl + 1);
    mfd[i] = open((mirror_dir + "/" + b).c_str(), O_RDONLY);
    struct stat sp, sm;
    if (mfd[i] < 0 || ::stat(base.shard_paths[i].c_str(), &sp) || ::stat((mirror_dir + "/" + b).c_str(), &sm) ||
        sp.st_size != sm.st_size) {
      fprintf(stderr, "[mirror-verify] FAIL: shard %zu missing or size differs\n", i);
      return 1;
    }
  }
  int bad = 0;
  uint64_t bytes = 0;
  for (auto &[L, E] : keys) {
    for (const char *kind : {"gate", "up", "down"}) {
      Proj p;
      if (!find_proj(base, L, kind, p)) { fprintf(stderr, "[mirror-verify] FAIL: no %s for layer %d\n", kind, L); ++bad; continue; }
      std::vector<uint8_t> a(p.per), b(p.per);
      uint64_t off = p.t->offset + (uint64_t)E * p.per;
      if (pread(base.shard_fds[p.t->shard_idx], a.data(), p.per, (off_t)off) != (ssize_t)p.per ||
          pread(mfd[p.t->shard_idx], b.data(), p.per, (off_t)off) != (ssize_t)p.per || a != b) {
        if (bad++ < 20) fprintf(stderr, "[mirror-verify] FAIL: (%d,%d) %s differs\n", L, E, kind);
      } else bytes += p.per;
    }
  }
  printf("[mirror-verify] %zu experts checked, %.2f GiB compared, %d failure(s)\n", keys.size(), bytes / 1073741824.0, bad);
  return bad ? 1 : 0;
}

} // namespace

namespace gguf_tool {

int cmd_mirror_subset(int argc, char **argv) {
  std::string model, mdir, usage, pusage, against;
  double hot_gib = -1, frac = 0.5, pw = 0.25;
  bool dry = false, do_verify = false;
  int sample = 0;
  std::vector<std::string> pos;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--usage-file") usage = next();
    else if (a == "--prefill-usage-file") pusage = next();
    else if (a == "--prefill-weight") pw = atof(next().c_str());
    else if (a == "--hot-gib") hot_gib = atof(next().c_str());
    else if (a == "--fraction") frac = atof(next().c_str());
    else if (a == "--dry-run") dry = true;
    else if (a == "--verify") do_verify = true;
    else if (a == "--against") against = next();
    else if (a == "--sample") sample = atoi(next().c_str());
    else pos.push_back(a);
  }
  if (do_verify) {
    if (pos.empty() || against.empty()) { fprintf(stderr, "usage: mirror-subset --verify <mirror_dir> --against <model> [--sample N]\n"); return 2; }
    return verify(pos[0], against, sample);
  }
  if (pos.size() != 2 || usage.empty() || hot_gib < 0 || frac < 0 || frac > 1) {
    fprintf(stderr, "usage: mirror-subset <model_file_or_dir> <mirror_dir> --usage-file <u> [--prefill-usage-file <p>] "
                    "[--prefill-weight W] --hot-gib G [--fraction F] [--dry-run]\n");
    return 2;
  }
  model = pos[0];
  mdir = pos[1];
  GGUFFile f;
  if (!f.load_header(model)) { fprintf(stderr, "[mirror] FATAL: cannot load %s\n", model.c_str()); return 1; }

  std::map<std::pair<int, int>, uint64_t> use;
  if (!load_usage(usage, use, 1.0)) { fprintf(stderr, "[mirror] FATAL: cannot read %s\n", usage.c_str()); return 1; }
  if (!pusage.empty()) load_usage(pusage, use, pw);

  std::vector<Expert> all;
  uint64_t total = 0;
  for (int L = 0; L < 1024; ++L) {
    Proj p[3];
    if (!find_proj(f, L, "gate", p[0]) || !find_proj(f, L, "up", p[1]) || !find_proj(f, L, "down", p[2])) continue;
    uint64_t slab = p[0].per + p[1].per + p[2].per;
    for (uint64_t e = 0; e < p[0].n; ++e) {
      auto it = use.find({L, (int)e});
      all.push_back({L, (int)e, slab, it == use.end() ? 0 : it->second});
      total += slab;
    }
  }
  if (all.empty()) { fprintf(stderr, "[mirror] FATAL: no packed routed experts found\n"); return 1; }
  // Hottest first; ties (never-seen experts) interleave across layers so the choice is not concentrated.
  std::stable_sort(all.begin(), all.end(), [](const Expert &a, const Expert &b) {
    if (a.score != b.score) return a.score > b.score;
    if (a.expert != b.expert) return a.expert < b.expert;
    return a.layer < b.layer;
  });

  const uint64_t hot_bytes = (uint64_t)(hot_gib * 1073741824.0);
  uint64_t hot_used = 0, cold_total = 0;
  size_t first_cold = 0;
  while (first_cold < all.size() && hot_used + all[first_cold].bytes <= hot_bytes) hot_used += all[first_cold++].bytes;
  for (size_t i = first_cold; i < all.size(); ++i) cold_total += all[i].bytes;

  std::vector<Expert> chosen;
  uint64_t mirror_bytes = 0, seen = 0;
  uint64_t cold_traffic = 0, mirror_traffic = 0;
  for (size_t i = first_cold; i < all.size(); ++i) {
    seen += all[i].bytes;
    cold_traffic += all[i].score;
    // Error diffusion on bytes: take this expert for the mirror whenever the mirror is behind its target share.
    if ((double)mirror_bytes < frac * (double)seen) {
      chosen.push_back(all[i]);
      mirror_bytes += all[i].bytes;
      mirror_traffic += all[i].score;
    }
  }
  printf("[mirror] experts=%zu total=%.2f GiB | hot (DDR+VRAM) %zu experts %.2f GiB | cold %.2f GiB | mirror gets %zu experts "
         "%.2f GiB (%.1f%% of cold bytes, %.1f%% of cold traffic)\n",
         all.size(), total / 1073741824.0, first_cold, hot_used / 1073741824.0, cold_total / 1073741824.0, chosen.size(),
         mirror_bytes / 1073741824.0, cold_total ? 100.0 * mirror_bytes / cold_total : 0.0,
         cold_traffic ? 100.0 * mirror_traffic / cold_traffic : 0.0);
  if (dry) return 0;

  std::error_code ec;
  fs::create_directories(mdir, ec);
  std::vector<int> ofd(f.shard_paths.size(), -1);
  for (size_t i = 0; i < f.shard_paths.size(); ++i) {
    std::string b = fs::path(f.shard_paths[i]).filename().string();
    struct stat sp;
    if (::stat(f.shard_paths[i].c_str(), &sp)) { fprintf(stderr, "[mirror] FATAL: stat %s\n", f.shard_paths[i].c_str()); return 1; }
    ofd[i] = open((mdir + "/" + b).c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (ofd[i] < 0 || ftruncate(ofd[i], sp.st_size)) { fprintf(stderr, "[mirror] FATAL: create %s/%s: %s\n", mdir.c_str(), b.c_str(), strerror(errno)); return 1; }
  }
  std::ofstream idx(mdir + "/mirror_subset.tsv");
  std::vector<uint8_t> buf;
  uint64_t done = 0;
  for (auto &x : chosen) {
    for (const char *kind : {"gate", "up", "down"}) {
      Proj p;
      find_proj(f, x.layer, kind, p);
      uint64_t off = p.t->offset + (uint64_t)x.expert * p.per;
      buf.resize(p.per);
      int sh = p.t->shard_idx;
      if (pread(f.shard_fds[sh], buf.data(), p.per, (off_t)off) != (ssize_t)p.per ||
          pwrite(ofd[sh], buf.data(), p.per, (off_t)off) != (ssize_t)p.per) {
        fprintf(stderr, "[mirror] FATAL: copy of (%d,%d) %s failed: %s\n", x.layer, x.expert, kind, strerror(errno));
        return 1;
      }
    }
    idx << x.layer << '\t' << x.expert << '\n';
    done += x.bytes;
    if (((done / x.bytes) % 2000) == 0) { printf("[mirror] %.1f / %.1f GiB\r", done / 1073741824.0, mirror_bytes / 1073741824.0); fflush(stdout); }
  }
  for (int fd : ofd) { fsync(fd); close(fd); }
  idx.close();
  printf("\n[mirror] wrote %zu experts, %.2f GiB of data into sparse shards under %s\n", chosen.size(), done / 1073741824.0, mdir.c_str());
  return 0;
}

} // namespace gguf_tool

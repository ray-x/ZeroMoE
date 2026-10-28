// tools/gguf_offload.cpp
//
// gguf_tool offload: build an SSD "offload store" holding the coldest N routed experts of a MoE model, laid out
// for O_DIRECT, and verify such a store against its source model.
//
//   gguf_tool offload <model_file_or_dir> <store.gguf> --usage-file <decode usage> [--prefill-usage-file <p>]
//       (--count N | --fraction F) [--layers 11-44 | --layers 11,12,40-44] [--exclude-layers L,...]
//       [--prefill-weight W=0] [--align N=4096] [--dry-run]
//   gguf_tool offload --verify <store.gguf> --against <model_file_or_dir> [--sample N]
//
// Selection: eligible experts are those of the chosen layers (default: every layer that has packed
// ffn_{gate,up,down}_exps). They are ranked by traffic (decode count + W * prefill count), coldest first. Ties
// (most never-seen experts tie at 0) are broken round-robin across layers so the offloaded set does not
// concentrate in a few layers. The first N ranks are written.
//
// Layout (what the engine's single-DMA-per-expert read path wants): for every selected expert the gate, up and
// down planes are written back to back with NO gap, and the gate plane starts on an `align` boundary, so one
// aligned O_DIRECT read of [gate start, down end) fetches the whole expert. The file ends padded to `align`.
// Tensors are named offload.blk.<L>.ffn_{gate,up,down}.<E>.weight -- deliberately NOT the unpacked model names
// (blk.<L>.ffn_*.<E>.weight): the engine requires an unpacked layer to be complete, and a partial store must never
// be merged into the model by a directory scan. KV metadata carries the index (zeromoe.offload.layer/expert).
// Pure byte copy, no requantization.
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
#include <sstream>
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

bool load_usage(const std::string &path, std::map<std::pair<int, int>, uint64_t> &m) {
  std::ifstream in(path);
  if (!in) return false;
  int l, e; uint64_t c;
  while (in >> l >> e >> c) m[{l, e}] += c;
  return !m.empty();
}

std::set<int> parse_layers(const std::string &s) {
  std::set<int> r; std::stringstream ss(s); std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (tok.empty()) continue;
    auto d = tok.find('-');
    if (d == std::string::npos) r.insert(std::stoi(tok));
    else for (int i = std::stoi(tok.substr(0, d)); i <= std::stoi(tok.substr(d + 1)); ++i) r.insert(i);
  }
  return r;
}

// ---- minimal GGUF writer pieces ----
void w32(std::ofstream &o, uint32_t v) { o.write(reinterpret_cast<const char *>(&v), 4); }
void w64(std::ofstream &o, uint64_t v) { o.write(reinterpret_cast<const char *>(&v), 8); }
void wstr(std::ofstream &o, const std::string &s) { w64(o, s.size()); o.write(s.data(), (std::streamsize)s.size()); }
void kv_u32(std::ofstream &o, const std::string &k, uint32_t v) { wstr(o, k); w32(o, 4); w32(o, v); }
void kv_u64(std::ofstream &o, const std::string &k, uint64_t v) { wstr(o, k); w32(o, 10); w64(o, v); }
void kv_str(std::ofstream &o, const std::string &k, const std::string &v) { wstr(o, k); w32(o, 8); wstr(o, v); }
void kv_arr_u32(std::ofstream &o, const std::string &k, const std::vector<uint32_t> &v) {
  wstr(o, k); w32(o, 9); w32(o, 4); w64(o, v.size());
  o.write(reinterpret_cast<const char *>(v.data()), (std::streamsize)(v.size() * 4));
}

struct Planned {
  std::string name; uint32_t type; std::vector<uint64_t> shape;
  int src_shard; uint64_t src_off, size, out_off; int layer, expert; int kind; // 0 gate 1 up 2 down
};

// ---- minimal GGUF reader (for --verify; independent of the model loader) ----
struct Rd {
  FILE *f;
  template <class T> T get() { T v{}; if (fread(&v, sizeof(T), 1, f) != 1) throw std::runtime_error("short read"); return v; }
  std::string str() { uint64_t n = get<uint64_t>(); std::string s(n, '\0'); if (n && fread(s.data(), 1, n, f) != n) throw std::runtime_error("short read"); return s; }
};
void skip_val(Rd &r, uint32_t t, std::vector<uint32_t> *arr_u32 = nullptr) {
  switch (t) {
  case 0: case 1: case 7: r.get<uint8_t>(); break;
  case 2: case 3: r.get<uint16_t>(); break;
  case 4: case 5: case 6: { uint32_t v = r.get<uint32_t>(); (void)v; break; }
  case 10: case 11: case 12: r.get<uint64_t>(); break;
  case 8: r.str(); break;
  case 9: {
    uint32_t et = r.get<uint32_t>(); uint64_t n = r.get<uint64_t>();
    for (uint64_t i = 0; i < n; ++i) {
      if (et == 4 && arr_u32) arr_u32->push_back(r.get<uint32_t>()); else skip_val(r, et);
    }
    break;
  }
  default: throw std::runtime_error("unknown kv type");
  }
}
struct StoreTensor { std::string name; std::vector<uint64_t> dims; uint32_t type; uint64_t off; };
struct Store {
  uint64_t align = 32, data_start = 0, file_size = 0; std::vector<StoreTensor> ts;
  std::vector<uint32_t> layer_idx, expert_idx;
};
Store read_store(const std::string &path) {
  Store s; Rd r{fopen(path.c_str(), "rb")};
  if (!r.f) throw std::runtime_error("cannot open " + path);
  char m[4]; if (fread(m, 1, 4, r.f) != 4 || memcmp(m, "GGUF", 4)) throw std::runtime_error("not a GGUF");
  r.get<uint32_t>(); uint64_t nt = r.get<uint64_t>(), nk = r.get<uint64_t>();
  for (uint64_t i = 0; i < nk; ++i) {
    std::string k = r.str(); uint32_t t = r.get<uint32_t>();
    if (k == "general.alignment") { s.align = r.get<uint32_t>(); continue; }
    if (k == "zeromoe.offload.layer") { skip_val(r, t, &s.layer_idx); continue; }
    if (k == "zeromoe.offload.expert") { skip_val(r, t, &s.expert_idx); continue; }
    skip_val(r, t);
  }
  for (uint64_t i = 0; i < nt; ++i) {
    StoreTensor t; t.name = r.str(); uint32_t nd = r.get<uint32_t>();
    for (uint32_t d = 0; d < nd; ++d) t.dims.push_back(r.get<uint64_t>());
    t.type = r.get<uint32_t>(); t.off = r.get<uint64_t>(); s.ts.push_back(t);
  }
  uint64_t pos = (uint64_t)ftell(r.f);
  s.data_start = (pos + s.align - 1) / s.align * s.align;
  fseek(r.f, 0, SEEK_END); s.file_size = (uint64_t)ftell(r.f); fclose(r.f);
  return s;
}

int verify_store(const std::string &store_path, const std::string &model, int sample) {
  GGUFFile base;
  if (!base.load_header(model)) { fprintf(stderr, "[offload-verify] FATAL: cannot load model %s\n", model.c_str()); return 1; }
  Store st;
  try { st = read_store(store_path); } catch (const std::exception &e) { fprintf(stderr, "[offload-verify] FATAL: %s\n", e.what()); return 1; }
  printf("[offload-verify] store: %zu tensors, align=%llu, data_start=%llu, file_size=%llu, index entries=%zu\n", st.ts.size(),
         (unsigned long long)st.align, (unsigned long long)st.data_start, (unsigned long long)st.file_size, st.layer_idx.size());
  int bad = 0;
  auto fail = [&](const std::string &m) { if (bad++ < 20) fprintf(stderr, "[offload-verify] FAIL: %s\n", m.c_str()); };
  if (st.align != 4096) fail("general.alignment is " + std::to_string(st.align) + ", expected 4096 for O_DIRECT");
  if (st.file_size % st.align) fail("file size is not a multiple of the alignment");
  if (st.layer_idx.size() != st.expert_idx.size()) fail("index arrays differ in length");
  std::map<std::pair<int, int>, StoreTensor[3]> by;
  std::map<std::pair<int, int>, int> have;
  for (auto &t : st.ts) {
    int L, E; char kind[8];
    if (sscanf(t.name.c_str(), "offload.blk.%d.ffn_%7[a-z].%d.weight", &L, kind, &E) != 3) { fail("unexpected tensor name " + t.name); continue; }
    int k = !strcmp(kind, "gate") ? 0 : !strcmp(kind, "up") ? 1 : !strcmp(kind, "down") ? 2 : -1;
    if (k < 0) { fail("unknown projection in " + t.name); continue; }
    by[{L, E}][k] = t; have[{L, E}] |= 1 << k;
  }
  std::vector<std::pair<int, int>> keys;
  for (auto &[k, mask] : have) { if (mask != 7) fail("expert (" + std::to_string(k.first) + "," + std::to_string(k.second) + ") is missing a projection"); else keys.push_back(k); }
  if (keys.size() != st.layer_idx.size()) fail("index has " + std::to_string(st.layer_idx.size()) + " entries but tensors describe " + std::to_string(keys.size()) + " experts");
  if (sample > 0 && (size_t)sample < keys.size()) { std::vector<std::pair<int, int>> s2; for (size_t i = 0; i < keys.size(); i += keys.size() / sample + 1) s2.push_back(keys[i]); keys = s2; }
#ifdef O_DIRECT
  int dfd = open(store_path.c_str(), O_RDONLY | O_DIRECT);
#else
  // O_DIRECT is Linux-specific; other platforms verify with buffered reads.
  int dfd = -1;
  errno = EOPNOTSUPP;
#endif
  bool direct = dfd >= 0; if (!direct) { fprintf(stderr, "[offload-verify] WARNING: O_DIRECT open failed (%s); verifying with buffered reads\n", strerror(errno)); dfd = open(store_path.c_str(), O_RDONLY); }
  uint64_t checked_bytes = 0; size_t n_ok = 0;
  for (auto &[L, E] : keys) {
    auto &tr = by[{L, E}];
    uint64_t abs0 = st.data_start + tr[0].off, slab = 0;
    if (abs0 % st.align) fail("expert (" + std::to_string(L) + "," + std::to_string(E) + ") gate start " + std::to_string(abs0) + " is not aligned");
    Proj p[3]; const char *kinds[3] = {"gate", "up", "down"};
    bool ok = true;
    for (int k = 0; k < 3; ++k) {
      if (!find_proj(base, L, kinds[k], p[k])) { fail("source model lacks layer " + std::to_string(L) + " " + kinds[k]); ok = false; break; }
      slab += p[k].per;
    }
    if (!ok) continue;
    if (st.data_start + tr[1].off != abs0 + p[0].per || st.data_start + tr[2].off != abs0 + p[0].per + p[1].per) { fail("planes of expert (" + std::to_string(L) + "," + std::to_string(E) + ") are not tight"); continue; }
    uint64_t a0 = abs0 & ~(uint64_t)4095, alen = ((abs0 + slab + 4095) & ~(uint64_t)4095) - a0;
    void *buf = nullptr; if (posix_memalign(&buf, 4096, alen)) { fail("posix_memalign"); continue; }
    ssize_t g = pread(dfd, buf, alen, (off_t)a0);
    if (g < (ssize_t)(abs0 - a0 + slab)) { fail("short store read at expert (" + std::to_string(L) + "," + std::to_string(E) + ")"); free(buf); continue; }
    std::vector<uint8_t> src(slab); uint64_t o = 0; bool same = true;
    for (int k = 0; k < 3 && same; ++k) {
      int fd = base.shard_fds[p[k].t->shard_idx];
      uint64_t so = p[k].t->offset + (uint64_t)E * p[k].per;
      if (pread(fd, src.data() + o, p[k].per, (off_t)so) != (ssize_t)p[k].per) { fail("short source read"); same = false; break; }
      o += p[k].per;
    }
    if (same && memcmp(src.data(), (uint8_t *)buf + (abs0 - a0), slab) != 0) { fail("bytes differ for expert (" + std::to_string(L) + "," + std::to_string(E) + ")"); same = false; }
    free(buf);
    if (same) { ++n_ok; checked_bytes += slab; }
  }
  close(dfd);
  printf("[offload-verify] %zu/%zu experts byte-identical to the source via %s reads (%.2f GB); %d problem(s)\n", n_ok, keys.size(), direct ? "O_DIRECT" : "buffered", checked_bytes / 1e9, bad);
  printf("[offload-verify] %s\n", bad == 0 ? "PASS" : "FAIL");
  return bad == 0 ? 0 : 1;
}

} // namespace

namespace gguf_tool {

int cmd_offload(int argc, char **argv) {
  std::string model, store, usage_file, prefill_file, against;
  double prefill_w = 0.0, fraction = 0.0; long count = 0; uint64_t align = 4096;
  std::set<int> layer_filter, layer_exclude; bool dry = false, verify = false; int sample = 0;
  long expert_mod = 0, expert_rem = 0; // --expert-mod M --expert-rem R: only experts with id % M == R (one part of a store striped over M drives)
  std::vector<std::string> pos;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char *n) -> const char * { if (i + 1 >= argc) { fprintf(stderr, "[offload] FATAL: %s needs a value\n", n); exit(2); } return argv[++i]; };
    if (a == "--usage-file") usage_file = need("--usage-file");
    else if (a == "--prefill-usage-file") prefill_file = need("--prefill-usage-file");
    else if (a == "--prefill-weight") prefill_w = atof(need("--prefill-weight"));
    else if (a == "--count") count = atol(need("--count"));
    else if (a == "--fraction") fraction = atof(need("--fraction"));
    else if (a == "--expert-mod") expert_mod = atol(need("--expert-mod"));
    else if (a == "--expert-rem") expert_rem = atol(need("--expert-rem"));
    else if (a == "--layers") layer_filter = parse_layers(need("--layers"));
    else if (a == "--exclude-layers") layer_exclude = parse_layers(need("--exclude-layers"));
    else if (a == "--align") align = (uint64_t)atoll(need("--align"));
    else if (a == "--dry-run") dry = true;
    else if (a == "--verify") { verify = true; store = need("--verify"); }
    else if (a == "--against") against = need("--against");
    else if (a == "--sample") sample = atoi(need("--sample"));
    else if (!a.empty() && a[0] == '-') { fprintf(stderr, "[offload] FATAL: unknown option '%s'\n", a.c_str()); return 2; }
    else pos.push_back(a);
  }
  if (verify) {
    if (against.empty()) { fprintf(stderr, "[offload] FATAL: --verify needs --against <model>\n"); return 2; }
    return verify_store(store, against, sample);
  }
  if (pos.size() != 2 || usage_file.empty() || (count <= 0 && fraction <= 0.0)) {
    fprintf(stderr, "Usage: gguf_tool offload <model_file_or_dir> <store.gguf> --usage-file <decode usage> [--prefill-usage-file <p>]\n"
                    "         (--count N | --fraction F) [--layers 11-44] [--exclude-layers L,..] [--expert-mod M --expert-rem R] [--prefill-weight W] [--align 4096] [--dry-run]\n"
                    "       gguf_tool offload --verify <store.gguf> --against <model_file_or_dir> [--sample N]\n");
    return 2;
  }
  if (align < 512 || (align & (align - 1))) { fprintf(stderr, "[offload] FATAL: --align must be a power of two >= 512\n"); return 2; }
  model = pos[0]; store = pos[1];

  GGUFFile base;
  if (!base.load_header(model)) { fprintf(stderr, "[offload] FATAL: cannot load model %s\n", model.c_str()); return 1; }
  std::map<std::pair<int, int>, uint64_t> dec, pre;
  if (!load_usage(usage_file, dec)) { fprintf(stderr, "[offload] FATAL: no parseable entries in %s (refusing to pick blind)\n", usage_file.c_str()); return 1; }
  if (!prefill_file.empty() && !load_usage(prefill_file, pre)) { fprintf(stderr, "[offload] FATAL: no parseable entries in %s\n", prefill_file.c_str()); return 1; }

  struct Cand { int layer, expert; double score; uint64_t hits; int tie; };
  std::vector<Cand> cands; std::map<int, std::array<Proj, 3>> layers;
  for (int L = 0; L < 4096; ++L) {
    if (!layer_filter.empty() && !layer_filter.count(L)) continue;
    if (layer_exclude.count(L)) continue;
    std::array<Proj, 3> p; const char *kinds[3] = {"gate", "up", "down"}; bool ok = true;
    for (int k = 0; k < 3; ++k) if (!find_proj(base, L, kinds[k], p[k])) { ok = false; break; }
    if (!ok || p[0].n != p[1].n || p[0].n != p[2].n) continue;
    layers[L] = p;
  }
  if (layers.empty()) { fprintf(stderr, "[offload] FATAL: no eligible layer (packed ffn_*_exps) in the selection\n"); return 1; }
  uint64_t total_hits = 0;
  for (auto &[L, p] : layers) {
    std::vector<Cand> lc;
    for (uint64_t e = 0; e < p[0].n; ++e) {
      if (expert_mod > 1 && (long)(e % (uint64_t)expert_mod) != expert_rem) continue;
      uint64_t d = dec.count({L, (int)e}) ? dec[{L, (int)e}] : 0, q = pre.count({L, (int)e}) ? pre[{L, (int)e}] : 0;
      lc.push_back({L, (int)e, (double)d + prefill_w * (double)q, d, 0}); total_hits += d;
    }
    std::sort(lc.begin(), lc.end(), [](const Cand &a, const Cand &b) { return a.score != b.score ? a.score < b.score : a.expert < b.expert; });
    int run = 0;
    for (size_t i = 0; i < lc.size(); ++i) { run = (i && lc[i].score == lc[i - 1].score) ? run + 1 : 0; lc[i].tie = run; }
    cands.insert(cands.end(), lc.begin(), lc.end());
  }
  std::sort(cands.begin(), cands.end(), [](const Cand &a, const Cand &b) {
    if (a.score != b.score) return a.score < b.score;
    if (a.tie != b.tie) return a.tie < b.tie;            // round-robin across layers within a tie
    return a.layer != b.layer ? a.layer < b.layer : a.expert < b.expert; });
  size_t n = count > 0 ? (size_t)count : (size_t)(fraction * (double)cands.size());
  if (n == 0 || n > cands.size()) { fprintf(stderr, "[offload] FATAL: selection of %zu experts out of %zu eligible is invalid\n", n, cands.size()); return 1; }
  cands.resize(n);
  std::sort(cands.begin(), cands.end(), [](const Cand &a, const Cand &b) { return a.layer != b.layer ? a.layer < b.layer : a.expert < b.expert; });

  // ---- plan: per expert gate,up,down back to back; gate aligned ----
  std::vector<Planned> plan; uint64_t running = 0, sel_hits = 0, slab_total = 0; std::map<int, int> per_layer;
  std::vector<uint32_t> idx_l, idx_e; const char *kinds[3] = {"gate", "up", "down"};
  for (auto &c : cands) {
    auto &p = layers[c.layer]; running = gguf_tool::align_up(running, align);
    for (int k = 0; k < 3; ++k) {
      Planned t; t.name = "offload.blk." + std::to_string(c.layer) + ".ffn_" + kinds[k] + "." + std::to_string(c.expert) + ".weight";
      t.type = (uint32_t)p[k].t->type; t.shape.assign(p[k].t->shape.begin(), p[k].t->shape.end() - 1);
      t.src_shard = p[k].t->shard_idx; t.src_off = p[k].t->offset + (uint64_t)c.expert * p[k].per; t.size = p[k].per;
      t.out_off = running; t.layer = c.layer; t.expert = c.expert; t.kind = k; running += t.size; plan.push_back(std::move(t));
      slab_total += p[k].per;
    }
    sel_hits += c.hits; per_layer[c.layer]++; idx_l.push_back((uint32_t)c.layer); idx_e.push_back((uint32_t)c.expert);
  }
  uint64_t data_bytes = gguf_tool::align_up(running, align);
  printf("[offload] model: %zu tensors / %zu shard(s); eligible layers %zu, eligible experts %zu\n", base.tensors.size(), base.shard_paths.size(), layers.size(), layers.size() ? (size_t)layers.begin()->second[0].n * layers.size() : 0);
  printf("[offload] selected %zu coldest experts = %.2f GiB of payload (%.2f GiB file incl. alignment padding); they carry %.3f%% of the survey's decode hits\n", n, slab_total / 1073741824.0, data_bytes / 1073741824.0, total_hits ? 100.0 * (double)sel_hits / (double)total_hits : 0.0);
  printf("[offload] experts per layer:"); for (auto &[L, c] : per_layer) printf(" L%d:%d", L, c); printf("\n");
  if (dry) { printf("[offload] --dry-run: stopping before any write.\n"); return 0; }

  std::ofstream out(store, std::ios::binary | std::ios::trunc);
  if (!out) { fprintf(stderr, "[offload] FATAL: cannot create %s\n", store.c_str()); return 1; }
  out.exceptions(std::ofstream::failbit | std::ofstream::badbit);
  std::vector<uint8_t> buf(64ULL << 20); uint64_t written = 0;
  try {
    out.write("GGUF", 4); w32(out, 3); w64(out, plan.size()); w64(out, 8);
    kv_u32(out, "general.alignment", (uint32_t)align);
    kv_str(out, "general.architecture", "zeromoe-offload");
    kv_u32(out, "zeromoe.offload.version", 1);
    kv_str(out, "zeromoe.offload.source", fs::path(model).filename().string());
    kv_u64(out, "zeromoe.offload.count", n);
    kv_u64(out, "zeromoe.offload.slab_bytes_first", plan.size() >= 3 ? plan[0].size + plan[1].size + plan[2].size : 0);
    kv_arr_u32(out, "zeromoe.offload.layer", idx_l);
    kv_arr_u32(out, "zeromoe.offload.expert", idx_e);
    for (auto &t : plan) {
      wstr(out, t.name); w32(out, (uint32_t)t.shape.size());
      for (auto d : t.shape) w64(out, d);
      w32(out, t.type); w64(out, t.out_off);
    }
    uint64_t after = (uint64_t)out.tellp(), data_start = gguf_tool::align_up(after, align);
    if (data_start > after) { std::vector<char> pad(data_start - after, 0); out.write(pad.data(), (std::streamsize)pad.size()); }
    uint64_t so_far = 0, last_gb = 0;
    for (auto &t : plan) {
      if (t.out_off > so_far) { std::vector<char> pad(t.out_off - so_far, 0); out.write(pad.data(), (std::streamsize)pad.size()); }
      uint64_t rem = t.size, off = t.src_off; int fd = base.shard_fds[t.src_shard];
      while (rem) {
        size_t ch = (size_t)std::min<uint64_t>(rem, buf.size()); ssize_t r = pread(fd, buf.data(), ch, (off_t)off);
        if (r != (ssize_t)ch) { fprintf(stderr, "\n[offload] FATAL: short source read for %s\n", t.name.c_str()); return 1; }
        out.write(reinterpret_cast<const char *>(buf.data()), (std::streamsize)ch); rem -= ch; off += ch;
      }
      so_far = t.out_off + t.size; written += t.size;
      if (written / (1ULL << 30) != last_gb) { last_gb = written / (1ULL << 30); printf("\r[offload] written %llu GiB...", (unsigned long long)last_gb); fflush(stdout); }
    }
    if (data_bytes > so_far) { std::vector<char> pad(data_bytes - so_far, 0); out.write(pad.data(), (std::streamsize)pad.size()); }
    out.close();
  } catch (const std::ios_base::failure &) {
    fprintf(stderr, "\n[offload] FATAL: write to %s failed (errno=%d: %s) -- likely ENOSPC. The file is INCOMPLETE; do not use it.\n", store.c_str(), errno, strerror(errno));
    return 1;
  }
  printf("\n[offload] wrote %s: %zu experts, %zu tensors, %.2f GiB payload\n", store.c_str(), n, plan.size(), slab_total / 1073741824.0);
  printf("[offload] next: gguf_tool offload --verify %s --against %s\n", store.c_str(), model.c_str());
  return 0;
}

} // namespace gguf_tool

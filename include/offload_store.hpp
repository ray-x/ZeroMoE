#pragma once
// Host-side I/O for the SSD expert offload (Linux): the aligned store reader, a fallback reader over the main model,
// a virtual per-layer arena whose resident set is controlled page by page, and ExpertResidency, the glue that makes a
// layer's routed ids resident (the call a graph "miss node" makes).
//
//   OffloadStore   reads the 4096-aligned store written by `gguf_tool offload`. Each expert's gate, up and down planes
//                  are contiguous and start on a 4096 boundary, so ONE preadv(O_DIRECT) scatters a whole expert
//                  straight into three aligned destination planes (zero copy). Anything misaligned falls back to
//                  buffered per-plane reads (correct, slower).
//   ModelReader    reads experts of the packed model (blk.L.ffn_{gate,up,down}_exps.weight) with buffered preads; the
//                  source for experts that are not in the store (for example a hot expert that was evicted).
//   VirtualArena   per layer and projection, a full-size mapping with slot = expert id (mmap NORESERVE): only
//                  populated slots cost memory. evict() returns a slot's pages (MADV_DONTNEED). Because slot index ==
//                  expert id, MUL_MAT_ID over the arena needs no remapping.
//   ExpertResidency  cache policy (expert_cache.hpp) + arena + sources: ensure(layer, ids) loads what is missing.
// Not thread-safe; call from the decode thread (or serialize).
#if !defined(__linux__)
#error "offload_store.hpp needs Linux (O_DIRECT, preadv, mincore)"
#endif
#include "expert_cache.hpp"

#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace moestream::offload {

enum class ReadStatus { Ok, NotFound, SizeMismatch, IoError, ShortRead };
inline const char *to_string(ReadStatus s) {
  switch (s) {
  case ReadStatus::Ok: return "ok";
  case ReadStatus::NotFound: return "not found";
  case ReadStatus::SizeMismatch: return "plane size mismatch";
  case ReadStatus::IoError: return "I/O error";
  case ReadStatus::ShortRead: return "short read";
  }
  return "?";
}

// ---------------------------------------------------------------------------------------------------- GGUF header
struct GgufTensorInfo {
  std::string name;
  std::vector<uint64_t> dims;
  uint32_t type = 0;
  uint64_t offset = 0; // relative to data_start
};
struct GgufHeader {
  uint32_t alignment = 32;
  std::string architecture;
  uint64_t data_start = 0, file_size = 0;
  std::vector<GgufTensorInfo> tensors;
  std::vector<uint32_t> offload_layer, offload_expert; // moestream.offload.layer/expert
};

namespace detail {
struct File {
  FILE *f = nullptr;
  explicit File(const std::string &p) : f(std::fopen(p.c_str(), "rb")) {}
  ~File() { if (f) std::fclose(f); }
  File(const File &) = delete;
  File &operator=(const File &) = delete;
  template <class T> bool get(T &v) { return std::fread(&v, sizeof(T), 1, f) == 1; }
  bool str(std::string &s) {
    uint64_t n = 0;
    if (!get(n) || n > (1ull << 32)) return false;
    s.assign(static_cast<size_t>(n), '\0');
    return n == 0 || std::fread(s.data(), 1, static_cast<size_t>(n), f) == n;
  }
};
inline bool skip_value(File &r, uint32_t t, std::vector<uint32_t> *u32_out) {
  switch (t) {
  case 0: case 1: case 7: { uint8_t v; return r.get(v); }
  case 2: case 3: { uint16_t v; return r.get(v); }
  case 4: case 5: case 6: { uint32_t v; return r.get(v); }
  case 10: case 11: case 12: { uint64_t v; return r.get(v); }
  case 8: { std::string s; return r.str(s); }
  case 9: {
    uint32_t et = 0;
    uint64_t n = 0;
    if (!r.get(et) || !r.get(n)) return false;
    for (uint64_t i = 0; i < n; ++i) {
      if (et == 4 && u32_out) { uint32_t v; if (!r.get(v)) return false; u32_out->push_back(v); }
      else if (!skip_value(r, et, nullptr)) return false;
    }
    return true;
  }
  default: return false;
  }
}
} // namespace detail

inline bool read_gguf_header(const std::string &path, GgufHeader &h, std::string *err = nullptr) {
  auto fail = [&](const char *m) { if (err) *err = std::string(m) + ": " + path; return false; };
  detail::File r(path);
  if (!r.f) return fail("cannot open");
  char magic[4];
  if (std::fread(magic, 1, 4, r.f) != 4 || std::memcmp(magic, "GGUF", 4) != 0) return fail("not a GGUF file");
  uint32_t version = 0;
  uint64_t nt = 0, nk = 0;
  if (!r.get(version) || !r.get(nt) || !r.get(nk)) return fail("truncated header");
  for (uint64_t i = 0; i < nk; ++i) {
    std::string key;
    uint32_t t = 0;
    if (!r.str(key) || !r.get(t)) return fail("truncated kv");
    if (key == "general.alignment" && t == 4) { if (!r.get(h.alignment)) return fail("truncated kv"); }
    else if (key == "general.architecture" && t == 8) { if (!r.str(h.architecture)) return fail("truncated kv"); }
    else if (key == "moestream.offload.layer") { if (!detail::skip_value(r, t, &h.offload_layer)) return fail("bad kv"); }
    else if (key == "moestream.offload.expert") { if (!detail::skip_value(r, t, &h.offload_expert)) return fail("bad kv"); }
    else if (!detail::skip_value(r, t, nullptr)) return fail("unsupported kv type");
  }
  h.tensors.clear();
  h.tensors.reserve(static_cast<size_t>(nt));
  for (uint64_t i = 0; i < nt; ++i) {
    GgufTensorInfo t;
    uint32_t nd = 0;
    if (!r.str(t.name) || !r.get(nd) || nd > 8) return fail("bad tensor info");
    t.dims.resize(nd);
    for (uint32_t d = 0; d < nd; ++d) if (!r.get(t.dims[d])) return fail("bad tensor info");
    if (!r.get(t.type) || !r.get(t.offset)) return fail("bad tensor info");
    h.tensors.push_back(std::move(t));
  }
  const uint64_t pos = static_cast<uint64_t>(std::ftell(r.f));
  const uint64_t a = h.alignment ? h.alignment : 32;
  h.data_start = (pos + a - 1) / a * a;
  std::fseek(r.f, 0, SEEK_END);
  h.file_size = static_cast<uint64_t>(std::ftell(r.f));
  return true;
}

// Bytes of one row of `ne0` elements for the ggml types that appear in MoE expert tensors (0 = unknown type).
inline uint64_t ggml_row_bytes(uint32_t type, uint64_t ne0) {
  struct T { uint32_t type, block, bytes; };
  static constexpr T table[] = {{0, 1, 4}, {1, 1, 2}, {2, 32, 18}, {8, 32, 34}, {12, 256, 144}, {13, 256, 176}, {14, 256, 210},
                                {16, 256, 66}, {17, 256, 74}, {18, 256, 98}, {19, 256, 50}, {20, 32, 18}, {21, 256, 110},
                                {22, 256, 82}, {23, 256, 136}, {29, 256, 56}, {30, 1, 2}, {42, 32, 10}};
  for (const T &e : table)
    if (e.type == type) return (ne0 / e.block) * e.bytes;
  return 0;
}

// ---------------------------------------------------------------------------------------------------- sources
class ExpertSource {
public:
  virtual ~ExpertSource() = default;
  virtual bool contains(int layer, int expert) const = 0;
  // Byte size of the gate/up/down plane of this expert.
  virtual bool plane_bytes(int layer, int expert, uint64_t out[3]) const = 0;
  virtual ReadStatus read_expert(int layer, int expert, void *const dst[3]) = 0;
};

namespace detail {
inline bool preadv_all(int fd, struct iovec *iov, int n, off_t off) {
  while (n > 0) {
    ssize_t r = ::preadv(fd, iov, n, off);
    if (r < 0) { if (errno == EINTR) continue; return false; }
    if (r == 0) return false;
    off += r;
    while (n > 0 && static_cast<size_t>(r) >= iov->iov_len) { r -= static_cast<ssize_t>(iov->iov_len); ++iov; --n; }
    if (n > 0 && r > 0) { iov->iov_base = static_cast<char *>(iov->iov_base) + r; iov->iov_len -= static_cast<size_t>(r); }
  }
  return true;
}
inline bool pread_all(int fd, void *dst, size_t len, off_t off) {
  struct iovec v = {dst, len};
  return preadv_all(fd, &v, 1, off);
}
inline uint64_t key(int layer, int expert) { return (static_cast<uint64_t>(static_cast<uint32_t>(layer)) << 32) | static_cast<uint32_t>(expert); }
} // namespace detail

class OffloadStore final : public ExpertSource {
public:
  OffloadStore() = default;
  ~OffloadStore() override { close_fds(); }
  OffloadStore(const OffloadStore &) = delete;
  OffloadStore &operator=(const OffloadStore &) = delete;

  // Returns false (with *err set) if the file is not a valid, expert-complete store.
  bool open(const std::string &path, std::string *err = nullptr, bool try_direct = true) {
    close_fds();
    slabs_.clear();
    GgufHeader h;
    if (!read_gguf_header(path, h, err)) return false;
    auto fail = [&](const std::string &m) { if (err) *err = m + ": " + path; return false; };
    if (h.architecture != "moestream-offload") return fail("not an offload store (architecture '" + h.architecture + "')");
    align_ = h.alignment;
    if (align_ == 0 || (align_ & (align_ - 1))) return fail("bad alignment");
    struct Part { uint64_t off[3] = {0, 0, 0}, bytes[3] = {0, 0, 0}; int seen = 0; };
    std::unordered_map<uint64_t, Part> parts;
    for (const GgufTensorInfo &t : h.tensors) {
      int L = 0, E = 0;
      char kind[8] = {0};
      if (std::sscanf(t.name.c_str(), "offload.blk.%d.ffn_%7[a-z].%d.weight", &L, kind, &E) != 3) return fail("unexpected tensor " + t.name);
      int k = !std::strcmp(kind, "gate") ? 0 : !std::strcmp(kind, "up") ? 1 : !std::strcmp(kind, "down") ? 2 : -1;
      if (k < 0 || t.dims.size() < 2) return fail("bad tensor " + t.name);
      const uint64_t bytes = ggml_row_bytes(t.type, t.dims[0]) * t.dims[1];
      if (bytes == 0) return fail("unknown quant type in " + t.name);
      Part &p = parts[detail::key(L, E)];
      p.off[k] = h.data_start + t.offset;
      p.bytes[k] = bytes;
      p.seen |= 1 << k;
    }
    for (auto &[key, p] : parts) {
      if (p.seen != 7) return fail("an expert is missing a projection");
      Slab s;
      for (int k = 0; k < 3; ++k) { s.off[k] = p.off[k]; s.bytes[k] = p.bytes[k]; }
      s.tight = p.off[1] == p.off[0] + p.bytes[0] && p.off[2] == p.off[1] + p.bytes[1];
      slabs_[key] = s;
    }
    if (h.offload_layer.size() != h.offload_expert.size() || h.offload_layer.size() != slabs_.size()) return fail("index/tensor count mismatch");
    fd_buf_ = ::open(path.c_str(), O_RDONLY);
    if (fd_buf_ < 0) return fail("cannot open for reading");
    if (try_direct) fd_dir_ = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    return true;
  }

  size_t size() const { return slabs_.size(); }
  uint32_t alignment() const { return align_; }
  bool direct() const { return fd_dir_ >= 0; }
  std::vector<ExpertKey> keys() const {
    std::vector<ExpertKey> out;
    out.reserve(slabs_.size());
    for (const auto &[k, s] : slabs_) out.push_back({static_cast<int32_t>(k >> 32), static_cast<int32_t>(k & 0xffffffffu)});
    std::sort(out.begin(), out.end());
    return out;
  }
  bool contains(int layer, int expert) const override { return slabs_.count(detail::key(layer, expert)) != 0; }
  bool plane_bytes(int layer, int expert, uint64_t out[3]) const override {
    auto it = slabs_.find(detail::key(layer, expert));
    if (it == slabs_.end()) return false;
    for (int k = 0; k < 3; ++k) out[k] = it->second.bytes[k];
    return true;
  }
  // Counters (diagnostics): how many reads took the zero-copy O_DIRECT path vs the buffered fallback.
  uint64_t direct_reads() const { return direct_reads_; }
  uint64_t buffered_reads() const { return buffered_reads_; }

  ReadStatus read_expert(int layer, int expert, void *const dst[3]) override {
    auto it = slabs_.find(detail::key(layer, expert));
    if (it == slabs_.end()) return ReadStatus::NotFound;
    const Slab &s = it->second;
    bool fast = fd_dir_ >= 0 && s.tight && s.off[0] % align_ == 0;
    for (int k = 0; k < 3 && fast; ++k)
      if (reinterpret_cast<uintptr_t>(dst[k]) % align_ != 0 || s.bytes[k] % align_ != 0) fast = false;
    if (fast) {
      struct iovec iov[3] = {{dst[0], static_cast<size_t>(s.bytes[0])}, {dst[1], static_cast<size_t>(s.bytes[1])}, {dst[2], static_cast<size_t>(s.bytes[2])}};
      if (detail::preadv_all(fd_dir_, iov, 3, static_cast<off_t>(s.off[0]))) { ++direct_reads_; return ReadStatus::Ok; }
      // fall through to the buffered path (O_DIRECT can refuse on some filesystems)
    }
    for (int k = 0; k < 3; ++k) // each plane at its own offset: a store need not be tight
      if (!detail::pread_all(fd_buf_, dst[k], static_cast<size_t>(s.bytes[k]), static_cast<off_t>(s.off[k]))) return ReadStatus::IoError;
    ++buffered_reads_;
    return ReadStatus::Ok;
  }

private:
  struct Slab {
    uint64_t off[3] = {0, 0, 0}, bytes[3] = {0, 0, 0}; // absolute file offset and size of each plane
    bool tight = false;                                 // planes back to back: one preadv can fetch the whole expert
  };
  void close_fds() {
    if (fd_dir_ >= 0) ::close(fd_dir_);
    if (fd_buf_ >= 0) ::close(fd_buf_);
    fd_dir_ = fd_buf_ = -1;
  }
  std::unordered_map<uint64_t, Slab> slabs_;
  uint32_t align_ = 4096;
  int fd_dir_ = -1, fd_buf_ = -1;
  uint64_t direct_reads_ = 0, buffered_reads_ = 0;
};

class ModelReader final : public ExpertSource {
public:
  ModelReader() = default;
  ~ModelReader() override { for (int fd : fds_) if (fd >= 0) ::close(fd); }
  ModelReader(const ModelReader &) = delete;
  ModelReader &operator=(const ModelReader &) = delete;

  bool open(const std::vector<std::string> &shards, std::string *err = nullptr) {
    for (const std::string &path : shards) {
      GgufHeader h;
      if (!read_gguf_header(path, h, err)) return false;
      const int fd = ::open(path.c_str(), O_RDONLY);
      if (fd < 0) { if (err) *err = "cannot open " + path; return false; }
      fds_.push_back(fd);
      for (const GgufTensorInfo &t : h.tensors) {
        int L = 0;
        char kind[8] = {0};
        if (std::sscanf(t.name.c_str(), "blk.%d.ffn_%7[a-z]_exps.weight", &L, kind) != 2 || t.dims.size() < 3) continue;
        int k = !std::strcmp(kind, "gate") ? 0 : !std::strcmp(kind, "up") ? 1 : !std::strcmp(kind, "down") ? 2 : -1;
        if (k < 0) continue;
        const uint64_t per = ggml_row_bytes(t.type, t.dims[0]) * t.dims[1];
        if (per == 0 || t.dims[2] == 0) continue;
        Proj &p = layers_[L].p[k];
        p = {fd, h.data_start + t.offset, per, t.dims[2]};
      }
    }
    return !layers_.empty();
  }
  bool contains(int layer, int expert) const override {
    auto it = layers_.find(layer);
    if (it == layers_.end()) return false;
    for (const Proj &p : it->second.p) if (p.fd < 0 || static_cast<uint64_t>(expert) >= p.n_experts) return false;
    return expert >= 0;
  }
  bool plane_bytes(int layer, int expert, uint64_t out[3]) const override {
    if (!contains(layer, expert)) return false;
    for (int k = 0; k < 3; ++k) out[k] = layers_.at(layer).p[k].per_expert;
    return true;
  }
  ReadStatus read_expert(int layer, int expert, void *const dst[3]) override {
    if (!contains(layer, expert)) return ReadStatus::NotFound;
    for (int k = 0; k < 3; ++k) {
      const Proj &p = layers_.at(layer).p[k];
      if (!detail::pread_all(p.fd, dst[k], static_cast<size_t>(p.per_expert), static_cast<off_t>(p.offset + static_cast<uint64_t>(expert) * p.per_expert))) return ReadStatus::IoError;
    }
    return ReadStatus::Ok;
  }

private:
  struct Proj { int fd = -1; uint64_t offset = 0, per_expert = 0, n_experts = 0; };
  struct Layer { Proj p[3]; };
  std::vector<int> fds_;
  std::unordered_map<int, Layer> layers_;
};

// ---------------------------------------------------------------------------------------------------- arena
class VirtualArena {
public:
  // plane_bytes[l][k] = size of the gate/up/down plane of one expert of layer (first_layer + l).
  VirtualArena(int first_layer, int n_layers, int n_experts, const std::vector<std::array<uint64_t, 3>> &plane_bytes, bool no_huge_pages = false)
      : first_(first_layer), n_layers_(n_layers), n_experts_(n_experts) {
    for (int l = 0; l < n_layers; ++l) {
      for (int k = 0; k < 3; ++k) {
        Map m;
        m.plane = plane_bytes[static_cast<size_t>(l)][static_cast<size_t>(k)];
        m.stride = (m.plane + kPage - 1) / kPage * kPage;
        m.len = m.stride * static_cast<uint64_t>(n_experts);
        void *p = ::mmap(nullptr, static_cast<size_t>(m.len), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (p == MAP_FAILED) { ok_ = false; m.base = nullptr; }
        else {
          m.base = static_cast<char *>(p);
          if (no_huge_pages) ::madvise(p, static_cast<size_t>(m.len), MADV_NOHUGEPAGE);
        }
        maps_.push_back(m);
      }
    }
  }
  ~VirtualArena() { for (const Map &m : maps_) if (m.base) ::munmap(m.base, static_cast<size_t>(m.len)); }
  VirtualArena(const VirtualArena &) = delete;
  VirtualArena &operator=(const VirtualArena &) = delete;

  bool ok() const { return ok_; }
  void *ptr(int layer, int expert, int plane) const {
    const Map &m = map(layer, plane);
    return m.base ? m.base + static_cast<uint64_t>(expert) * m.stride : nullptr;
  }
  uint64_t plane_bytes(int layer, int plane) const { return map(layer, plane).plane; }
  uint64_t stride(int layer, int plane) const { return map(layer, plane).stride; }
  // Return one expert's pages to the OS; the slot reads as zeros until it is populated again.
  void evict(int layer, int expert) {
    for (int k = 0; k < 3; ++k) {
      const Map &m = map(layer, k);
      if (m.base) ::madvise(m.base + static_cast<uint64_t>(expert) * m.stride, static_cast<size_t>(m.stride), MADV_DONTNEED);
    }
  }
  // Resident 4 KiB pages of one expert's three planes (mincore).
  size_t resident_pages(int layer, int expert) const {
    size_t n = 0;
    for (int k = 0; k < 3; ++k) {
      const Map &m = map(layer, k);
      if (!m.base) continue;
      std::vector<unsigned char> v(static_cast<size_t>(m.stride / kPage));
      if (::mincore(m.base + static_cast<uint64_t>(expert) * m.stride, static_cast<size_t>(m.stride), v.data()) == 0)
        for (unsigned char c : v) n += c & 1u;
    }
    return n;
  }

private:
  static constexpr uint64_t kPage = 4096;
  struct Map { char *base = nullptr; uint64_t plane = 0, stride = 0, len = 0; };
  const Map &map(int layer, int plane) const { return maps_[static_cast<size_t>(layer - first_) * 3 + static_cast<size_t>(plane)]; }
  int first_, n_layers_, n_experts_;
  bool ok_ = true;
  std::vector<Map> maps_;
};

// ---------------------------------------------------------------------------------------------------- residency
class ExpertResidency {
public:
  struct Ensure {
    int misses = 0, evictions = 0, loads = 0, errors = 0, not_cached = 0;
    double load_ms = 0.0;
  };
  // `sources` are tried in order; the first one that contains the expert supplies it (store first, model as fallback).
  ExpertResidency(ExpertCache &cache, VirtualArena &arena, std::vector<ExpertSource *> sources)
      : cache_(cache), arena_(arena), sources_(std::move(sources)) {}

  // Make every non-negative id of `layer` resident. Misses are loaded synchronously. A miss that cannot be loaded (no
  // source, size mismatch, I/O error) is removed from the cache and counted in `errors`: the caller must fail the request
  // rather than compute with a slot that does not hold the expert. `not_cached` counts misses of a Static policy cache.
  Ensure ensure(int layer, const int32_t *ids, int n) {
    Ensure r;
    for (int i = 0; i < n; ++i) {
      if (ids[i] < 0) continue;
      const ExpertKey key{layer, ids[i]};
      const ExpertCache::Access a = cache_.access(key);
      if (a.hit) continue;
      ++r.misses;
      if (cache_.config().policy == ExpertCache::Policy::Static) { ++r.not_cached; continue; }
      if (a.evicted) { arena_.evict(a.victim.layer, a.victim.expert); ++r.evictions; }
      if (!a.inserted) { ++r.errors; continue; }
      const auto t0 = std::chrono::steady_clock::now();
      const bool ok = load(layer, ids[i]);
      r.load_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      if (ok) ++r.loads;
      else { cache_.erase(key); arena_.evict(layer, ids[i]); ++r.errors; }
    }
    return r;
  }

  // Fill an expert's three planes from the first source that has it.
  bool load(int layer, int expert) {
    for (ExpertSource *s : sources_) {
      if (!s || !s->contains(layer, expert)) continue;
      uint64_t pb[3];
      if (!s->plane_bytes(layer, expert, pb)) continue;
      void *dst[3];
      for (int k = 0; k < 3; ++k) {
        if (pb[k] != arena_.plane_bytes(layer, k)) return false;
        dst[k] = arena_.ptr(layer, expert, k);
        if (!dst[k]) return false;
      }
      return s->read_expert(layer, expert, dst) == ReadStatus::Ok;
    }
    return false;
  }

  // Load every currently resident (seeded) expert; returns how many could not be loaded (they are dropped).
  size_t populate_resident() {
    size_t failed = 0;
    const ExpertCache::Config &c = cache_.config();
    for (int L = c.first_layer; L < c.first_layer + c.n_layers; ++L)
      for (int E = 0; E < c.n_experts; ++E)
        if (cache_.resident({L, E}) && !load(L, E)) { cache_.erase({L, E}); ++failed; }
    return failed;
  }

private:
  ExpertCache &cache_;
  VirtualArena &arena_;
  std::vector<ExpertSource *> sources_;
};

} // namespace moestream::offload

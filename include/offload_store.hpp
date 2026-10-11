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
#include <filesystem>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <unordered_set>
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
    const int fb = ::open(path.c_str(), O_RDONLY);
    if (fb < 0) return fail("cannot open for reading");
    fds_buf_.push_back(fb);
    fds_dir_.push_back(try_direct ? ::open(path.c_str(), O_RDONLY | O_DIRECT) : -1);
    return true;
  }

  // Embedded mode: the tight per-expert blocks (offload.blk.<L>.ffn_<k>.<E>.weight) that `gguf_tool pack --expert-interleave`
  // writes INTO the model shards, next to the normal tensors. Every shard is scanned; tensors that are not offload.* are ignored.
  // An expert whose three planes are not all present in one shard is an error (the tool never splits a layer across shards).
  bool open_embedded(const std::vector<std::string> &paths, std::string *err = nullptr, bool try_direct = true) {
    close_fds();
    slabs_.clear();
    auto fail = [&](const std::string &m) { close_fds(); slabs_.clear(); if (err) *err = m; return false; };
    for (size_t f = 0; f < paths.size(); ++f) {
      GgufHeader h;
      if (!read_gguf_header(paths[f], h, err)) return fail(err && !err->empty() ? *err : "cannot read " + paths[f]);
      if (f == 0 || h.alignment > align_) align_ = f == 0 ? h.alignment : std::max(align_, h.alignment);
      struct Part { uint64_t off[3] = {0, 0, 0}, bytes[3] = {0, 0, 0}; int seen = 0; };
      std::unordered_map<uint64_t, Part> parts;
      for (const GgufTensorInfo &t : h.tensors) {
        int L = 0, E = 0;
        char kind[8] = {0};
        if (t.name.rfind("offload.blk.", 0) != 0) continue;
        if (std::sscanf(t.name.c_str(), "offload.blk.%d.ffn_%7[a-z].%d.weight", &L, kind, &E) != 3) continue;
        const int k = !std::strcmp(kind, "gate") ? 0 : !std::strcmp(kind, "up") ? 1 : !std::strcmp(kind, "down") ? 2 : -1;
        if (k < 0 || t.dims.size() < 2) return fail("bad tensor " + t.name + " in " + paths[f]);
        const uint64_t bytes = ggml_row_bytes(t.type, t.dims[0]) * t.dims[1];
        if (bytes == 0) return fail("unknown quant type in " + t.name);
        Part &p = parts[detail::key(L, E)];
        p.off[k] = h.data_start + t.offset;
        p.bytes[k] = bytes;
        p.seen |= 1 << k;
      }
      if (parts.empty()) continue;
      const int fb = ::open(paths[f].c_str(), O_RDONLY);
      if (fb < 0) return fail("cannot open " + paths[f]);
      fds_buf_.push_back(fb);
      fds_dir_.push_back(try_direct ? ::open(paths[f].c_str(), O_RDONLY | O_DIRECT) : -1);
      const int file = static_cast<int>(fds_buf_.size()) - 1;
      for (auto &[key, p] : parts) {
        if (p.seen != 7) return fail("an expert is missing a projection in " + paths[f]);
        Slab s;
        for (int k = 0; k < 3; ++k) { s.off[k] = p.off[k]; s.bytes[k] = p.bytes[k]; }
        s.tight = p.off[1] == p.off[0] + p.bytes[0] && p.off[2] == p.off[1] + p.bytes[1];
        s.file = file;
        slabs_[key] = s;
      }
    }
    if (slabs_.empty()) return fail("no offload.* expert blocks in the given files");
    return true;
  }

  // Open either a stand-alone store (architecture moestream-offload) or the shards of a model that embeds the blocks. `path` may be a
  // file or a directory of shards.
  bool open_auto(const std::string &path, std::string *err = nullptr, bool try_direct = true) {
    std::error_code ec;
    if (std::filesystem::is_directory(path, ec)) {
      std::vector<std::string> files;
      for (const auto &e : std::filesystem::directory_iterator(path))
        if (e.is_regular_file() && e.path().extension() == ".gguf") files.push_back(e.path().string());
      std::sort(files.begin(), files.end());
      return open_embedded(files, err, try_direct);
    }
    GgufHeader h;
    if (read_gguf_header(path, h, nullptr) && h.architecture == "moestream-offload") return open(path, err, try_direct);
    return open_embedded({path}, err, try_direct);
  }

  size_t size() const { return slabs_.size(); }
  uint32_t alignment() const { return align_; }
  bool direct() const { return !fds_dir_.empty() && fds_dir_[0] >= 0; }
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
    const int fd_dir = fds_dir_[static_cast<size_t>(s.file)], fd_buf = fds_buf_[static_cast<size_t>(s.file)];
    bool fast = fd_dir >= 0 && s.tight && s.off[0] % align_ == 0;
    for (int k = 0; k < 3 && fast; ++k)
      if (reinterpret_cast<uintptr_t>(dst[k]) % align_ != 0 || s.bytes[k] % align_ != 0) fast = false;
    if (fast) {
      struct iovec iov[3] = {{dst[0], static_cast<size_t>(s.bytes[0])}, {dst[1], static_cast<size_t>(s.bytes[1])}, {dst[2], static_cast<size_t>(s.bytes[2])}};
      if (detail::preadv_all(fd_dir, iov, 3, static_cast<off_t>(s.off[0]))) { ++direct_reads_; return ReadStatus::Ok; }
      // fall through to the buffered path (O_DIRECT can refuse on some filesystems)
    }
    for (int k = 0; k < 3; ++k) // each plane at its own offset: a store need not be tight
      if (!detail::pread_all(fd_buf, dst[k], static_cast<size_t>(s.bytes[k]), static_cast<off_t>(s.off[k]))) return ReadStatus::IoError;
    ++buffered_reads_;
    return ReadStatus::Ok;
  }

private:
  struct Slab {
    uint64_t off[3] = {0, 0, 0}, bytes[3] = {0, 0, 0}; // absolute file offset and size of each plane
    bool tight = false;                                 // planes back to back: one preadv can fetch the whole expert
    int file = 0;                                       // index into fds_dir_/fds_buf_ (embedded mode spans several shards)
  };
  void close_fds() {
    for (int fd : fds_dir_) if (fd >= 0) ::close(fd);
    for (int fd : fds_buf_) if (fd >= 0) ::close(fd);
    fds_dir_.clear();
    fds_buf_.clear();
  }
  std::unordered_map<uint64_t, Slab> slabs_;
  uint32_t align_ = 4096;
  std::vector<int> fds_dir_, fds_buf_;
  uint64_t direct_reads_ = 0, buffered_reads_ = 0;
};

class ModelReader final : public ExpertSource {
public:
  bool drop_page_cache = true;
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
    // The three planes sit at unrelated file offsets: read them concurrently (queue depth 3) instead of back to back.
    auto one = [&](int k) {
      const Proj &p = layers_.at(layer).p[k];
      const off_t off = static_cast<off_t>(p.offset + static_cast<uint64_t>(expert) * p.per_expert);
      if (!detail::pread_all(p.fd, dst[k], static_cast<size_t>(p.per_expert), off)) return false;
      // The arena is the DDR tier; keeping the same bytes in the page cache would double-count them and evict it.
      if (drop_page_cache) ::posix_fadvise(p.fd, off, static_cast<off_t>(p.per_expert), POSIX_FADV_DONTNEED);
      return true;
    };
    auto f1 = std::async(std::launch::async, one, 1);
    auto f2 = std::async(std::launch::async, one, 2);
    const bool ok0 = one(0);
    const bool ok1 = f1.get();
    const bool ok2 = f2.get();
    if (!(ok0 && ok1 && ok2)) return ReadStatus::IoError;
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
  // Optional integration hooks (e.g. pinning the slot for DMA): after_populate runs once an expert's planes hold valid data,
  // before_evict runs right before its pages are returned to the OS.
  std::function<void(int, int)> after_populate, before_evict;
  void populated(int layer, int expert) { if (after_populate) after_populate(layer, expert); }
  // Return one expert's pages to the OS; the slot reads as zeros until it is populated again.
  void evict(int layer, int expert) {
    if (before_evict) before_evict(layer, expert);
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
  // `kind` tells a segmented cache whether these ids were really routed (Demand) or only predicted (Prefetch).
  Ensure ensure(int layer, const int32_t *ids, int n, ExpertCache::Kind kind = ExpertCache::Kind::Demand) {
    Ensure r;
    std::vector<int> to_load;
    for (int i = 0; i < n; ++i) {
      if (ids[i] < 0) continue;
      const ExpertKey key{layer, ids[i]};
      const ExpertCache::Access a = cache_.access(key, kind);
      if (a.hit) continue;
      ++r.misses;
      if (cache_.config().policy == ExpertCache::Policy::Static) { ++r.not_cached; continue; }
      for (const ExpertKey &v : a.victims) { arena_.evict(v.layer, v.expert); ++r.evictions; }
      if (!a.inserted) { ++r.errors; continue; }
      to_load.push_back(ids[i]);
    }
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<char> ok = load_many(layer, to_load);
    r.load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    for (size_t k = 0; k < to_load.size(); ++k) {
      if (ok[k]) { ++r.loads; arena_.populated(layer, to_load[k]); }
      else { cache_.erase({layer, to_load[k]}); arena_.evict(layer, to_load[k]); ++r.errors; }
    }
    return r;
  }

  // ---- Asynchronous protocol -------------------------------------------------------------------------------------
  // ensure() above does all of its work (cache bookkeeping AND the slow reads) under the caller's arena lock, so a
  // background prefetch and a demand miss serialize on I/O. begin()/finish() split it: bookkeeping stays under the lock,
  // the reads run without it, and an expert another thread is still reading is marked "loading" so nobody computes with
  // (or evicts) a half-filled slot.
  struct Plan {
    Ensure r;
    std::vector<int> to_load; // misses this call must read
    std::vector<int> to_wait; // experts another thread is reading that this (demand) call needs
    int layer = 0;
  };

  // Phase 1. The caller holds the arena lock exclusively. Bookkeeping only, no I/O.
  Plan begin(int layer, const int32_t *ids, int n, ExpertCache::Kind kind = ExpertCache::Kind::Demand) {
    Plan p;
    p.layer = layer;
    for (int i = 0; i < n; ++i) {
      if (ids[i] < 0) continue;
      const ExpertKey key{layer, ids[i]};
      const bool ld = is_loading(key);
      if (ld && !cache_.resident(key)) { // evicted while a read was in flight: wait for it, then load again
        if (kind == ExpertCache::Kind::Demand) p.to_wait.push_back(ids[i]);
        continue;
      }
      const ExpertCache::Access a = cache_.access(key, kind);
      if (a.hit) {
        if (ld && kind == ExpertCache::Kind::Demand) p.to_wait.push_back(ids[i]);
        continue;
      }
      ++p.r.misses;
      if (cache_.config().policy == ExpertCache::Policy::Static) { ++p.r.not_cached; continue; }
      for (const ExpertKey &v : a.victims) {
        if (!defer_evict_if_loading(v)) arena_.evict(v.layer, v.expert);
        ++p.r.evictions;
      }
      if (!a.inserted) { ++p.r.errors; continue; }
      mark_loading(key);
      p.to_load.push_back(ids[i]);
    }
    return p;
  }

  // Phase 2. Call WITHOUT the arena lock. `lock()` must return an RAII holder of the exclusive arena lock; it is taken
  // only for the short publish step after the reads. Waits for experts other threads are loading and re-loads any of
  // those that failed or were evicted meanwhile.
  template <class LockFn>
  Ensure finish(Plan &p, LockFn &&lock, int depth = 0) {
    Ensure r = p.r;
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<char> ok = load_many(p.layer, p.to_load);
    r.load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!p.to_load.empty()) {
      {
        auto guard = lock();
        for (size_t k = 0; k < p.to_load.size(); ++k) {
          const ExpertKey key{p.layer, p.to_load[k]};
          if (take_deferred(key)) { arena_.evict(key.layer, key.expert); if (ok[k]) ++r.loads; continue; }
          if (ok[k]) { arena_.populated(key.layer, key.expert); ++r.loads; }
          else { cache_.erase(key); arena_.evict(key.layer, key.expert); ++r.errors; }
        }
      }
      {
        std::lock_guard<std::mutex> g(ld_mu_);
        for (int e : p.to_load) loading_.erase(kid({p.layer, e}));
      }
      ld_cv_.notify_all();
    }
    if (!p.to_wait.empty()) {
      const auto w0 = std::chrono::steady_clock::now();
      {
        std::unique_lock<std::mutex> g(ld_mu_);
        ld_cv_.wait(g, [&] {
          for (int e : p.to_wait) if (loading_.count(kid({p.layer, e}))) return false;
          return true;
        });
      }
      waited_.fetch_add(p.to_wait.size(), std::memory_order_relaxed);
      wait_us_.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - w0).count()), std::memory_order_relaxed);
      std::vector<int> retry;
      {
        auto guard = lock();
        for (int e : p.to_wait) if (!cache_.resident({p.layer, e})) retry.push_back(e);
      }
      if (!retry.empty()) {
        if (depth >= 3) { r.errors += static_cast<int>(retry.size()); return r; }
        Plan q;
        {
          auto guard = lock();
          q = begin(p.layer, retry.data(), static_cast<int>(retry.size()), ExpertCache::Kind::Demand);
        }
        const Ensure e2 = finish(q, lock, depth + 1);
        r.misses += e2.misses; r.evictions += e2.evictions; r.loads += e2.loads; r.errors += e2.errors;
        r.not_cached += e2.not_cached; r.load_ms += e2.load_ms;
      }
    }
    return r;
  }

  // Demand lookups that found their expert still being read by another thread, and the time spent waiting for them.
  uint64_t waited_experts() const { return waited_.load(std::memory_order_relaxed); }
  double wait_ms() const { return static_cast<double>(wait_us_.load(std::memory_order_relaxed)) / 1000.0; }
  // True when the expert is resident AND its planes hold valid data (not mid-read). Caller holds the arena lock.
  bool ready(ExpertKey k) const { return cache_.resident(k) && !is_loading(k); }
  bool is_loading(ExpertKey k) const {
    std::lock_guard<std::mutex> g(ld_mu_);
    return loading_.count(kid(k)) != 0;
  }

  // Load several experts of one layer concurrently (the first on the calling thread).
  std::vector<char> load_many(int layer, const std::vector<int> &experts) {
    std::vector<char> ok(experts.size(), 0);
    std::vector<std::future<bool>> fut;
    for (size_t k = 1; k < experts.size(); ++k)
      fut.push_back(std::async(std::launch::async, [this, layer, e = experts[k]] { return load(layer, e); }));
    if (!experts.empty()) ok[0] = load(layer, experts[0]);
    for (size_t k = 1; k < experts.size(); ++k) ok[k] = fut[k - 1].get();
    return ok;
  }

  // Prefill path: make every id of `layer` readable without touching the cache (a prompt touches most experts of a
  // layer once, so caching them would only evict the decode working set). Experts loaded here are dropped again at the
  // next ensure_transient() call, i.e. when the following layer starts.
  Ensure ensure_transient(int layer, const int32_t *ids, int n) {
    Ensure r;
    for (const ExpertKey &k : transient_) arena_.evict(k.layer, k.expert);
    transient_.clear();
    for (int i = 0; i < n; ++i) {
      if (ids[i] < 0) continue;
      const ExpertKey key{layer, ids[i]};
      if (cache_.resident(key)) continue;
      bool dup = false;
      for (const ExpertKey &t : transient_) if (t.layer == key.layer && t.expert == key.expert) { dup = true; break; }
      if (dup) continue;
      ++r.misses;
      const auto t0 = std::chrono::steady_clock::now();
      const bool ok = load(layer, ids[i]);
      r.load_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      transient_.push_back(key);
      if (ok) ++r.loads; else ++r.errors;
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
        if (cache_.resident({L, E})) {
          if (load(L, E)) arena_.populated(L, E);
          else { cache_.erase({L, E}); ++failed; }
        }
    return failed;
  }

private:
  static uint64_t kid(ExpertKey k) { return (static_cast<uint64_t>(static_cast<uint32_t>(k.layer)) << 32) | static_cast<uint32_t>(k.expert); }
  void mark_loading(ExpertKey k) { std::lock_guard<std::mutex> g(ld_mu_); loading_.insert(kid(k)); }
  // A victim whose read is still in flight cannot have its pages returned yet; the reader frees the slot when it ends.
  bool defer_evict_if_loading(ExpertKey k) {
    std::lock_guard<std::mutex> g(ld_mu_);
    if (!loading_.count(kid(k))) return false;
    deferred_.insert(kid(k));
    return true;
  }
  bool take_deferred(ExpertKey k) {
    std::lock_guard<std::mutex> g(ld_mu_);
    return deferred_.erase(kid(k)) != 0;
  }

  ExpertCache &cache_;
  VirtualArena &arena_;
  std::vector<ExpertSource *> sources_;
  std::vector<ExpertKey> transient_;
  mutable std::mutex ld_mu_;          // leaf lock: never held while taking another lock
  std::condition_variable ld_cv_;
  std::unordered_set<uint64_t> loading_, deferred_;
  std::atomic<uint64_t> waited_{0}, wait_us_{0};
};

} // namespace moestream::offload

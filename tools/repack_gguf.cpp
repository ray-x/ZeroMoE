// tools/repack_gguf.cpp
//
// Offline GGUF repack tool. Processes GGUF shard(s) ONE FILE AT A TIME --
// pass a single shard file to process just that one (see
// process_one_shard below), or pass a DIRECTORY and it walks every .gguf
// file in it (sorted) and processes them one after another, identical to
// invoking this tool once per file by hand (see main's directory-mode
// branch). Either way, only one source shard + its output ever need to
// exist on disk at once, so peak disk usage stays roughly "one source
// shard + its output" instead of "the whole model, twice" -- combine
// with -d/--delete-source to free each shard automatically as soon as
// its own verification passes, so a full directory-mode run can convert
// an entire multi-shard model unattended within that same disk budget.
// Each shard's tensors get rewritten into a new GGUF model where:
//
//   1. Every packed routed-expert tensor found IN THAT FILE (Unsloth-
//      style blk.N.ffn_{gate,up,down}_exps.weight, shape
//      [a, b, n_experts]) is unpacked into n_experts individual tensors
//      (blk.N.ffn_{gate,up,down}.E.weight) -- exactly the "legacy
//      per-expert" layout core/engine.hpp's load_expert_slab already
//      knows how to load as a single combined pread per expert.
//   2. general.alignment is set to a large value (default 4096) so every
//      tensor -- including each newly-unpacked expert -- starts at a
//      filesystem-block-aligned offset, the prerequisite for O_DIRECT
//      reads.
//
// A layer's gate/up/down projections are unpacked independently of each
// other -- this tool does NOT require all three to be found in the same
// invocation's file, since when processing shards one at a time they
// could in principle live in different source files (in practice, for
// typical GGUF exports, a layer's tensors usually stay together in one
// shard, but nothing here assumes that).
//
// This is a pure LAYOUT change. Every quantized byte is copied verbatim
// from source to output -- no requantization, no precision change, no
// value is reinterpreted. See CLAUDE.md's "repack tool" discussion for
// the full design rationale and the two correctness gaps this tool exists
// to close (GGUFFile::read_kv_value silently discards array-typed KV
// entries -- would lose the entire tokenizer vocabulary on a naive reuse
// -- and the loader used to hardcode 32-byte alignment, both fixed
// alongside this tool; see gguf_loader.hpp's general.alignment read).
//
// Usage:
//   repack_gguf <source_shard_file_or_dir> <output_dir> [--align N]
//   [--max-shard-gb N] [--verify-samples N] [-d|--delete-source]
//
// Design:
//   - Reuses GGUFFile::load_header (the real, already-proven loader),
//     pointed at a SINGLE FILE -- it already supports this directly (the
//     "is_directory ? glob : push_back(path)" branch), treating the file
//     as a self-contained one-shard model with correctly-computed tensor
//     offsets. That part was never the problem, no need to reimplement it.
//   - Adds its OWN raw KV reader (read_raw_kv_section below) purely to
//     capture what GGUFFile's loader discards: array values, needed to
//     reproduce every KV entry byte-for-byte in this file's own output.
//   - Cannot be tested against a real multi-hundred-GB model on this
//     machine (darwin, no CUDA, no real GGUF file available) -- the
//     --verify pass at the end (byte-comparing a sample of tensors and
//     confirming tensor counts) is NOT optional. Read its output before
//     trusting any repacked file for real use.

#include "gguf_loader.hpp"
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

// ============================================================
// Raw, byte-exact KV value capture (see file header comment for why
// GGUFFile's own loader can't be reused for this part).
// ============================================================

struct RawKV {
  std::string key;
  uint32_t type;                    // GGUF value type id, 0-12
  std::vector<uint8_t> value_bytes; // exact on-disk bytes for the VALUE
                                    // (key+type already consumed)
};

static uint64_t fixed_scalar_width(uint32_t type) {
  switch (type) {
  case 0:
  case 1:
  case 7:
    return 1; // UINT8 / INT8 / BOOL
  case 2:
  case 3:
    return 2; // UINT16 / INT16
  case 4:
  case 5:
  case 6:
    return 4; // UINT32 / INT32 / FLOAT32
  case 10:
  case 11:
  case 12:
    return 8; // UINT64 / INT64 / FLOAT64
  default:
    return 0; // STRING(8) / ARRAY(9): variable-length
  }
}

static std::string read_gguf_string(std::ifstream &f) {
  uint64_t len = 0;
  f.read(reinterpret_cast<char *>(&len), 8);
  std::string s(len, '\0');
  if (len > 0)
    f.read(&s[0], (std::streamsize)len);
  return s;
}

// Reads and returns the raw on-disk bytes for ONE value of the given
// type, starting at the current stream position. GGUF does not allow
// ARRAY-of-ARRAY, so this recursion is bounded to depth 2.
static std::vector<uint8_t> read_raw_value(std::ifstream &f, uint32_t type) {
  uint64_t w = fixed_scalar_width(type);
  if (w > 0) {
    std::vector<uint8_t> out(w);
    f.read(reinterpret_cast<char *>(out.data()),
           static_cast<std::streamsize>(w));
    return out;
  }
  if (type == 8) { // STRING: 8-byte length + bytes
    uint64_t len = 0;
    f.read(reinterpret_cast<char *>(&len), 8);
    std::vector<uint8_t> out(8 + len);
    std::memcpy(out.data(), &len, 8);
    if (len > 0)
      f.read(reinterpret_cast<char *>(out.data() + 8),
             static_cast<std::streamsize>(len));
    return out;
  }
  if (type == 9) { // ARRAY: 4-byte sub_type + 8-byte count + elements
    uint32_t sub_type = 0;
    uint64_t count = 0;
    f.read(reinterpret_cast<char *>(&sub_type), 4);
    f.read(reinterpret_cast<char *>(&count), 8);
    std::vector<uint8_t> out(12);
    std::memcpy(out.data(), &sub_type, 4);
    std::memcpy(out.data() + 4, &count, 8);
    out.reserve(out.size() +
                count * 4); // rough hint, exact for common scalar sub-types
    for (uint64_t i = 0; i < count; ++i) {
      auto elem = read_raw_value(f, sub_type);
      out.insert(out.end(), elem.begin(), elem.end());
    }
    return out;
  }
  throw std::runtime_error("[repack] FATAL: unknown GGUF KV value type " +
                           std::to_string(type) +
                           " -- refusing to guess, aborting");
}

// Reads every KV entry of ONE shard file's header, verbatim, in order.
// Also returns that shard's own tensor_count (from the header, not used
// for tensor parsing here -- GGUFFile::load_header already did that
// correctly -- only kept so callers can sanity-check shard boundaries).
static std::vector<RawKV> read_raw_kv_section(const std::string &path,
                                              uint64_t &out_tensor_count) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    throw std::runtime_error("[repack] FATAL: cannot open " + path);
  }
  std::array<char, 4> magic{};
  f.read(magic.data(), magic.size());
  if (std::string_view(magic.data(), magic.size()) != "GGUF") {
    throw std::runtime_error("[repack] FATAL: " + path +
                             " is not a GGUF file (bad magic)");
  }
  uint32_t version = 0;
  f.read(reinterpret_cast<char *>(&version), 4);
  uint64_t tensor_count = 0, kv_count = 0;
  f.read(reinterpret_cast<char *>(&tensor_count), 8);
  f.read(reinterpret_cast<char *>(&kv_count), 8);
  out_tensor_count = tensor_count;

  std::vector<RawKV> kvs;
  kvs.reserve(kv_count);
  for (uint64_t i = 0; i < kv_count; ++i) {
    RawKV kv;
    kv.key = read_gguf_string(f);
    f.read(reinterpret_cast<char *>(&kv.type), 4);
    kv.value_bytes = read_raw_value(f, kv.type);
    kvs.push_back(std::move(kv));
  }
  return kvs;
}
static RawKV make_uint16_kv(const std::string &key, uint16_t value) {
  RawKV kv;
  kv.key = key;
  kv.type = 2; // UINT16 (GGUF Type ID 2)
  kv.value_bytes.resize(2);
  std::memcpy(kv.value_bytes.data(), &value, 2);
  return kv;
}
static RawKV make_uint32_kv(const std::string &key, uint32_t value) {
  RawKV kv;
  kv.key = key;
  kv.type = 4; // UINT32
  kv.value_bytes.resize(4);
  std::memcpy(kv.value_bytes.data(), &value, 4);
  return kv;
}

static void write_raw_kv(std::ofstream &out, const RawKV &kv) {
  uint64_t klen = kv.key.size();
  out.write(reinterpret_cast<const char *>(&klen), 8);
  out.write(kv.key.data(), static_cast<std::streamsize>(klen));
  out.write(reinterpret_cast<const char *>(&kv.type), 4);
  out.write(reinterpret_cast<const char *>(kv.value_bytes.data()),
            static_cast<std::streamsize>(kv.value_bytes.size()));
}

static inline uint64_t align_up(uint64_t v, uint64_t alignment) {
  return (v + alignment - 1) & ~(alignment - 1);
}

// Fixed-width zero-padded integer, so lexicographic string comparison of
// composite sort keys orders numbers numerically (not "10" < "2"). 8 digits
// is far beyond any real layer count or expert count this convention will
// ever see.
static std::string pad_sort(int v) {
  std::ostringstream oss;
  oss << std::setw(8) << std::setfill('0') << v;
  return oss.str();
}

// ============================================================
// Output tensor plan
// ============================================================

struct OutTensor {
  std::string name;
  GGUFType type;
  std::vector<uint64_t> shape; // GGUF ne0,ne1,... order, as stored
  int src_shard_idx;
  uint64_t src_offset; // absolute byte offset in the SOURCE shard file
  uint64_t size_bytes;
  int out_shard_idx = -1;
  uint64_t out_offset = 0; // relative to that output shard's data_start
  // Physical-layout sort key (NOT the tensor name). Per-expert MoE tensors
  // MUST be laid out contiguously per expert in gate->up->down order: the
  // engine's per-expert loader reads a combined per-expert slab and (more
  // importantly for streaming) wants the three projections adjacent so the
  // kernel coalesces their reads. A plain lexicographic NAME sort orders
  // them down<gate<up ("ffn_down"<"ffn_gate"<"ffn_up"), splitting each
  // expert's projections across distant file regions. Per-expert tensors
  // therefore key on (layer, expert, projection-rank); everything else keys
  // on its name. See the sort below.
  std::string sort_key;
};

// Processes exactly ONE source shard file: the full pipeline (load,
// discover+unpack packed experts, plan layout, write, verify, optionally
// delete). Returns 0 on success, matching this project's process-exit-code
// convention (1 = a real failure, 2 = a usage/argument error -- though
// argument validation itself lives in main(), before this is ever called).
// Factored out of main() so directory mode (below) can call this once per
// discovered shard, in a loop, with identical behavior to running this
// tool once per file by hand.
int process_one_shard(const std::string &input_path,
                      const std::string &output_dir, uint64_t alignment,
                      uint64_t max_shard_bytes, int verify_samples,
                      bool delete_source) {
  // ---- Step 1: reuse the real, proven loader for the full, correctly-
  // offset tensor table. Deliberately NOT using src.architecture_config
  // here -- that struct is only ever populated by
  // GGUFGraphInitializer::build_model_graph (graph_initializer.cpp),
  // never by load_header alone, so it would silently stay all-zero for
  // a standalone tool that only calls load_header (a real bug caught by
  // actually compiling and dry-running this tool, not by static review
  // alone). Layer/expert discovery below is architecture-agnostic
  // instead: scan tensor NAMES for the blk.N.ffn_{gate,up,down}_exps
  // convention directly, and derive n_experts from the packed tensors'
  // own shape (shape.back(), already established as the expert-count
  // dimension) rather than any KV-key guessing -- works for any
  // architecture using this packing convention, not just GLM-5.2. ----
  // input_path is a SINGLE SOURCE SHARD FILE, processed independently of
  // every other shard -- see the file header comment. load_header
  // already supports being pointed at one file directly (it just treats
  // it as a self-contained one-shard model, taking the "if
  // (is_directory) ... else { gguf_files.push_back(path); }" branch),
  // so this needs no change on the loader side to work per-file.
  GGUFFile src;
  if (!src.load_header(input_path)) {
    std::cerr << "[repack] FATAL: failed to load source GGUF file at "
              << input_path << '\n';
    return 1;
  }
  std::cout << "[repack] Source file: " << input_path << " ("
            << src.tensors.size() << " tensor(s))\n";

  // Identify every packed expert tensor PRESENT IN THIS FILE, one
  // projection at a time -- deliberately NOT requiring gate/up/down for
  // the same layer to all be found together. When processing shards one
  // at a time, a layer's three projections could in principle be split
  // across different source files; each one found here is unpacked
  // independently regardless of whether its siblings live in this same
  // file or a different one already processed (or yet to be). A file
  // with NO packed tensors at all (a dense-only shard, or the metadata-
  // only shard some multi-shard layouts use) is completely normal, not
  // an error -- everything in it just passes through unchanged below.
  int n_experts = -1;

  // ---- Step 2: this tool's own byte-exact KV pass, per shard. ----
  // Per observed convention (and GGUF's own multi-shard design): shard 0
  // carries the full metadata; every shard (including shard 0) also
  // carries its own split.count/split.no/split.tensors.count triplet.
  // We don't rely on that pattern holding exactly -- we just capture
  // whatever each shard actually has, verbatim, and decide what to keep
  // vs. recompute below.
  std::vector<std::vector<RawKV>> shard_kvs(src.shard_paths.size());
  for (size_t s = 0; s < src.shard_paths.size(); ++s) {
    uint64_t tc = 0;
    shard_kvs[s] = read_raw_kv_section(src.shard_paths[s], tc);
  }
  // "Full" metadata = whichever shard has the most KV entries (matches
  // the observed "shard 0 carries everything, others carry just
  // split.*" pattern without hardcoding shard index 0 specifically).
  size_t full_meta_shard = 0;
  for (size_t s = 1; s < shard_kvs.size(); ++s) {
    if (shard_kvs[s].size() > shard_kvs[full_meta_shard].size())
      full_meta_shard = s;
  }
  std::vector<RawKV> full_metadata = shard_kvs[full_meta_shard];
  // Strip split.* and general.alignment from the copied metadata -- both
  // get regenerated below for the NEW shard structure/alignment, never
  // just carried over from source (source shard count/alignment almost
  // certainly differ from the output's).
  auto strip_key = [](std::vector<RawKV> &kvs, const std::string &key) {
    kvs.erase(std::remove_if(kvs.begin(), kvs.end(),
                             [&](const RawKV &kv) { return kv.key == key; }),
              kvs.end());
  };
  for (const auto *key : {"split.count", "split.no", "split.tensors.count",
                          "general.alignment"}) {
    strip_key(full_metadata, key);
  }

  std::cout << "[repack] Captured " << full_metadata.size()
            << " metadata KV entries verbatim (source shard " << full_meta_shard
            << " had the most: " << shard_kvs[full_meta_shard].size()
            << " total)\n";

  // ---- Step 3: build the output tensor plan. ----
  std::vector<OutTensor> out_tensors;
  out_tensors.reserve(src.tensors.size() * 4);

  // Deterministic order: sort source tensor names lexicographically.
  // Physical byte order in the output has no correctness meaning (only
  // the tensor table's declared offsets matter) -- this just makes the
  // output reproducible run-to-run for the same input.
  std::vector<std::string> names;
  names.reserve(src.tensors.size());
  for (const auto &[name, t] : src.tensors)
    names.push_back(name);
  std::sort(names.begin(), names.end());

  // Identify every packed expert tensor present in THIS file, by name
  // pattern -- one projection at a time, independent of its siblings
  // (see this block's own comment above for why co-presence is not
  // required in single-file mode).
  std::unordered_set<std::string> packed_names;
  int packed_tensors_found = 0;
  const std::string suffix = "_exps.weight";
  for (auto &name : names) {
    if (name.rfind("blk.", 0) != 0)
      continue;
    size_t suf_pos = name.rfind(suffix);
    if (suf_pos == std::string::npos || suf_pos + suffix.size() != name.size())
      continue;
    // Must be specifically ffn_{gate,up,down}_exps.weight -- a name
    // ending in "_exps.weight" for some OTHER reason (unlikely, but
    // don't silently misclassify it) is left as a passthrough tensor
    // instead of guessed at.
    bool is_gate = name.find(".ffn_gate_exps.weight") != std::string::npos;
    bool is_up = name.find(".ffn_up_exps.weight") != std::string::npos;
    bool is_down = name.find(".ffn_down_exps.weight") != std::string::npos;
    if (!is_gate && !is_up && !is_down)
      continue;

    const GGUFTensor &pt = src.tensors.at(name);
    if (pt.shape.empty()) {
      fprintf(stderr, "[repack] FATAL: %s has an empty shape\n", name.c_str());
      return 1;
    }
    // Derive n_experts from this tensor's own packed shape (already
    // established: [a, b, n_experts], expert count is the LAST
    // dimension) instead of any KV-key lookup -- and require every
    // packed tensor found in this file to agree, since a real
    // mismatch would mean something is structurally wrong with the
    // source, not something to silently paper over.
    int this_n = (int)pt.shape.back();
    if (n_experts < 0) {
      n_experts = this_n;
    } else if (this_n != n_experts) {
      fprintf(stderr,
              "[repack] FATAL: %s shape ends in %d, but an earlier tensor in "
              "this same "
              "file established n_experts=%d -- refusing to guess\n",
              name.c_str(), this_n, n_experts);
      return 1;
    }
    if (pt.size_bytes % (uint64_t)n_experts != 0) {
      fprintf(stderr,
              "[repack] FATAL: %s size_bytes (%llu) not evenly divisible by "
              "n_experts (%d)\n",
              name.c_str(), (unsigned long long)pt.size_bytes, n_experts);
      return 1;
    }
    uint64_t per_expert_bytes = pt.size_bytes / (uint64_t)n_experts;
    std::vector<uint64_t> per_expert_shape(pt.shape.begin(),
                                           pt.shape.end() - 1);
    std::string base_no_exps = name.substr(
        0, suf_pos); // "blk.5.ffn_gate_exps.weight" -> "blk.5.ffn_gate"

    // Sort-key inputs for this packed tensor's per-expert children
    // (constant across the e-loop below): layer from the "blk.<L>." prefix,
    // projection rank gate=0/up=1/down=2 so each expert's three projections
    // land CONTIGUOUS and in gate->up->down order on disk (a lexicographic
    // name sort would order them down<gate<up). See OutTensor::sort_key.
    int layer_num = 0;
    {
      size_t p = 4;
      int v = 0; // skip "blk."
      while (p < name.size() && name[p] >= '0' && name[p] <= '9') {
        v = v * 10 + (name[p] - '0');
        ++p;
      }
      layer_num = v;
    }
    char proj_rank = is_gate ? '0' : (is_up ? '1' : '2');

    packed_names.insert(name);
    packed_tensors_found++;
    for (int e = 0; e < n_experts; ++e) {
      OutTensor ot;
      ot.name = base_no_exps + "." + std::to_string(e) + ".weight";
      ot.type = pt.type;
      ot.shape = per_expert_shape;
      ot.src_shard_idx = pt.shard_idx;
      ot.src_offset = pt.offset + (uint64_t)e * per_expert_bytes;
      ot.size_bytes = per_expert_bytes;
      ot.sort_key =
          "P" + pad_sort(layer_num) + "." + pad_sort(e) + "." + proj_rank;
      out_tensors.push_back(std::move(ot));
    }
  }
  if (packed_tensors_found == 0) {
    printf("[repack] No packed expert tensors found in this file -- nothing to "
           "unpack here "
           "(a dense-only or metadata-only shard is completely normal). Every "
           "tensor in this "
           "file will pass through unchanged below.\n");
  } else {
    printf("[repack] Unpacked %d packed tensor(s) in this file into %d "
           "individual per-expert tensors "
           "(n_experts=%d)\n",
           packed_tensors_found, packed_tensors_found * n_experts, n_experts);
  }

  // Every OTHER tensor passes through unchanged.
  int passthrough_count = 0;
  for (auto &name : names) {
    if (packed_names.count(name))
      continue;
    const GGUFTensor &t = src.tensors.at(name);
    OutTensor ot;
    ot.name = name;
    ot.type = t.type;
    ot.shape = t.shape;
    ot.src_shard_idx = t.shard_idx;
    ot.src_offset = t.offset;
    ot.size_bytes = t.size_bytes;
    ot.sort_key = "Q" + name; // passthrough: stable lexicographic order, after
                              // per-expert tensors
    out_tensors.push_back(std::move(ot));
    passthrough_count++;
  }
  printf("[repack] %d tensor(s) passed through unchanged (dense weights, "
         "norms, embeddings, etc.)\n",
         passthrough_count);
  printf("[repack] Output plan: %zu total tensors (source had %zu)\n",
         out_tensors.size(), src.tensors.size());

  // Stable, deterministic overall order for layout, by sort_key (NOT name).
  // Per-expert tensors key on (layer, expert, projection-rank) so each
  // expert's gate/up/down are contiguous on disk in gate->up->down order --
  // the layout the engine's per-expert slab loader expects, and the layout
  // that keeps the three projections adjacent for streaming-read coalescing.
  // A plain name sort would order them down<gate<up and scatter each
  // expert's projections across distant file regions. Passthrough tensors
  // key on their name ("Q" prefix sorts after per-expert "P" prefixes) and
  // are independent of the per-expert grouping. Whole-list re-sort also makes
  // a re-run of this tool byte-identical regardless of unordered_map order.
  std::sort(out_tensors.begin(), out_tensors.end(),
            [](const OutTensor &a, const OutTensor &b) {
              return a.sort_key < b.sort_key;
            });

  // ---- Step 4: assign tensors to output shards by a size threshold. ----
  std::vector<std::vector<size_t>> shard_tensor_idx; // indices into out_tensors
  shard_tensor_idx.push_back({});
  uint64_t cur_shard_bytes = 0;
  for (size_t i = 0; i < out_tensors.size(); ++i) {
    uint64_t padded = align_up(out_tensors[i].size_bytes, alignment);
    if (cur_shard_bytes > 0 && cur_shard_bytes + padded > max_shard_bytes) {
      shard_tensor_idx.push_back({});
      cur_shard_bytes = 0;
    }
    int shard_idx = (int)shard_tensor_idx.size() - 1;
    out_tensors[i].out_shard_idx = shard_idx;
    shard_tensor_idx[shard_idx].push_back(i);
    cur_shard_bytes += padded;
  }
  int n_out_shards = (int)shard_tensor_idx.size();
  printf(
      "[repack] Output will be %d shard(s) (max ~%.1f GB/shard, align=%llu)\n",
      n_out_shards, max_shard_bytes / 1e9, (unsigned long long)alignment);

  // Per-shard offsets, computed in the SAME order tensors will actually
  // be written -- so the write pass below can just write sequentially
  // and padding bytes land exactly where this pass already decided.
  //
  // Alignment is applied per READ UNIT, not per tensor, so each expert's
  // gate/up/down are packed TIGHTLY (no padding between them) inside one
  // O_DIRECT-aligned expert block -- the layout that lets the
  // engine fetch a whole expert in a SINGLE pread (gate@block, up@gate+gb,
  // down@gate+gb+ub, span == gb+ub+db, no wasted padding). A read unit is:
  //   * a per-expert GATE  (sort_key "P...0") -- starts a new expert block;
  //   * a passthrough tensor ("Q...")         -- its own unit;
  //   * a per-expert UP/DOWN ("P...1"/"P...2") -- NOT a unit start: it
  //     follows its gate tightly with no alignment, so the block stays
  //     contiguous. Only the BLOCK's start needs to be O_DIRECT-aligned,
  //     not each projection inside it.
  for (auto &idxs : shard_tensor_idx) {
    uint64_t off = 0;
    for (size_t i : idxs) {
      char proj = out_tensors[i].sort_key.back();
      // Only align if it is NOT an up ('1') or down ('2') tensor
      if (proj != '1' && proj != '2') {
        off = align_up(off, alignment);
      }
      out_tensors[i].out_offset = off;
      off += out_tensors[i].size_bytes;
    }
  }

  // ---- Step 5: write each output shard. ----
  // Non-throwing overload deliberately: fs::create_directories(path)
  // (no error_code arg) throws std::filesystem::filesystem_error on
  // failure (e.g. permission denied, or a parent that isn't a
  // directory) -- uncaught anywhere in this program, that crashes the
  // whole process via std::terminate()/SIGABRT instead of a clean FATAL
  // message and a normal nonzero exit, a real failure mode confirmed by
  // actually triggering it (an unwritable output directory) on this
  // machine.
  {
    std::error_code ec;
    fs::create_directories(output_dir, ec);
    if (ec) {
      fprintf(stderr, "[repack] FATAL: cannot create output directory %s: %s\n",
              output_dir.c_str(), ec.message().c_str());
      return 1;
    }
  }
  std::vector<std::string> out_paths;
  uint64_t total_bytes_written = 0;
  std::vector<uint8_t> copy_buf(
      64ULL * 1024 * 1024); // 64MB scratch for the pread/write copy loop

  // Output filenames are tagged with the SOURCE file's own basename
  // (extension stripped) -- required for single-file-at-a-time
  // processing: separate invocations of this tool, one per source
  // shard, must never produce colliding output filenames (both would
  // otherwise independently start numbering at "00001" and overwrite
  // each other). MoEStream's own loader doesn't care about output
  // filenames beyond the .gguf extension -- it globs every .gguf file
  // in the output directory regardless of name -- so this tag only
  // needs to be unique across invocations, not meaningful in any other way.
  std::string in_tag = fs::path(input_path).stem().string();
  for (int s = 0; s < n_out_shards; ++s) {
    std::ostringstream fname;
    fname << "repacked-" << in_tag << "-part" << std::setw(5)
          << std::setfill('0') << (s + 1) << "-of-" << std::setw(5)
          << std::setfill('0') << n_out_shards << ".gguf";
    std::string out_path = (fs::path(output_dir) / fname.str()).string();
    out_paths.push_back(out_path);

    std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
      std::cerr << "[repack] FATAL: cannot create output shard " << out_path
                << '\n';
      return 1;
    }
    // Every out.write() call below (including inside write_raw_kv) was
    // previously UNCHECKED -- ofstream::write() does not throw or
    // report failure by default; if the underlying write ever fails
    // (ENOSPC is the realistic case here: this shard's own output can
    // be tens of GB, and without -d the source shard has to coexist
    // with it on disk too), failbit is set silently and every
    // subsequent write() call on this stream becomes a no-op while the
    // code keeps computing offsets and reporting success as if nothing
    // happened -- exactly the failure mode a real run hit (a narrow
    // band of tensors right at a shard boundary reading back short
    // during verify, with everything before it fine). Enabling
    // exceptions here makes ANY write failure on this stream throw
    // immediately, at the exact byte it happened, instead of silently
    // corrupting everything written after that point.
    out.exceptions(std::ofstream::failbit | std::ofstream::badbit);

    // KV list for this shard: full metadata only on shard 0, every
    // shard gets its own general.alignment + split.* triplet
    // (general.alignment is written into EVERY shard, not just the
    // metadata-carrying one, so the loader never has to depend on
    // shard load order to know it -- see gguf_loader.hpp's comment).
    std::vector<RawKV> shard_kv_out;
    if (s == 0)
      shard_kv_out = full_metadata;

    shard_kv_out.push_back(
        make_uint32_kv("general.alignment", (uint32_t)alignment));
    shard_kv_out.push_back(
        make_uint16_kv("split.count", (uint16_t)n_out_shards));      // UINT16
    shard_kv_out.push_back(make_uint16_kv("split.no", (uint16_t)s)); // UINT16
    shard_kv_out.push_back(
        make_uint32_kv("split.tensors.count", (uint32_t)out_tensors.size()));
    auto &idxs = shard_tensor_idx[s];
    uint64_t tensor_count = idxs.size();
    uint64_t kv_count = shard_kv_out.size();

    // Everything from here through out.close() can throw now that
    // exceptions() is enabled on `out` -- caught below as one unit so
    // ANY write failure (disk full being the realistic one: this
    // shard's own data alone can be tens of GB) is reported loudly and
    // precisely, with the shard/tensor that was in flight when it
    // happened, instead of silently producing a truncated file that
    // only fails much later, mysteriously, during verify.
    uint64_t written_so_far = 0;
    try {
      out.write("GGUF", 4);
      out.write(reinterpret_cast<const char *>(&src.version), 4);
      out.write(reinterpret_cast<const char *>(&tensor_count), 8);
      out.write(reinterpret_cast<const char *>(&kv_count), 8);
      for (auto &kv : shard_kv_out)
        write_raw_kv(out, kv);

      for (size_t i : idxs) {
        auto &t = out_tensors[i];
        uint64_t klen = t.name.size();
        out.write(reinterpret_cast<const char *>(&klen), 8);
        out.write(t.name.data(), (std::streamsize)klen);
        uint32_t n_dims = (uint32_t)t.shape.size();
        out.write(reinterpret_cast<const char *>(&n_dims), 4);
        out.write(reinterpret_cast<const char *>(t.shape.data()),
                  (std::streamsize)(t.shape.size() * 8));
        uint32_t type_u32 = (uint32_t)t.type;
        out.write(reinterpret_cast<const char *>(&type_u32), 4);
        out.write(reinterpret_cast<const char *>(&t.out_offset), 8);
      }

      uint64_t pos_after_table = (uint64_t)out.tellp();
      uint64_t data_start = align_up(pos_after_table, alignment);
      if (data_start > pos_after_table) {
        std::vector<char> pad(data_start - pos_after_table, 0);
        out.write(pad.data(), (std::streamsize)pad.size());
      }

      // Write tensor data, sequentially, padding between tensors so
      // each one lands exactly at data_start + its planned
      // out_offset (the SAME arithmetic the layout pass above
      // already used).
      for (size_t i : idxs) {
        auto &t = out_tensors[i];
        if (t.out_offset > written_so_far) {
          std::vector<char> pad(t.out_offset - written_so_far, 0);
          out.write(pad.data(), (std::streamsize)pad.size());
        }
        int src_fd = src.shard_fds[t.src_shard_idx];
        uint64_t remaining = t.size_bytes;
        uint64_t src_off = t.src_offset;
        while (remaining > 0) {
          size_t chunk = (size_t)std::min<uint64_t>(remaining, copy_buf.size());
          ssize_t r = pread(src_fd, copy_buf.data(), chunk, (off_t)src_off);
          if (r <= 0 || (size_t)r != chunk) {
            fprintf(stderr,
                    "[repack] FATAL: short/failed pread for %s (wanted %zu, "
                    "got %zd) at src_offset=%llu\n",
                    t.name.c_str(), chunk, r, (unsigned long long)src_off);
            return 1;
          }
          out.write(reinterpret_cast<const char *>(copy_buf.data()),
                    (std::streamsize)chunk);
          remaining -= chunk;
          src_off += chunk;
        }
        written_so_far = t.out_offset + t.size_bytes;
        total_bytes_written += t.size_bytes;
      }
      out.close();
    } catch (const std::ios_base::failure &) {
      int err = errno;
      fprintf(
          stderr,
          "[repack] FATAL: write to output shard %s failed after %.2f GB "
          "of this shard's data (errno=%d: %s). This is very likely "
          "ENOSPC (out of disk space) -- source + output need to coexist on "
          "disk during this shard's conversion (peak: roughly this one "
          "source file's size plus its own repacked output; use -d to free "
          "each source shard as soon as ITS OWN verification passes, and "
          "make sure the FIRST shard's conversion alone has enough headroom "
          "before it starts). This output file is INCOMPLETE -- do not use it; "
          "free up space and re-run.\n",
          out_path.c_str(), (double)written_so_far / 1e9, err, strerror(err));
      return 1;
    }
    std::cout << "[repack] Wrote shard " << (s + 1) << "/" << n_out_shards
              << ": " << out_path << " (" << tensor_count << " tensors, "
              << std::fixed << std::setprecision(2)
              << (static_cast<double>(written_so_far) / 1e9) << " GB)\n";
  }
  std::cout << "[repack] Done writing. Total tensor bytes copied: "
            << std::fixed << std::setprecision(2) << (total_bytes_written / 1e9)
            << " GB across " << n_out_shards << " shard(s)\n";

  // ---- Step 6: mandatory verification pass. ----
  // This tool cannot be tested against a real model on this machine --
  // this pass is the real correctness check, not optional.
  if (verify_samples <= 0) {
    printf("[repack] --verify-samples 0: SKIPPING verification. Do not trust "
           "this output without "
           "running verification separately before using it for real.\n");
    return 0;
  }
  printf("[repack] Verifying: reloading THIS invocation's own output file(s) "
         "with the real loader "
         "and byte-comparing %d sample tensor(s)...\n",
         verify_samples);
  // Deliberately NOT out_gguf.load_header(output_dir) -- in the one-
  // file-at-a-time workflow, output_dir accumulates files from EVERY
  // invocation (this run's, plus any earlier ones for other source
  // shards), and load_header globs every .gguf file in a directory
  // together as one combined model. Verifying against that would
  // compare THIS run's tensor count against the union of every
  // invocation's output ever written there -- a real bug caught by
  // actually running this tool twice in a row against two source
  // shards, not by static review (see CLAUDE.md). Instead, load ONLY
  // the file(s) this invocation just wrote (out_paths), each as its own
  // independent single-file GGUFFile, and merge their tensor tables for
  // verification purposes only.
  std::vector<std::unique_ptr<GGUFFile>> out_files;
  std::unordered_map<std::string, std::pair<GGUFFile *, GGUFTensor>> out_lookup;
  for (auto &path : out_paths) {
    auto gf = std::make_unique<GGUFFile>();
    if (!gf->load_header(path)) {
      fprintf(stderr,
              "[repack] VERIFY FAILED: the real loader could not open this "
              "invocation's "
              "own output file %s\n",
              path.c_str());
      return 1;
    }
    for (auto &[name, t] : gf->tensors)
      out_lookup[name] = {gf.get(), t};
    out_files.push_back(std::move(gf));
  }
  uint64_t expected_count = out_tensors.size();
  if (out_lookup.size() != expected_count) {
    fprintf(stderr,
            "[repack] VERIFY FAILED: this invocation's output has %zu tensors, "
            "expected %llu\n",
            out_lookup.size(), (unsigned long long)expected_count);
    return 1;
  }
  // architecture_config is deliberately NOT checked here -- it's only
  // ever populated by GGUFGraphInitializer::build_model_graph, which
  // this standalone tool never calls (see Step 1's own comment); it
  // would read as all-zero on BOTH source and reloaded output, making
  // any comparison meaningless rather than a real check. Instead,
  // re-derive the expanded-tensor-name set from the reload directly:
  // for a sample of (layer, expert) pairs this run's own plan produced,
  // confirm the exact expected name exists in the reloaded tensor
  // table -- a direct, meaningful check that the unpacking naming
  // scheme round-trips correctly through the real loader.
  printf("[repack] Tensor count matches: %llu tensors\n",
         (unsigned long long)expected_count);

  // A metadata-only shard (0 tensors) is legitimate and already confirmed
  // above via the tensor-count check -- there's nothing left to sample.
  // out_tensors.size() - 1 would underflow (size_t) to SIZE_MAX in this
  // case, handing uniform_int_distribution a huge range and reading far
  // out of bounds on the empty vector below -- a real SIGSEGV hit against
  // a real production shard (GLM-5.2's own shard 1 of 6 carries all KV
  // metadata and zero tensors), not caught by any synthetic fixture since
  // every one of those had tensors in every shard. Skip sampling (not an
  // early return) so the shared VERIFY PASSED / -d deletion logic below
  // still runs exactly once, the same way it does for every other shard.
  std::mt19937 rng(12345); // fixed seed: reproducible verification runs
  int mismatches = 0;
  int checked = 0;
  int actual_samples = out_tensors.empty() ? 0 : verify_samples;
  if (out_tensors.empty()) {
    printf("[repack] This shard has 0 tensors -- nothing to sample, "
           "verification is just the "
           "tensor-count check above.\n");
  }
  std::uniform_int_distribution<size_t> pick_tensor(
      0, out_tensors.empty() ? 0 : out_tensors.size() - 1);
  for (int i = 0; i < actual_samples; ++i) {
    const OutTensor &plan = out_tensors[pick_tensor(rng)];
    auto reloaded_it = out_lookup.find(plan.name);
    if (reloaded_it == out_lookup.end()) {
      fprintf(stderr, "[repack] VERIFY FAILED: %s missing after reload\n",
              plan.name.c_str());
      mismatches++;
      continue;
    }
    GGUFFile *owner = reloaded_it->second.first;
    const GGUFTensor &reloaded = reloaded_it->second.second;
    if (reloaded.size_bytes != plan.size_bytes) {
      fprintf(stderr,
              "[repack] VERIFY FAILED: %s size_bytes mismatch (planned %llu, "
              "reloaded %llu)\n",
              plan.name.c_str(), (unsigned long long)plan.size_bytes,
              (unsigned long long)reloaded.size_bytes);
      mismatches++;
      continue;
    }
    std::vector<uint8_t> src_buf(plan.size_bytes), out_buf(plan.size_bytes);
    ssize_t r1 = pread(src.shard_fds[plan.src_shard_idx], src_buf.data(),
                       plan.size_bytes, (off_t)plan.src_offset);
    ssize_t r2 = pread(owner->fd_for_tensor(reloaded), out_buf.data(),
                       plan.size_bytes, (off_t)reloaded.offset);
    if (r1 != (ssize_t)plan.size_bytes || r2 != (ssize_t)plan.size_bytes) {
      fprintf(stderr, "[repack] VERIFY FAILED: pread error while checking %s\n",
              plan.name.c_str());
      mismatches++;
      continue;
    }
    if (std::memcmp(src_buf.data(), out_buf.data(), plan.size_bytes) != 0) {
      fprintf(stderr,
              "[repack] VERIFY FAILED: %s bytes differ between source and "
              "repacked output\n",
              plan.name.c_str());
      mismatches++;
      continue;
    }
    checked++;
  }
  printf(
      "[repack] Verify: %d/%d sample tensors byte-identical, %d mismatch(es)\n",
      checked, actual_samples, mismatches);
  if (mismatches > 0) {
    fprintf(stderr, "[repack] VERIFY FAILED overall -- do NOT trust this "
                    "output. See mismatches above.\n");
    return 1;
  }
  printf("[repack] VERIFY PASSED. Repacked output at %s looks byte-correct "
         "against the source.\n",
         output_dir.c_str());

  if (delete_source) {
    // Deliberately done HERE -- in the same process, immediately after
    // the SAME check that just confirmed byte-correctness -- rather
    // than left to a wrapper script's own separate exit-code check
    // (tools/repack_all.sh forwards this flag instead of doing its own
    // delete for exactly this reason: one source of truth for "did
    // this actually verify", no risk of a wrapper's shell-level logic
    // drifting from what this tool itself considers success). Safe to
    // delete while `src`'s file descriptors are still open (its
    // destructor hasn't run yet) -- POSIX unlink only removes the
    // directory entry; the underlying data isn't reclaimed until the
    // last open fd closes, so this can never race the verify reads
    // that already completed above.
    std::error_code ec;
    fs::remove(input_path, ec);
    if (ec) {
      fprintf(
          stderr,
          "[repack] WARNING: --delete-source failed to remove %s: %s "
          "(output is still verified-correct; delete it manually if desired)\n",
          input_path.c_str(), ec.message().c_str());
    } else {
      printf("[repack] --delete-source: removed %s\n", input_path.c_str());
    }
  }
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr,
            "Usage: %s <source_shard_file_or_dir> <output_dir> [--align N] "
            "[--max-shard-gb N] [--verify-samples N] [-d|--delete-source]\n"
            "\n"
            "Processes GGUF shard(s) one at a time -- pass a SINGLE shard file "
            "to process\n"
            "just that one, or a DIRECTORY to walk every .gguf file in it "
            "(sorted) and\n"
            "process them one after another, identical to running this tool "
            "once per file\n"
            "by hand. Either way, only one source shard + its output need "
            "exist on disk at\n"
            "once, so peak disk usage stays roughly \"one shard + its "
            "output\", not the whole\n"
            "model twice -- combine with -d/--delete-source to free each shard "
            "automatically\n"
            "as soon as its own verification passes, before moving to the next "
            "one.\n"
            "\n"
            "  --align N              byte alignment for every tensor's start "
            "offset (default 4096)\n"
            "  --max-shard-gb N       approx max size per OUTPUT shard from "
            "EACH source file, in GB (default 40)\n"
            "  --verify-samples N     random tensors to byte-compare source vs "
            "output after writing (default 200, 0 = skip)\n"
            "  -d, --delete-source    DESTRUCTIVE: delete each source shard "
            "right after ITS OWN verification\n"
            "                         passes. Refuses to combine with "
            "--verify-samples 0 (would delete without\n"
            "                         ever checking anything). Off by default. "
            "No undo.\n",
            argv[0]);
    return 2;
  }
  std::string input_path = argv[1];
  std::string output_dir = argv[2];
  uint64_t alignment = 4096;
  uint64_t max_shard_bytes = 40ULL * 1000 * 1000 * 1000;
  int verify_samples = 200;
  bool delete_source = false;
  for (int i = 3; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--align" && i + 1 < argc)
      alignment = std::stoull(argv[++i]);
    else if (a == "--max-shard-gb" && i + 1 < argc)
      max_shard_bytes = (uint64_t)(std::stod(argv[++i]) * 1e9);
    else if (a == "--verify-samples" && i + 1 < argc)
      verify_samples = std::stoi(argv[++i]);
    else if (a == "-d" || a == "--delete-source")
      delete_source = true;
    else {
      fprintf(stderr, "[repack] FATAL: unknown argument '%s'\n", a.c_str());
      return 2;
    }
  }
  if (delete_source && verify_samples <= 0) {
    fprintf(
        stderr,
        "[repack] FATAL: --delete-source refuses to combine with "
        "--verify-samples 0 -- "
        "that would delete the source without ever having verified anything. "
        "Either drop --delete-source or use a nonzero --verify-samples.\n");
    return 2;
  }
  if ((alignment & (alignment - 1)) != 0 || alignment == 0) {
    fprintf(stderr,
            "[repack] FATAL: --align must be a power of two, got %llu\n",
            (unsigned long long)alignment);
    return 2;
  }

  std::error_code ec;
  bool is_dir = fs::is_directory(input_path, ec);
  if (ec) {
    fprintf(stderr, "[repack] FATAL: cannot stat '%s': %s\n",
            input_path.c_str(), ec.message().c_str());
    return 1;
  }
  if (!is_dir) {
    return process_one_shard(input_path, output_dir, alignment, max_shard_bytes,
                             verify_samples, delete_source);
  }

  // Directory mode: walk every .gguf file directly in input_path (not
  // recursive -- a model directory is expected flat), sorted -- matches
  // GGUF's own "-00001-of-NNNNN.gguf" shard naming convention, so
  // progress output reads in a predictable order. Not required for
  // correctness (each shard is fully independent, see process_one_shard's
  // own header comment) -- just easier to follow, and easier to resume
  // after an interruption: a shard already deleted via -d simply won't
  // be in the next run's listing, so re-running this same command is
  // naturally resumable with no extra state-tracking needed.
  std::vector<std::string> shard_files;
  for (auto &entry : fs::directory_iterator(input_path)) {
    if (entry.is_regular_file() && entry.path().extension() == ".gguf")
      shard_files.push_back(entry.path().string());
  }
  std::sort(shard_files.begin(), shard_files.end());
  if (shard_files.empty()) {
    fprintf(stderr, "[repack] FATAL: no .gguf files found in directory %s\n",
            input_path.c_str());
    return 1;
  }
  printf("[repack] Directory mode: found %zu shard(s) in %s\n",
         shard_files.size(), input_path.c_str());
  for (auto &p : shard_files)
    printf("  %s\n", p.c_str());
  printf("\n");

  for (size_t i = 0; i < shard_files.size(); ++i) {
    printf("[repack] ===== Shard %zu/%zu: %s =====\n", i + 1,
           shard_files.size(), shard_files[i].c_str());
    int rc = process_one_shard(shard_files[i], output_dir, alignment,
                               max_shard_bytes, verify_samples, delete_source);
    if (rc != 0) {
      fprintf(stderr,
              "[repack] Directory mode: shard %zu/%zu FAILED (%s) -- stopping "
              "here, "
              "remaining shard(s) not processed. Fix the issue and re-run "
              "against the same "
              "directory -- shards already verified above are unaffected "
              "(re-running safely "
              "overwrites only the failed shard's own output).\n",
              i + 1, shard_files.size(), shard_files[i].c_str());
      return rc;
    }
    printf("\n");
  }
  printf("[repack] Directory mode: all %zu shard(s) processed and verified "
         "successfully.\n",
         shard_files.size());
  return 0;
}

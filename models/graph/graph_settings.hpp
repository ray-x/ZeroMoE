#pragma once

// Typed graph settings with one resolved value per knob and its source.
// Design: docs/design/settings-resolver.md. Implemented layers: defaults,
// family overrides, root TOML keys and compatibility environment aliases.
//
// Pure: no logger, no CUDA, no getenv. The environment is an injected view so
// tests never call setenv; graph_settings_global.hpp binds the real one.
//
// Defaults preserve existing behaviour: every default and override equals the
// value the code used before, and every legacy env var keeps its old parse
// semantics (strict strtol, atoi, "on unless 0", ...).

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <charconv>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <string>
#include <vector>

namespace zeromoe::graphsettings {

// GlmGraph is the shared GLM graph implementation (GLM-5.x, GLM-DSA and Ling run through it); Ling also has
// its own entry for the settings its model class reads.
enum class Family : uint8_t { Other, MiMo2, Ling, GlmGraph, Qwen };
enum class Source : uint8_t { Default, Family, Toml, Env };

inline const char *source_name(Source s) {
  switch (s) {
  case Source::Default: return "default";
  case Source::Family: return "family";
  case Source::Toml: return "toml";
  case Source::Env: return "env";
  }
  return "?";
}

template <class T> struct Resolved {
  T value{};
  Source src = Source::Default;
  std::string origin; // canonical TOML key or legacy environment name
  std::string reason; // family overrides only
};

// Tunables: may change performance or numerics. Only these feed the drift hash.
struct GraphSettings {
  Resolved<long> device_expert_layers;
  Resolved<long> device_expert_trailing;
  Resolved<long> phase_cache_mib;          // optional decode-only extension, 0 = disabled
  Resolved<long> phase_cache_reserve_mib;  // free VRAM kept for decode workspace
  Resolved<long> cpu_threads_decode;
  Resolved<long> prefill_chunk;            // 0 = family/engine decides
  Resolved<long> prefill_min_batch;
  Resolved<long> prefill_op_offload;       // -1 auto, 0 off, 1 on
  Resolved<long> prefill_offload_min_tokens;
  Resolved<long> offload_pinned_pct;
  Resolved<long> offload_pin;              // bool
  Resolved<long> offload_async;            // bool
  Resolved<long> offload_first_layer;      // -1 = no offload tier
  Resolved<long> gpu_assist_slots;
  Resolved<long> use_gpu1;                 // bool
  // GLM graph offload tier (GLM-5.x, GLM-DSA, Ling); MiMo's own knobs are the fields above.
  Resolved<long> offload_thp;              // bool: huge pages for the arena
  Resolved<long> offload_split_reads;      // bool
  Resolved<long> offload_byte_cap;         // bool: pool capacity counted in bytes
  Resolved<long> offload_pin_dma;          // -1 = pin when GPU assist is on, 0 off, 1 on
  Resolved<long> offload_seed_gpu1_hot;    // bool
  // GLM graph lane / staging switches.
  Resolved<long> early_ready;              // -1 = on when the layer has overlap work, 0 off, 1 forced
  Resolved<long> split_load, lane_with_gpu1, pilot_copy;  // bool, on by default
  Resolved<long> gpu_assist_any_type, dynamic_cache;      // bool, off by default
  Resolved<long> gpu_assist2_slots;
  Resolved<double> offload_pinned_fraction;
  Resolved<double> offload_probation_fraction;
  Resolved<std::string> offload_policy;    // "lru" or "seg"
  Resolved<std::string> offload_mirror, offload_store, offload_model;
  Resolved<double> offload_ddr_gib;        // < 0 = auto
  Resolved<std::string> offload_usage_file;
};

// Diagnostics: never hashed, never drift-checked.
struct DiagnosticSettings {
  Resolved<long> verify_alias;             // bool
  Resolved<long> node_trace;
  Resolved<std::string> usage_record_file;  // survey mode: write per-expert route counts here at shutdown
  Resolved<long> preflight_strict;         // bool: a preflight finding aborts instead of warning
};

struct ResolvedSettings {
  Family family = Family::Other;
  GraphSettings tun;
  DiagnosticSettings diag;
  std::vector<std::string> warnings;
};

using EnvView = std::function<const char *(const char *)>;
using ConfigValues = std::map<std::string, std::string>;

enum class Parse { IntStrict, IntAtoi, FlagOn, FlagUnlessOff, TriState, EqualsOne, RealOrAuto, Text };

struct Alias {
  const char *env;
  Family only; // scope, ignored when `all`
  bool all;
};
inline Alias any_family(const char *env) { return {env, Family::Other, true}; }
inline Alias only_family(Family f, const char *env) { return {env, f, false}; }

template <class T> struct Field {
  const char *name;
  Resolved<T> &(*ref)(ResolvedSettings &);
  T dflt;
  Parse parse;
  std::vector<Alias> aliases; // priority order; the first present value wins
  bool diagnostic;
};

#define ZEROMOE_SETTINGS_REF(member_path) \
  [](ResolvedSettings &r) -> decltype((r.member_path)) { return r.member_path; }

inline const std::vector<Field<long>> &long_fields() {
  using F = Family;
  static const std::vector<Field<long>> t = {
      {"device_expert_layers", ZEROMOE_SETTINGS_REF(tun.device_expert_layers), 0, Parse::IntStrict,
       {only_family(F::Qwen, "ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_LAYERS")}, false},
      {"device_expert_trailing", ZEROMOE_SETTINGS_REF(tun.device_expert_trailing), 0, Parse::FlagOn,
       {only_family(F::Qwen, "ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_TRAILING")}, false},
      {"phase_cache_mib", ZEROMOE_SETTINGS_REF(tun.phase_cache_mib), 0, Parse::IntStrict,
       {any_family("ZEROMOE_PHASE_CACHE_MIB")}, false},
      {"phase_cache_reserve_mib", ZEROMOE_SETTINGS_REF(tun.phase_cache_reserve_mib), 512, Parse::IntStrict,
       {any_family("ZEROMOE_PHASE_CACHE_RESERVE_MIB")}, false},
      {"cpu_threads_decode", ZEROMOE_SETTINGS_REF(tun.cpu_threads_decode), 10, Parse::IntStrict,
       {only_family(F::MiMo2, "ZEROMOE_MIMO2_CPU_THREADS"),
        only_family(F::Qwen, "ZEROMOE_QWEN_CPU_THREADS")}, false},
      {"prefill_chunk", ZEROMOE_SETTINGS_REF(tun.prefill_chunk), 0, Parse::IntAtoi,
       {only_family(F::MiMo2, "ZEROMOE_MIMO2_PREFILL_CHUNK"),
        only_family(F::Ling, "ZEROMOE_LING_PREFILL_CHUNK")}, false},
      {"prefill_min_batch", ZEROMOE_SETTINGS_REF(tun.prefill_min_batch), 64, Parse::IntStrict, {}, false},
      {"prefill_op_offload", ZEROMOE_SETTINGS_REF(tun.prefill_op_offload), -1, Parse::TriState,
       {only_family(F::MiMo2, "ZEROMOE_MIMO2_PREFILL_OP_OFFLOAD"),
        only_family(F::GlmGraph, "ZEROMOE_LING_PREFILL_OP_OFFLOAD"), only_family(F::GlmGraph, "ZEROMOE_GLM5NEXT_PREFILL_OP_OFFLOAD")}, false},
      {"prefill_offload_min_tokens", ZEROMOE_SETTINGS_REF(tun.prefill_offload_min_tokens), 256, Parse::IntStrict,
       {only_family(F::MiMo2, "ZEROMOE_MIMO2_PREFILL_OFFLOAD_MIN")}, false},
      {"offload_pinned_pct", ZEROMOE_SETTINGS_REF(tun.offload_pinned_pct), 50, Parse::IntStrict,
       {only_family(F::MiMo2, "ZEROMOE_MIMO2_OFFLOAD_PINNED_PCT")}, false},
      {"offload_pin", ZEROMOE_SETTINGS_REF(tun.offload_pin), 1, Parse::FlagUnlessOff,
       {only_family(F::MiMo2, "ZEROMOE_MIMO2_OFFLOAD_PIN")}, false},
      {"offload_async", ZEROMOE_SETTINGS_REF(tun.offload_async), 0, Parse::FlagOn,
       {only_family(F::GlmGraph, "ZEROMOE_LING_OFFLOAD_ASYNC"), only_family(F::GlmGraph, "ZEROMOE_GLM5NEXT_OFFLOAD_ASYNC")}, false},
      {"offload_first_layer", ZEROMOE_SETTINGS_REF(tun.offload_first_layer), -1, Parse::IntAtoi,
       {only_family(F::GlmGraph, "ZEROMOE_LING_OFFLOAD_FIRST_LAYER"), only_family(F::GlmGraph, "ZEROMOE_GLM5NEXT_OFFLOAD_FIRST_LAYER")}, false},
      {"gpu_assist_slots", ZEROMOE_SETTINGS_REF(tun.gpu_assist_slots), 0, Parse::IntStrict,
       {only_family(F::MiMo2, "ZEROMOE_MIMO2_GPU_ASSIST"), only_family(F::GlmGraph, "ZEROMOE_GLM5NEXT_GPU_ASSIST")}, false},
      {"use_gpu1", ZEROMOE_SETTINGS_REF(tun.use_gpu1), 1, Parse::FlagUnlessOff,
       {only_family(F::MiMo2, "ZEROMOE_MIMO2_GPU1")}, false},
      {"offload_thp", ZEROMOE_SETTINGS_REF(tun.offload_thp), 0, Parse::EqualsOne,
       {only_family(F::GlmGraph, "ZEROMOE_LING_OFFLOAD_THP"), only_family(F::GlmGraph, "ZEROMOE_GLM5NEXT_OFFLOAD_THP")}, false},
      {"offload_split_reads", ZEROMOE_SETTINGS_REF(tun.offload_split_reads), 0, Parse::EqualsOne,
       {only_family(F::GlmGraph, "ZEROMOE_LING_OFFLOAD_SPLIT"), only_family(F::GlmGraph, "ZEROMOE_GLM5NEXT_OFFLOAD_SPLIT")}, false},
      {"offload_byte_cap", ZEROMOE_SETTINGS_REF(tun.offload_byte_cap), 0, Parse::EqualsOne,
       {only_family(F::GlmGraph, "ZEROMOE_LING_OFFLOAD_BYTE_CAP"), only_family(F::GlmGraph, "ZEROMOE_GLM5NEXT_OFFLOAD_BYTE_CAP")}, false},
      {"offload_pin_dma", ZEROMOE_SETTINGS_REF(tun.offload_pin_dma), -1, Parse::EqualsOne,
       {only_family(F::GlmGraph, "ZEROMOE_LING_OFFLOAD_PIN"), only_family(F::GlmGraph, "ZEROMOE_GLM5NEXT_OFFLOAD_PIN")}, false},
      {"offload_seed_gpu1_hot", ZEROMOE_SETTINGS_REF(tun.offload_seed_gpu1_hot), 1, Parse::EqualsOne,
       {only_family(F::GlmGraph, "ZEROMOE_LING_OFFLOAD_SEED_GPU1_HOT"), only_family(F::GlmGraph, "ZEROMOE_GLM5NEXT_OFFLOAD_SEED_GPU1_HOT")}, false},
      {"early_ready", ZEROMOE_SETTINGS_REF(tun.early_ready), -1, Parse::TriState,
       {only_family(F::GlmGraph, "ZEROMOE_GLM5NEXT_EARLY_READY")}, false},
      {"split_load", ZEROMOE_SETTINGS_REF(tun.split_load), 1, Parse::FlagUnlessOff,
       {only_family(F::GlmGraph, "ZEROMOE_GLM5NEXT_SPLIT_LOAD")}, false},
      {"lane_with_gpu1", ZEROMOE_SETTINGS_REF(tun.lane_with_gpu1), 1, Parse::FlagUnlessOff,
       {only_family(F::GlmGraph, "ZEROMOE_GLM5NEXT_LANE_WITH_GPU1")}, false},
      {"pilot_copy", ZEROMOE_SETTINGS_REF(tun.pilot_copy), 1, Parse::FlagUnlessOff,
       {only_family(F::GlmGraph, "ZEROMOE_GLM5NEXT_PILOT_COPY")}, false},
      {"gpu_assist_any_type", ZEROMOE_SETTINGS_REF(tun.gpu_assist_any_type), 0, Parse::FlagOn,
       {only_family(F::GlmGraph, "ZEROMOE_GLM5NEXT_GPU_ASSIST_ANY_TYPE")}, false},
      {"dynamic_cache", ZEROMOE_SETTINGS_REF(tun.dynamic_cache), 0, Parse::FlagOn,
       {only_family(F::GlmGraph, "ZEROMOE_GLM5NEXT_DYNAMIC_CACHE")}, false},
      {"gpu_assist2_slots", ZEROMOE_SETTINGS_REF(tun.gpu_assist2_slots), 1, Parse::IntAtoi,
       {only_family(F::GlmGraph, "ZEROMOE_GLM5NEXT_ASSIST2_SLOTS"), only_family(F::MiMo2, "ZEROMOE_MIMO2_GPU_ASSIST2")}, false},
      {"verify_alias", ZEROMOE_SETTINGS_REF(diag.verify_alias), 1, Parse::FlagUnlessOff,
       {only_family(F::MiMo2, "ZEROMOE_MIMO2_VERIFY_ALIAS")}, true},
      {"node_trace", ZEROMOE_SETTINGS_REF(diag.node_trace), 0, Parse::IntStrict,
       {only_family(F::MiMo2, "ZEROMOE_MIMO2_NODE_TRACE")}, true},
      {"preflight_strict", ZEROMOE_SETTINGS_REF(diag.preflight_strict), 0, Parse::FlagOn,
       {any_family("ZEROMOE_PREFLIGHT_STRICT")}, true},
  };
  return t;
}

inline const std::vector<Field<double>> &real_fields() {
  static const std::vector<Field<double>> t = {
      {"offload_ddr_gib", ZEROMOE_SETTINGS_REF(tun.offload_ddr_gib), -1.0, Parse::RealOrAuto,
       {only_family(Family::MiMo2, "ZEROMOE_MIMO2_OFFLOAD_DDR_GIB"),
        only_family(Family::GlmGraph, "ZEROMOE_LING_OFFLOAD_DDR_GIB"), only_family(Family::GlmGraph, "ZEROMOE_GLM5NEXT_OFFLOAD_DDR_GIB")}, false},
      {"offload_pinned_fraction", ZEROMOE_SETTINGS_REF(tun.offload_pinned_fraction), 0.5, Parse::RealOrAuto,
       {only_family(Family::GlmGraph, "ZEROMOE_LING_OFFLOAD_PINNED"), only_family(Family::GlmGraph, "ZEROMOE_GLM5NEXT_OFFLOAD_PINNED")}, false},
      {"offload_probation_fraction", ZEROMOE_SETTINGS_REF(tun.offload_probation_fraction), 0.25, Parse::RealOrAuto,
       {only_family(Family::GlmGraph, "ZEROMOE_LING_OFFLOAD_PROBATION"), only_family(Family::GlmGraph, "ZEROMOE_GLM5NEXT_OFFLOAD_PROBATION")}, false},
  };
  return t;
}

inline const std::vector<Field<std::string>> &text_fields() {
  static const std::vector<Field<std::string>> t = {
      {"offload_usage_file", ZEROMOE_SETTINGS_REF(tun.offload_usage_file), std::string(), Parse::Text,
       {only_family(Family::MiMo2, "ZEROMOE_MIMO2_OFFLOAD_USAGE"),
        only_family(Family::GlmGraph, "ZEROMOE_LING_OFFLOAD_USAGE"), only_family(Family::GlmGraph, "ZEROMOE_GLM5NEXT_OFFLOAD_USAGE")}, false},
      {"usage_record_file", ZEROMOE_SETTINGS_REF(diag.usage_record_file), std::string(), Parse::Text,
       {only_family(Family::MiMo2, "ZEROMOE_MIMO2_USAGE_RECORD"),
        only_family(Family::GlmGraph, "ZEROMOE_LING_USAGE_RECORD"), only_family(Family::GlmGraph, "ZEROMOE_GLM5NEXT_USAGE_RECORD")}, true},
      {"offload_policy", ZEROMOE_SETTINGS_REF(tun.offload_policy), std::string("lru"), Parse::Text,
       {only_family(Family::GlmGraph, "ZEROMOE_LING_OFFLOAD_POLICY"), only_family(Family::GlmGraph, "ZEROMOE_GLM5NEXT_OFFLOAD_POLICY")}, false},
      {"offload_mirror", ZEROMOE_SETTINGS_REF(tun.offload_mirror), std::string(), Parse::Text,
       {only_family(Family::GlmGraph, "ZEROMOE_LING_OFFLOAD_MIRROR"), only_family(Family::GlmGraph, "ZEROMOE_GLM5NEXT_OFFLOAD_MIRROR")}, false},
      {"offload_store", ZEROMOE_SETTINGS_REF(tun.offload_store), std::string(), Parse::Text,
       {only_family(Family::GlmGraph, "ZEROMOE_LING_OFFLOAD_STORE"), only_family(Family::GlmGraph, "ZEROMOE_GLM5NEXT_OFFLOAD_STORE")}, false},
      {"offload_model", ZEROMOE_SETTINGS_REF(tun.offload_model), std::string(), Parse::Text,
       {only_family(Family::GlmGraph, "ZEROMOE_LING_OFFLOAD_MODEL"), only_family(Family::GlmGraph, "ZEROMOE_GLM5NEXT_OFFLOAD_MODEL")}, false},
  };
  return t;
}

template<class T> inline bool applies(const Field<T> &f, Family fam) {
  if (std::strcmp(f.name, "prefill_min_batch") == 0)
    return fam == Family::MiMo2 || fam == Family::Ling;
  if (f.aliases.empty()) return true;
  return std::any_of(f.aliases.begin(), f.aliases.end(),
                     [fam](const Alias &a) { return a.all || a.only == fam; });
}

inline bool boolean(Parse p) {
  return p == Parse::FlagOn || p == Parse::FlagUnlessOff || p == Parse::EqualsOne;
}
inline bool auto_flag(const std::string &name, Parse p) {
  return p == Parse::TriState || name == "offload_pin_dma";
}

inline bool valid_integer(const std::string &name, long n) {
  if (name == "device_expert_layers") return n >= 0 && n <= 48;
  if (name == "offload_first_layer") return n >= -1 && n <= 1000000;
  if (name == "prefill_min_batch") return n > 0 && n <= 1000000;
  if (name == "offload_pinned_pct") return n >= 0 && n <= 100;
  if (name == "gpu_assist2_slots") return n >= 0 && n <= 8;
  return n >= 0 && n <= 1048576;
}

template<class Fn> inline void each_field(Fn fn) {
  for (const auto &f : long_fields()) fn(f);
  for (const auto &f : real_fields()) fn(f);
  for (const auto &f : text_fields()) fn(f);
}

inline const char *family_name(Family fam) {
  switch (fam) {
  case Family::Other: return "shared";
  case Family::MiMo2: return "mimo2";
  case Family::Ling: return "ling";
  case Family::GlmGraph: return "glm-graph";
  case Family::Qwen: return "qwen4exp";
  }
  throw std::invalid_argument("unknown graph family");
}

inline const char *meaning(const std::string &name) {
  static const std::map<std::string, const char *> descriptions{
      {"device_expert_layers", "Qwen whole expert layers resident on GPU (0..48); separate from the partial hotset budget."},
      {"device_expert_trailing", "Place whole GPU-resident expert layers at the end rather than start of the model."},
      {"phase_cache_mib", "Optional decode-hot VRAM extension, in MiB; 0 disables it. Additional to the main expert cache."},
      {"phase_cache_reserve_mib", "Free VRAM retained for decode workspace when sizing the phase cache, in MiB."},
      {"cpu_threads_decode", "CPU expert worker count; Qwen uses this as a cap (0 = available threads), MiMo as its team size."},
      {"prefill_chunk", "Prompt tokens per graph chunk (MiMo maximum: 512); 0 lets the family/engine decide. Not a context limit."},
      {"prefill_min_batch", "Smallest multi-token prefill graph batch; smaller batches use single-token steps."},
      {"prefill_op_offload", "Copy host experts to GPU for prefill: true, false or \"auto\". Requires family support and workspace."},
      {"prefill_offload_min_tokens", "Minimum prompt batch in tokens for MiMo GPU expert op-offload."},
      {"offload_pinned_pct", "MiMo host arena percentage backed by pinned RAM (0..100); raises nonpageable memory usage."},
      {"offload_pin", "Allow MiMo expert-arena pinning for DMA; false can reduce host-to-GPU bandwidth."},
      {"offload_async", "Overlap GLM/Ling offload reads rather than performing them synchronously."},
      {"offload_first_layer", "First GLM/Ling layer in the host offload tier; -1 leaves the tier disabled."},
      {"gpu_assist_slots", "Concurrent staging slots for host experts computed on GPU; consumes extra VRAM."},
      {"gpu_assist2_slots", "Second assist-ring staging slots (0..8); used by GLM and MiMo."},
      {"use_gpu1", "Allow MiMo's second-GPU path when the configured topology provides one."},
      {"offload_thp", "Request transparent huge pages for the GLM/Ling host expert arena."},
      {"offload_split_reads", "Use the GLM/Ling split expert-read path."},
      {"offload_byte_cap", "Account GLM/Ling offload-pool capacity in bytes rather than uniform slots."},
      {"offload_pin_dma", "Pin GLM/Ling host DMA storage: true, false or \"auto\" (when GPU assist is active)."},
      {"offload_seed_gpu1_hot", "Seed the GLM/Ling second-GPU hot set during startup."},
      {"early_ready", "GLM early-ready overlap lane: true, false or \"auto\" when the layer has overlap work."},
      {"split_load", "Allow GLM split expert-load scheduling; false changes the overlap strategy."},
      {"lane_with_gpu1", "Allow the GLM overlap lane alongside second-GPU expert work."},
      {"pilot_copy", "Allow GLM pilot-driven expert-copy staging."},
      {"gpu_assist_any_type", "Experimental GLM assist for additional quantization types; leave at the validated family value."},
      {"dynamic_cache", "Experimental GLM dynamic expert cache rather than the fixed validated placement."},
      {"offload_pinned_fraction", "Pinned fraction of the GLM/Ling host offload pool (0..1), not a GiB budget."},
      {"offload_probation_fraction", "GLM/Ling segmented-cache probation fraction (0..1); applies to the segmented policy."},
      {"offload_policy", "GLM/Ling host cache policy: \"lru\" or \"seg\". Keep the measured recipe unless benchmarking."},
      {"offload_mirror", "GLM/Ling expert mirror path; not a second model family or GPU selection."},
      {"offload_store", "GLM/Ling expert offload-store path."},
      {"offload_model", "GLM/Ling model source used by expert offload."},
      {"offload_ddr_gib", "Host expert offload budget in GiB, or \"auto\". Separate from VRAM and pinned fractions."},
      {"offload_usage_file", "Historical expert-routing counts used to choose the hot set."},
      {"usage_record_file", "Diagnostic route-survey output path; empty disables recording. Excluded from default hashes."},
      {"verify_alias", "MiMo diagnostic alias checks; leave enabled unless performing a controlled experiment."},
      {"node_trace", "MiMo diagnostic node-trace level; 0 disables it and avoids instrumentation overhead."},
      {"preflight_strict", "Fail on graph health/performance findings rather than only warning. Does not weaken safety checks."},
  };
  const auto it = descriptions.find(name);
  if (it == descriptions.end()) throw std::logic_error("missing setting description: " + name);
  return it->second;
}

namespace detail {
inline void canonical(const Field<long> &f, const std::string &v, Resolved<long> &out) {
  long n = 0;
  if (boolean(f.parse) && !auto_flag(f.name, f.parse)) {
    if (v != "true" && v != "false") throw std::invalid_argument("expected true or false");
    n = v == "true";
  } else if (auto_flag(f.name, f.parse)) {
    if (v != "auto" && v != "true" && v != "false")
      throw std::invalid_argument("expected \"auto\", true or false");
    n = v == "auto" ? -1 : v == "true";
  } else {
    auto [end, ec] = std::from_chars(v.data(), v.data() + v.size(), n);
    if (ec != std::errc{} || end != v.data() + v.size() || !valid_integer(f.name, n))
      throw std::invalid_argument("integer outside supported range");
  }
  out = {n, Source::Toml, "graph_" + std::string(f.name), {}};
}
inline void canonical(const Field<double> &f, const std::string &v, Resolved<double> &out) {
  double n = 0;
  if (std::strcmp(f.name, "offload_ddr_gib") == 0 && v == "auto") n = -1;
  else {
    char *end = nullptr;
    errno = 0;
    n = std::strtod(v.c_str(), &end);
    const bool fraction = std::strstr(f.name, "fraction") != nullptr;
    if (end == v.c_str() || *end || errno || !std::isfinite(n) ||
        n < 0 || n > (fraction ? 1.0 : 1048576.0))
      throw std::invalid_argument(fraction ? "expected a fraction in [0, 1]" :
                                            "expected nonnegative GiB or \"auto\"");
  }
  out = {n, Source::Toml, "graph_" + std::string(f.name), {}};
}
inline void canonical(const Field<std::string> &f, const std::string &v, Resolved<std::string> &out) {
  if (std::strcmp(f.name, "offload_policy") == 0 && v != "lru" && v != "seg")
    throw std::invalid_argument("expected \"lru\" or \"seg\"");
  out = {v, Source::Toml, "graph_" + std::string(f.name), {}};
}
}

// A family's departure from the common default. The reason is mandatory and an
// override equal to the default is rejected as redundant.
struct Override {
  Family family;
  const char *field;
  long value;
  const char *reason;
};

inline const std::vector<Override> &family_overrides() {
  static const std::vector<Override> t = {
      {Family::MiMo2, "prefill_chunk", 512,
       "chunks >= 256 tokens let op-offload beat the CPU cold-expert path (docs/models/mimo26-flash.md)"},
      {Family::MiMo2, "prefill_min_batch", 8,
       "the default (64) made every chunk after the first run as one-token steps (docs/design/model-family-framework.md s1)"},
      {Family::GlmGraph, "gpu_assist_slots", 3,
       "GPU assist of 3 slots is the measured setting for the GLM graph (docs/GLM5.3-Flash-baseline.md)"},
      {Family::Ling, "prefill_min_batch", 2,
       "value used by Ling31FlashGraphModel before the resolver existed"},
      {Family::Qwen, "cpu_threads_decode", 0,
       "0 keeps the existing Qwen available-thread count; positive values cap it"},
  };
  return t;
}

namespace detail {

inline bool present(const char *v) { return v && v[0]; }

inline void read_long(const Field<long> &f, Family fam, const EnvView &env, Resolved<long> &out) {
  bool found = false, any_on = false;
  for (const Alias &a : f.aliases) {
    if (!a.all && a.only != fam)
      continue;
    const char *v = env(a.env);
    switch (f.parse) {
    case Parse::FlagOn:
      if (present(v)) {
        if (!found) out.origin = a.env;
        found = true;
        any_on = any_on || v[0] == '1';
      }
      break;
    case Parse::FlagUnlessOff:
      if (present(v)) {
        out.value = v[0] == '0' ? 0 : 1;
        out.src = Source::Env; out.origin = a.env;
        return;
      }
      break;
    case Parse::TriState:
      if (present(v)) {
        out.value = v[0] != '0';
        out.src = Source::Env; out.origin = a.env;
        return;
      }
      break;
    case Parse::EqualsOne:
      if (present(v)) {
        out.value = std::strcmp(v, "1") == 0 ? 1 : 0;
        out.src = Source::Env; out.origin = a.env;
        return;
      }
      break;
    case Parse::IntAtoi:
      if (present(v)) {
        out.value = std::atoi(v);
        out.src = Source::Env; out.origin = a.env;
        return;
      }
      break;
    case Parse::IntStrict:
      if (present(v)) {
        char *end = nullptr;
        errno = 0;
        const long n = std::strtol(v, &end, 10);
        if (end != v && *end == '\0' && errno == 0) {
          out.value = n;
          out.src = Source::Env; out.origin = a.env;
        }
        return; // a malformed value keeps the previous layer, as before
      }
      break;
    default:
      break;
    }
  }
  if (found) {
    out.value = any_on ? 1 : 0;
    out.src = Source::Env;
  }
}

} // namespace detail

// Resolves defaults -> family overrides -> TOML -> legacy env. Errors (redundant or reasonless
// overrides, unknown fields) are appended to *err, one per line.
inline ResolvedSettings resolve_settings(Family fam, const EnvView &env, std::string *err = nullptr,
                                         const ConfigValues &config = {}) {
  ResolvedSettings r;
  r.family = fam;
  auto note = [&](const std::string &m) { if (err) *err += m + "\n"; };
  for (const auto &f : long_fields()) f.ref(r).value = f.dflt;
  for (const auto &f : real_fields()) f.ref(r).value = f.dflt;
  for (const auto &f : text_fields()) f.ref(r).value = f.dflt;

  for (const Override &o : family_overrides()) {
    if (o.family != fam)
      continue;
    const Field<long> *hit = nullptr;
    for (const auto &f : long_fields())
      if (std::strcmp(f.name, o.field) == 0) hit = &f;
    if (!hit) { note(std::string("override of unknown field ") + o.field); continue; }
    if (!o.reason || !o.reason[0]) { note(std::string("override of ") + o.field + " has no reason"); continue; }
    if (o.value == hit->dflt) { note(std::string("override of ") + o.field + " equals the default"); continue; }
    auto &slot = hit->ref(r);
    slot.value = o.value; slot.src = Source::Family; slot.reason = o.reason;
  }

  each_field([&](const auto &f) {
    const auto it = config.find(f.name);
    if (it == config.end() || !applies(f, fam)) return;
    try { detail::canonical(f, it->second, f.ref(r)); }
    catch (const std::invalid_argument &e) {
      const std::string message = "graph_" + std::string(f.name) + ": " + e.what();
      if (!err) throw std::invalid_argument(message);
      note(message);
    }
  });
  if (fam == Family::MiMo2 && config.contains("cpu_threads_decode") &&
      r.tun.cpu_threads_decode.value == 0) {
    if (!err) throw std::invalid_argument("graph_cpu_threads_decode must be positive for MiMo");
    note("graph_cpu_threads_decode must be positive for MiMo");
  }
  if (fam == Family::MiMo2 && config.contains("prefill_chunk") &&
      r.tun.prefill_chunk.value > 512) {
    if (!err) throw std::invalid_argument("graph_prefill_chunk must be between 0 and 512 for MiMo");
    note("graph_prefill_chunk must be between 0 and 512 for MiMo");
  }
  for (const auto &[name, value] : config) {
    (void)value;
    bool known = false;
    each_field([&](const auto &f) { known |= name == f.name; });
    if (!known) {
      if (!err) throw std::invalid_argument("unknown setting graph_" + name);
      note("unknown setting graph_" + name);
    }
  }

  for (const auto &f : long_fields())
    detail::read_long(f, fam, env, f.ref(r));
  for (const auto &f : real_fields()) {
    for (const Alias &a : f.aliases) {
      if (!a.all && a.only != fam) continue;
      if (const char *v = env(a.env); detail::present(v)) {
        auto &s = f.ref(r);
        s.value = std::strcmp(v, "auto") == 0 ? -1.0 : std::atof(v);
        s.src = Source::Env; s.origin = a.env;
        break;
      }
    }
  }
  for (const auto &f : text_fields()) {
    for (const Alias &a : f.aliases) {
      if (!a.all && a.only != fam) continue;
      if (const char *v = env(a.env); detail::present(v)) {
        auto &s = f.ref(r);
        s.value = v; s.src = Source::Env; s.origin = a.env;
        break;
      }
    }
  }
  each_field([&](const auto &f) {
    for (const Alias &a : f.aliases) {
      if ((!a.all && a.only != fam) || !detail::present(env(a.env))) continue;
      std::string message = std::string(a.env) + " is a legacy override; use graph_" + f.name + " in TOML";
      if (config.contains(f.name))
        message += " (legacy environment takes precedence over TOML)";
      r.warnings.push_back(std::move(message));
    }
  });
  return r;
}

inline bool validate_config(const ConfigValues &values, const std::vector<Family> &families,
                            std::string &error) {
  for (const auto &[name, value] : values) {
    (void)value;
    bool known = false, supported = false;
    each_field([&](const auto &f) {
      if (name != f.name) return;
      known = true;
      for (Family fam : families) supported |= applies(f, fam);
    });
    if (!known) error += "unknown setting graph_" + name + "\n";
    else if (!supported) error += "graph_" + name + " is not supported by this model's graph adapter\n";
  }
  if (!error.empty()) return false;
  for (Family family : families)
    resolve_settings(family, [](const char *) -> const char * { return nullptr; }, &error, values);
  return error.empty();
}

inline long prefill_chunk_value(const Resolved<long> &chunk, long engine_chunk, long family_fallback) {
  if (chunk.src == Source::Env || (chunk.src == Source::Toml && chunk.value > 0))
    return chunk.value;
  if (engine_chunk > 0) return engine_chunk;
  return chunk.value > 0 ? chunk.value : family_fallback;
}

template<class T> inline std::string setting_value(const Field<T> &f, const T &value) {
  if constexpr (std::is_same_v<T, std::string>) return "\"" + value + "\"";
  else {
    if (auto_flag(f.name, f.parse) && value < 0) return "\"auto\"";
    if (boolean(f.parse) || auto_flag(f.name, f.parse)) return value ? "true" : "false";
    if constexpr (std::is_same_v<T, double>) {
      if (std::strcmp(f.name, "offload_ddr_gib") == 0 && value < 0) return "\"auto\"";
      char text[64];
      std::snprintf(text, sizeof text, "%g", value);
      return text;
    } else return std::to_string(value);
  }
}

inline std::string settings_reference() {
  std::string out = "Advanced graph settings (root TOML keys, before [[gpus]]).\n"
                    "Precedence: defaults < family < TOML < legacy environment.\n"
                    "These are not required setup switches. Use the model recipe first.\n"
                    "Families listed below are the adapters that consume each setting.\n"
                    "zeromoe_cpu does not consume graph settings; use omp_max_threads there.\n";
  each_field([&](const auto &f) {
    out += "\ngraph_" + std::string(f.name) + (f.diagnostic ? " [diagnostic]\n  " : " [tuning]\n  ");
    out += meaning(f.name);
    out += "\n  Family defaults:";
    for (Family fam : {Family::Other, Family::MiMo2, Family::Ling, Family::GlmGraph, Family::Qwen}) {
      if (!applies(f, fam)) continue;
      auto defaults = resolve_settings(fam, [](const char *) -> const char * { return nullptr; });
      out += " " + std::string(family_name(fam)) + "=" + setting_value(f, f.ref(defaults).value);
    }
    out += "\n  Legacy env:";
    for (const auto &a : f.aliases) out += " " + std::string(a.env);
    if (f.aliases.empty()) out += " (none)";
    out += '\n';
  });
  return out;
}

// Base defaults of every tunable, one "name=value" line each, sorted by name.
// The drift hash and the checked-in snapshot test both come from this text.
inline std::string defaults_table() {
  std::vector<std::string> lines;
  char buf[64];
  for (const auto &f : long_fields())
    if (!f.diagnostic) lines.push_back(std::string(f.name) + "=" + std::to_string(f.dflt));
  for (const auto &f : real_fields()) {
    std::snprintf(buf, sizeof buf, "%g", f.dflt);
    lines.push_back(std::string(f.name) + "=" + buf);
  }
  for (const auto &f : text_fields())
    if (!f.diagnostic) lines.push_back(std::string(f.name) + "=" + f.dflt);
  std::sort(lines.begin(), lines.end());
  std::string out;
  for (auto &l : lines) { out += l; out += '\n'; }
  return out;
}

inline uint64_t defaults_hash() {
  uint64_t h = 1469598103934665603ull; // FNV-1a
  for (unsigned char c : defaults_table()) { h ^= c; h *= 1099511628211ull; }
  return h;
}

// Combinations of resolved settings that silently cost performance (each one was a real
// bring-up trap, docs/design/model-family-framework.md s1). Warnings in development;
// ZEROMOE_PREFLIGHT_STRICT=1 (performance runs) turns them into an abort.
inline std::vector<std::string> settings_findings(const ResolvedSettings &r) {
  std::vector<std::string> out;
  const GraphSettings &t = r.tun;
  if (t.phase_cache_mib.value < 0 || t.phase_cache_mib.value > 1024*1024 ||
      t.phase_cache_reserve_mib.value < 0 || t.phase_cache_reserve_mib.value > 1024*1024)
    out.push_back("phase-cache size and reserve must be between 0 and 1048576 MiB");
  if (r.family != Family::Qwen && (t.cpu_threads_decode.value < 1 ||
      (t.cpu_threads_decode.src != Source::Default && t.cpu_threads_decode.value < 10)))
    out.push_back("cpu_threads_decode=" + std::to_string(t.cpu_threads_decode.value) +
                  " is below the common default (10): the early-ready lane runs cold experts on these threads");
  if (t.prefill_op_offload.value != 0 && !t.offload_pin.value && t.gpu_assist_slots.value == 0)
    out.push_back("prefill op-offload may copy experts from an unpinned arena (offload_pin=0): "
                  "copies run at pageable-memory bandwidth");
  if (t.prefill_chunk.value > 0 && t.prefill_chunk.value < t.prefill_min_batch.value)
    out.push_back("prefill_chunk=" + std::to_string(t.prefill_chunk.value) + " is below prefill_min_batch=" +
                  std::to_string(t.prefill_min_batch.value) + ": every chunk would run as one-token steps");
  return out;
}

inline std::string format_settings(const ResolvedSettings &r) {
  std::string out;
  auto line = [&](const char *name, const std::string &val, const auto &x) {
    std::string src = source_name(x.src);
    if (!x.origin.empty()) src += ":" + x.origin;
    if (x.src == Source::Family && !x.reason.empty()) src += " (" + x.reason + ")";
    out += "  graph_" + std::string(name) + " = " + val + " [" + src + "]\n";
  };
  auto settings = r;
  each_field([&](const auto &f) {
    if (!applies(f, r.family)) return;
    const auto &x = f.ref(settings);
    line(f.name, setting_value(f, x.value), x);
  });
  return out;
}

} // namespace zeromoe::graphsettings

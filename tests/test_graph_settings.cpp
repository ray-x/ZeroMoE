#include "graph_health.hpp"
#include "graph_settings.hpp"
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>

using namespace zeromoe::graphsettings;

static int failures = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); ++failures; } } while (0)

static EnvView env_of(std::map<std::string, std::string> m) {
  return [m](const char *n) -> const char * {
    auto it = m.find(n);
    return it == m.end() ? nullptr : it->second.c_str();
  };
}

int main(int argc, char **argv) {
  std::string err;
  for (Family f : {Family::Other, Family::MiMo2, Family::GlmGraph, Family::Ling}) {
    auto defaults = resolve_settings(f, env_of({}));
    CHECK(defaults.tun.phase_cache_mib.value == 0);
    CHECK(defaults.tun.phase_cache_reserve_mib.value == 512);
    auto configured = resolve_settings(f, env_of({{"ZEROMOE_PHASE_CACHE_MIB", "1024"},
                                                  {"ZEROMOE_PHASE_CACHE_RESERVE_MIB", "768"}}));
    CHECK(configured.tun.phase_cache_mib.value == 1024);
    CHECK(configured.tun.phase_cache_reserve_mib.value == 768);
    CHECK(configured.tun.phase_cache_mib.src == Source::Env);
  }
  CHECK(!settings_findings(resolve_settings(Family::Other,
      env_of({{"ZEROMOE_PHASE_CACHE_MIB", "-1"}}))).empty());
  // 1. Values the code used before the resolver existed (empty environment).
  {
    auto m = resolve_settings(Family::MiMo2, env_of({}), &err);
    CHECK(err.empty());
    CHECK(m.tun.cpu_threads_decode.value == 10);
    CHECK(m.tun.prefill_chunk.value == 512);
    CHECK(m.tun.prefill_min_batch.value == 8);
    CHECK(m.tun.prefill_op_offload.value == -1);
    CHECK(m.tun.prefill_offload_min_tokens.value == 256);
    CHECK(m.tun.offload_pinned_pct.value == 50);
    CHECK(m.tun.offload_pin.value == 1);
    CHECK(m.tun.use_gpu1.value == 1);
    CHECK(m.tun.gpu_assist_slots.value == 0);
    CHECK(m.tun.offload_ddr_gib.value < 0);
    CHECK(m.tun.offload_usage_file.value.empty());
    CHECK(m.diag.verify_alias.value == 1 && m.diag.node_trace.value == 0);
    auto l = resolve_settings(Family::Ling, env_of({}), &err);
    CHECK(l.tun.prefill_min_batch.value == 2);
    CHECK(l.tun.prefill_chunk.value == 0);       // engine decides
    CHECK(l.tun.offload_first_layer.value == -1);
    CHECK(l.tun.offload_async.value == 0);
    auto o = resolve_settings(Family::Other, env_of({}), &err);
    CHECK(o.tun.prefill_min_batch.value == 64);  // the FullGraphModel default
    CHECK(o.tun.prefill_chunk.src == Source::Default);
  }
  // 2. Legacy env semantics and family scoping.
  {
    auto m = resolve_settings(Family::MiMo2,
        env_of({{"ZEROMOE_MIMO2_PREFILL_CHUNK", "128"}, {"ZEROMOE_MIMO2_OFFLOAD_PIN", "0"},
                {"ZEROMOE_MIMO2_OFFLOAD_DDR_GIB", "60"}, {"ZEROMOE_MIMO2_CPU_THREADS", "abc"}}));
    CHECK(m.tun.prefill_chunk.value == 128 && m.tun.prefill_chunk.src == Source::Env);
    CHECK(m.tun.offload_pin.value == 0);
    CHECK(m.tun.offload_ddr_gib.value == 60.0);
    CHECK(m.tun.cpu_threads_decode.value == 10);  // malformed strtol keeps the default
    auto l = resolve_settings(Family::Ling, env_of({{"ZEROMOE_MIMO2_PREFILL_CHUNK", "128"}}));
    CHECK(l.tun.prefill_chunk.value == 0);        // another family's knob does not leak
    auto l2 = resolve_settings(Family::Ling, env_of({{"ZEROMOE_LING_PREFILL_CHUNK", "1024"}}));
    CHECK(l2.tun.prefill_chunk.value == 1024);
    auto a = resolve_settings(Family::GlmGraph, env_of({{"ZEROMOE_GLM5NEXT_OFFLOAD_ASYNC", "1"}}));
    CHECK(a.tun.offload_async.value == 1);
    auto f = resolve_settings(Family::GlmGraph, env_of({{"ZEROMOE_LING_OFFLOAD_FIRST_LAYER", "3"},
                                                    {"ZEROMOE_GLM5NEXT_OFFLOAD_FIRST_LAYER", "9"}}));
    CHECK(f.tun.offload_first_layer.value == 3);  // first alias wins, as before
    auto ddr = resolve_settings(Family::MiMo2, env_of({{"ZEROMOE_MIMO2_OFFLOAD_DDR_GIB", "auto"}}));
    CHECK(ddr.tun.offload_ddr_gib.value < 0);
    auto t = resolve_settings(Family::MiMo2, env_of({{"ZEROMOE_MIMO2_PREFILL_OP_OFFLOAD", "0"}}));
    CHECK(t.tun.prefill_op_offload.value == 0);
  }
  // 2b. The GLM graph's offload tier (shared by GLM-5.x, GLM-DSA and Ling).
  {
    auto g = resolve_settings(Family::GlmGraph, env_of({}));
    CHECK(g.tun.gpu_assist_slots.value == 3 && g.tun.gpu_assist_slots.src == Source::Family);
    CHECK(g.tun.offload_pinned_fraction.value == 0.5 && g.tun.offload_probation_fraction.value == 0.25);
    CHECK(g.tun.offload_policy.value == "lru" && g.tun.offload_pin_dma.value == -1);
    CHECK(g.tun.offload_seed_gpu1_hot.value == 1 && g.tun.offload_thp.value == 0);
    auto e = resolve_settings(Family::GlmGraph,
        env_of({{"ZEROMOE_LING_OFFLOAD_PINNED", "0.7"}, {"ZEROMOE_GLM5NEXT_OFFLOAD_PINNED", "0.2"},
                {"ZEROMOE_GLM5NEXT_OFFLOAD_POLICY", "seg"}, {"ZEROMOE_GLM5NEXT_OFFLOAD_PIN", "2"},
                {"ZEROMOE_GLM5NEXT_GPU_ASSIST", "0"}, {"ZEROMOE_GLM5NEXT_OFFLOAD_DDR_GIB", ""}}));
    CHECK(e.tun.offload_pinned_fraction.value == 0.7);      // the Ling name wins, as offload_env did
    CHECK(e.tun.offload_policy.value == "seg");
    CHECK(e.tun.offload_pin_dma.value == 0);                // "== \"1\"" semantics: 2 is off
    CHECK(e.tun.gpu_assist_slots.value == 0 && e.tun.gpu_assist_slots.src == Source::Env);
    CHECK(e.tun.offload_ddr_gib.src == Source::Default);    // empty = unset
    // MiMo's knobs do not see the GLM names, and the GLM graph does not see MiMo's.
    auto m = resolve_settings(Family::MiMo2, env_of({{"ZEROMOE_GLM5NEXT_OFFLOAD_PINNED", "0.9"}, {"ZEROMOE_GLM5NEXT_GPU_ASSIST", "5"}}));
    CHECK(m.tun.offload_pinned_fraction.value == 0.5 && m.tun.gpu_assist_slots.value == 0);
    auto g2 = resolve_settings(Family::GlmGraph, env_of({{"ZEROMOE_MIMO2_GPU_ASSIST", "5"}}));
    CHECK(g2.tun.gpu_assist_slots.value == 3);
  }
  // 2c. GLM lane/staging switches keep their legacy parses.
  {
    auto d = resolve_settings(Family::GlmGraph, env_of({}));
    CHECK(d.tun.early_ready.value == -1 && d.tun.split_load.value == 1 && d.tun.lane_with_gpu1.value == 1);
    CHECK(d.tun.pilot_copy.value == 1 && d.tun.gpu_assist_any_type.value == 0 && d.tun.dynamic_cache.value == 0);
    CHECK(d.tun.gpu_assist2_slots.value == 1);
    auto e = resolve_settings(Family::GlmGraph, env_of({{"ZEROMOE_GLM5NEXT_EARLY_READY", "0"}, {"ZEROMOE_GLM5NEXT_SPLIT_LOAD", "0"},
        {"ZEROMOE_GLM5NEXT_PILOT_COPY", "0"}, {"ZEROMOE_GLM5NEXT_DYNAMIC_CACHE", "1"}, {"ZEROMOE_GLM5NEXT_ASSIST2_SLOTS", "4"}}));
    CHECK(e.tun.early_ready.value == 0 && e.tun.split_load.value == 0 && e.tun.pilot_copy.value == 0);
    CHECK(e.tun.dynamic_cache.value == 1 && e.tun.gpu_assist2_slots.value == 4);
    auto f = resolve_settings(Family::GlmGraph, env_of({{"ZEROMOE_GLM5NEXT_EARLY_READY", "1"}}));
    CHECK(f.tun.early_ready.value == 1);
    auto m = resolve_settings(Family::MiMo2, env_of({{"ZEROMOE_GLM5NEXT_EARLY_READY", "0"}}));
    CHECK(m.tun.early_ready.value == -1);                   // MiMo does not see the GLM names
  }
  // 3. Overrides: layer order and provenance.
  {
    auto m = resolve_settings(Family::MiMo2, env_of({}));
    CHECK(m.tun.prefill_min_batch.src == Source::Family && !m.tun.prefill_min_batch.reason.empty());
    auto e = resolve_settings(Family::MiMo2, env_of({}));
    CHECK(e.tun.cpu_threads_decode.src == Source::Default);
  }
  // 4. Every override has a reason, differs from the default and names a real field.
  {
    for (Family f : {Family::MiMo2, Family::Ling, Family::Other, Family::GlmGraph}) {
      std::string e;
      resolve_settings(f, env_of({}), &e);
      CHECK(e.empty());
    }
    for (const auto &o : family_overrides()) CHECK(o.reason && o.reason[0]);
  }
  // 4b. Preflight findings: silent on every family's defaults, loud on the known traps.
  {
    for (Family f : {Family::MiMo2, Family::Ling, Family::Other, Family::GlmGraph})
      CHECK(settings_findings(resolve_settings(f, env_of({}))).empty());
    auto t = resolve_settings(Family::MiMo2, env_of({{"ZEROMOE_MIMO2_CPU_THREADS", "1"}}));
    CHECK(settings_findings(t).size() == 1);
    auto p = resolve_settings(Family::MiMo2, env_of({{"ZEROMOE_MIMO2_OFFLOAD_PIN", "0"}}));
    CHECK(settings_findings(p).size() == 1);
    auto p2 = resolve_settings(Family::MiMo2, env_of({{"ZEROMOE_MIMO2_OFFLOAD_PIN", "0"},
                                                      {"ZEROMOE_MIMO2_PREFILL_OP_OFFLOAD", "0"}}));
    CHECK(settings_findings(p2).empty());        // op-offload off: an unpinned arena is fine
    auto c = resolve_settings(Family::MiMo2, env_of({{"ZEROMOE_MIMO2_PREFILL_CHUNK", "4"}}));
    CHECK(settings_findings(c).size() == 1);
    auto st = resolve_settings(Family::Other, env_of({{"ZEROMOE_PREFLIGHT_STRICT", "1"}}));
    CHECK(st.diag.preflight_strict.value == 1);
  }
  // 4c. Run-time health findings.
  {
    using namespace zeromoe::graph;
    const auto all = measure_expert_coverage(1, 2, 4,
        [](int, int e) { return e < 2; }, [](int, int e) { return e >= 2; });
    CHECK(all.total == 8 && all.covered == 8 && all.duplicated == 0);
    CHECK(coverage_finding(all).empty() && coverage_finding(ExpertCoverage{}).empty());
    // Raw tier counts add up to capacity, but duplicated identities leave
    // two experts per layer uncovered. Layer zero is deliberately excluded.
    const auto overlap = measure_expert_coverage(1, 2, 4,
        [](int l, int e) { return l > 0 && e < 2; }, [](int, int e) { return e < 2; });
    CHECK(overlap.vram + overlap.ddr == 8 && overlap.covered == 4);
    CHECK(overlap.duplicated == 4 && overlap.missing() == 4);
    CHECK(!coverage_finding(overlap).empty());
    // An empty/unseeded DDR pool cannot pass because its capacity is large.
    const auto empty_ddr = measure_expert_coverage(3, 1, 4,
        [](int, int e) { return e == 0; }, [](int, int) { return false; });
    CHECK(empty_ddr.ddr == 0 && empty_ddr.missing() == 3);
    CHECK(!coverage_finding(empty_ddr).empty());
    CHECK(decode_miss_finding(5, 5, 32).empty() && !decode_miss_finding(9, 5, 32).empty());
    CHECK(idle_vram_finding(0, 3.9).empty() && !idle_vram_finding(1, 4.5).empty());
  }
  // 4d. The route-survey record file: per-family alias, diagnostic (outside the defaults table).
  {
    auto m = resolve_settings(Family::MiMo2, env_of({{"ZEROMOE_MIMO2_USAGE_RECORD", "/tmp/u"}, {"ZEROMOE_LING_USAGE_RECORD", "/x"}}));
    CHECK(m.diag.usage_record_file.value == "/tmp/u");
    auto g = resolve_settings(Family::GlmGraph, env_of({{"ZEROMOE_LING_USAGE_RECORD", "/l"}, {"ZEROMOE_GLM5NEXT_USAGE_RECORD", "/g"}, {"ZEROMOE_MIMO2_USAGE_RECORD", "/m"}}));
    CHECK(g.diag.usage_record_file.value == "/l");
    CHECK(resolve_settings(Family::Other, env_of({{"ZEROMOE_MIMO2_USAGE_RECORD", "/m"}})).diag.usage_record_file.value.empty());
    CHECK(defaults_table().find("usage_record_file") == std::string::npos);   // diagnostics never enter the hash
  }
  // 5. Defaults table vs the checked-in snapshot (a default change shows up in review).
  if (argc > 1) {
    std::ifstream in(argv[1]);
    std::stringstream ss; ss << in.rdbuf();
    if (ss.str() != defaults_table()) {
      std::fprintf(stderr, "defaults snapshot differs; update %s with:\n%s", argv[1], defaults_table().c_str());
      ++failures;
    }
  }
  std::printf("%s", format_settings(resolve_settings(Family::MiMo2, env_of({}))).c_str());
  std::printf("defaults_hash=%016llx\n", (unsigned long long)defaults_hash());
  if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
  std::printf("graph_settings_test: OK\n");
  return 0;
}

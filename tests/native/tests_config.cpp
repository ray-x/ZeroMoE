// Native test harness: tests_config (split from tests/native/main_test.cpp).
#include "native_tests.hpp"

// Test 32: a config allow-list must prevent an omitted physical adapter from
// becoming an ExpertVault. This is intentionally a topology-only test: it
// creates streams but no model, RAM tier, or expert cache allocation.
bool test_gpu_device_allowlist() {
    std::cout << "[Test 32] GPU config allow-list (omitted adapters stay unused)...\n";
    CudaManager gpu;
    gpu.topology.configured_device_ids = {0};
    if (!gpu.initialize()) {
        std::cout << "  -> Skipped (No CUDA Device Found on system)\n";
        return true;
    }
    bool ok = gpu.topology.core_gpu >= 0 &&
              gpu.topology.devices[(size_t)gpu.topology.core_gpu].id == 0 &&
              gpu.topology.vault_gpus.size() == 1 &&
              gpu.topology.vault_gpus[0] == gpu.topology.core_gpu;
    for (const auto& dev : gpu.topology.devices) {
        if (dev.id == 0) {
            ok = ok && dev.role == GpuRole::CoreEngine;
        } else {
            ok = ok && dev.role == GpuRole::Unused;
        }
    }
    std::cout << "  -> " << (ok ? "PASSED" : "FAILED")
              << " (configured IDs={0}, physical devices="
              << gpu.topology.devices.size() << ")\n";
    return ok;
}

bool test_qwen_prefill_state_handoff_config() {
    std::cout << "[Test] Qwen prefill state handoff config...\n";
    char config_path[] = "/tmp/zeromoe_state_handoff_config_XXXXXX";
    const int config_fd = mkstemp(config_path);
    if (config_fd < 0)
        return false;
    close(config_fd);
    bool ok = !EngineConfig{}.qwen4exp_prefill_state_handoff;
    for (const std::string& setting : {std::string(),
                                     std::string("qwen4exp_prefill_state_handoff = true\n"),
                                     std::string("qwen4exp_prefill_state_handoff = false\n")}) {
        {
            std::ofstream output(config_path);
            output << "model_path = \"/test/model.gguf\"\n" << setting;
            ok = ok && output.good();
        }
        EngineConfig config;
        const bool loaded = config.load(config_path);
        ok = ok && loaded &&
             config.qwen4exp_prefill_state_handoff ==
                 (setting.find("true") != std::string::npos);
    }
    EngineConfig::write_sample(config_path);
    EngineConfig sample;
    ok = sample.load(config_path) && !sample.qwen4exp_prefill_state_handoff && ok;
    std::remove(config_path);
    std::cout << "  -> " << (ok ? "PASSED" : "FAILED") << "\n";
  return ok;
}


bool test_graph_config() {
    std::cout << "[Test] graph_config env readers and registry...\n";
    namespace gc = zeromoe::graphcfg;
    const char *k = "ZEROMOE_TEST_GRAPH_CONFIG_KNOB";
    unsetenv(k);
    bool ok = !gc::present(k) && !gc::flag_on(k) && gc::flag_unless_off(k) &&
              gc::integer(k, 7) == 7 && gc::real(k, 1.5) == 1.5 &&
              gc::path(k) == nullptr;
    setenv(k, "1", 1);
    ok = ok && gc::present(k) && gc::flag_on(k) && gc::flag_unless_off(k) &&
         gc::integer(k, 7) == 1 && std::string(gc::env(k)) == "1";
    setenv(k, "0", 1);
    ok = ok && !gc::flag_on(k) && !gc::flag_unless_off(k) && gc::integer(k, 7) == 0;
    setenv(k, "12x", 1);  // strict whole-string parse, falls back to default
    ok = ok && gc::integer(k, 7) == 7;
    setenv(k, "", 1);     // empty is "set" for present() but not for path()
    ok = ok && gc::present(k) && gc::path(k) == nullptr && gc::integer(k, 9) == 9;
    setenv(k, "2.5", 1);
    ok = ok && gc::real(k, 0.0) == 2.5;
    // The registry saw the name once, with the value of the FIRST lookup.
    const auto snap = gc::Registry::get().snapshot();
    const auto it = snap.find(k);
    ok = ok && it != snap.end() && !it->second.set && it->second.reads >= 1;
    unsetenv(k);
    std::cout << "  -> " << (ok ? "PASSED" : "FAILED") << "\n";
    return ok;
}


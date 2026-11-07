// DSpark (DFlash) draft GGUF: metadata parse + topology validation on the real file.
// Run: ZEROMOE_TEST_ONLY_DSPARK_CONFIG=1 ZEROMOE_TEST_DSPARK_GGUF=<path> zeromoe_tests
#include "native_tests.hpp"
#include "dspark_config.hpp"

#include <cstdlib>
#include <iostream>

bool test_dspark_config() {
  std::cout << "[Test] DSpark (DFlash) draft GGUF config and tensor topology...\n";
  const char *env = std::getenv("ZEROMOE_TEST_DSPARK_GGUF");
  const std::string path = env ? env
      : std::string(std::getenv("HOME") ? std::getenv("HOME") : "") +
        "/vllm-workspace/DeepSeek-V41-Flash/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf";
  GGUFFile f;
  if (access(path.c_str(), R_OK) != 0) {
    std::cout << "  -> SKIPPED (no draft GGUF at " << path << ")\n";
    return true;
  }
  if (!f.load_header(path)) {
    std::cout << "  -> FAILED: could not read GGUF header\n";
    return false;
  }
  zeromoe::dspark::Config c;
  std::string err;
  if (!zeromoe::dspark::parse_config(f, c, err)) {
    std::cout << "  -> FAILED parse_config: " << err << "\n";
    return false;
  }
  if (!zeromoe::dspark::validate_tensors(f, c, err)) {
    std::cout << "  -> FAILED validate_tensors: " << err << "\n";
    return false;
  }
  std::cout << "  -> layers=" << c.n_layer << " n_embd=" << c.n_embd << " experts=" << c.n_expert
            << "/top" << c.n_expert_used << " block_size=" << c.block_size << " swa=" << c.swa
            << " hc=" << c.hc_count << " target_layers=";
  for (int l : c.target_layers) std::cout << l << ",";
  std::cout << " markov_rank=" << c.markov_rank << " vocab=" << c.n_vocab << "\n";
  // Both known drafts: 3 stages, width 5120, 128 experts, block 5, taps 37-39, Markov rank 256.
  bool ok = c.n_layer == 3 && c.n_embd == 5120 && c.block_size == 5 && c.n_expert == 128 &&
            c.target_layers.size() == 3 && c.markov_rank == 256 && c.n_vocab == 129280 &&
            c.has_confidence_head && c.sample_from_anchor;
  std::cout << "  -> yarn=" << c.rope_yarn << " factor=" << c.rope_yarn_factor
            << " orig_ctx=" << c.rope_orig_ctx << " dsv41_semantics=" << c.dsv41_semantics << "\n";
  std::cout << "  -> " << (ok ? "PASSED" : "FAILED (unexpected draft values)") << "\n";
  return ok;
}

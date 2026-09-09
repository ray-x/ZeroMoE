// tools/gguf_tool.cpp
//
// Single entry point for the GGUF offline tooling: pack (align/unpack
// shards for O_DIRECT streaming), io (A/B read-throughput bench, original
// vs repacked), and verify (pre-flight completeness + byte-compare gate).
// Dispatches argv[1] to the matching cmd_* in gguf_common.hpp, forwarding
// (argc-1, argv+1) so each subcommand parses its own args starting at its
// own argv[0] == the subcommand name.
//
// Build (no CMake / no CUDA -- macOS and Linux):
//   c++ -std=c++23 -O2 -Iinclude tools/gguf_tool.cpp tools/gguf_pack.cpp \
//       tools/gguf_io.cpp tools/gguf_verify.cpp tools/gguf_plan.cpp -o build/gguf_tool
#include "gguf_common.hpp"
#include <cstdio>
#include <cstring>

namespace {

int usage(const char *prog) {
  fprintf(stderr,
          "Usage: %s <subcommand> [args...]\n"
          "\n"
          "Subcommands:\n"
          "  pack    <source_shard_file_or_dir> <output_dir> [options]\n"
          "          Re-align (default) or unpack (--unpack) GGUF shard(s) "
          "for O_DIRECT streaming.\n"
          "  io      <orig_model> <repacked_model> [options]\n"
          "          A/B read-throughput bench: original (always plain "
          "buffered pread) vs repacked\n"
          "          (tight reads, optionally --direct for O_DIRECT/DMA).\n"
          "  verify  <model_file_or_dir> [--against <other_model>]\n"
          "          Pre-flight completeness gate (every per-layer "
          "attention/norm tensor present in\n"
          "          all layers) + optional exhaustive byte-compare "
          "against another model.\n"
          "  plan    <model_file_or_dir> [--usage-file <path>]\n"
          "          Read-only sizing advisor: dense/MoE VRAM requirement, "
          "detected GPU/RAM, and a\n"
          "          .moestream_usage summary -- for picking dense_gb / "
          "moe_cache_gb / pinned_vram_capacity\n"
          "          on a new model without trial-and-error real runs.\n"
          "  dma     <model.gguf|dir> [<model2> ...] [--experts N[,N2,...]] "
          "[--submissions M]\n"
          "          A/B of the REAL production io_uring/O_DIRECT read path: "
          "1 combined DMA read\n"
          "          per expert vs 3 separate DMA reads (gate/up/down) at "
          "the same file offsets --\n"
          "          random access across every layer. Linux-only for a "
          "real run.\n"
          "  mix     <baseline_dir> <cold_dir> <output_dir> --usage-file "
          "<path> [options]\n"
          "          Builds a real per-expert mixed-precision override "
          "shard: the coldest N%% of\n"
          "          each eligible layer's experts (by real traffic) come "
          "from <cold_dir>, the\n"
          "          rest stay on <baseline_dir>. Pure byte-copy, no "
          "requantization.\n"
          "\n"
          "Run '%s <subcommand>' with no further args for that "
          "subcommand's full usage.\n",
          prog, prog);
  return 2;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2)
    return usage(argv[0]);

  std::string sub = argv[1];
  // Forward (argc-1, argv+1): the subcommand's own argv[0] becomes the
  // subcommand name (e.g. "pack"), matching cmd_pack/cmd_io/cmd_verify's
  // existing argv[1]/argv[2]-based argument parsing.
  if (sub == "pack")
    return gguf_tool::cmd_pack(argc - 1, argv + 1);
  if (sub == "io")
    return gguf_tool::cmd_io(argc - 1, argv + 1);
  if (sub == "verify")
    return gguf_tool::cmd_verify(argc - 1, argv + 1);
  if (sub == "plan")
    return gguf_tool::cmd_plan(argc - 1, argv + 1);
  if (sub == "dma")
    return gguf_tool::cmd_dma(argc - 1, argv + 1);
  if (sub == "mix")
    return gguf_tool::cmd_mix(argc - 1, argv + 1);
  if (sub == "-h" || sub == "--help") {
    usage(argv[0]);
    return 0;
  }

  fprintf(stderr, "%s: unknown subcommand '%s'\n\n", argv[0], sub.c_str());
  return usage(argv[0]);
}

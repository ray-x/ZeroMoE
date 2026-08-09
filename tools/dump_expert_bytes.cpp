// dump_expert_bytes.cpp — extract one expert's raw quantized bytes from a
// packed blk.<L>.ffn_{gate,up,down}_exps.weight tensor and write them to a
// flat binary file.
//
// Built to feed src/main_test.cpp's real-expert IQ2_XXS dequant test: a live
// VERBOSE trace on a real DeepSeek-V4-Flash decode run showed specific
// experts producing an exactly-all-zero SiLU output on the GPU path, fully
// deterministic across repeated runs, while that block's own dequant scale
// samples completely normal (same order of magnitude as experts that DON'T
// zero out). A runtime cpu_only toggle can't isolate GPU-kernel-specific
// behavior from routing/dispatch-gating uncertainty; pulling the exact real
// bytes out once, offline, and feeding them through both dequant paths in
// total isolation can. Reuses GGUFFile/fd_for_tensor (the same tensor-offset
// math load_expert_slab uses in production) rather than re-deriving it.
//
// No CUDA dependency; builds with the system C++ compiler, e.g.:
//   g++ -std=c++23 -O2 -Iinclude tools/dump_expert_bytes.cpp -o /tmp/dump_expert_bytes
#include "gguf_loader.hpp"
#include <cstdlib>
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0]
              << " <model.gguf> <layer> <expert_id> <gate|up|down> [out.bin]\n";
    return 1;
  }
  const std::string model_path = argv[1];
  const int layer = std::atoi(argv[2]);
  const int expert_id = std::atoi(argv[3]);
  const std::string which = argv[4];
  const std::string out_path =
      argc > 5 ? argv[5]
                : ("expert_l" + std::to_string(layer) + "_e" +
                   std::to_string(expert_id) + "_" + which + ".bin");

  GGUFFile file;
  if (!file.load_header(model_path)) {
    std::cerr << "Failed to load GGUF: " << model_path << "\n";
    return 1;
  }

  const std::string tensor_name =
      "blk." + std::to_string(layer) + ".ffn_" + which + "_exps.weight";
  auto it = file.tensors.find(tensor_name);
  if (it == file.tensors.end()) {
    std::cerr << "Tensor not found: " << tensor_name << "\n";
    return 1;
  }
  const GGUFTensor &t = it->second;
  if (t.shape.empty()) {
    std::cerr << tensor_name << " has no shape info\n";
    return 1;
  }
  // Packed layout: expert index is the LAST dimension (see
  // StreamingEngine::load_expert_slab's own comment) -- expert E's bytes are
  // a contiguous [E*per_expert, (E+1)*per_expert) slice.
  const uint64_t n_experts = t.shape.back();
  if ((uint64_t)expert_id >= n_experts) {
    std::cerr << "expert_id " << expert_id << " out of range (n_experts=" << n_experts << ")\n";
    return 1;
  }
  if (t.size_bytes % n_experts != 0) {
    std::cerr << "Tensor size " << t.size_bytes << " doesn't divide evenly by n_experts=" << n_experts << "\n";
    return 1;
  }
  const uint64_t per_expert_bytes = t.size_bytes / n_experts;
  const uint64_t expert_offset = t.offset + (uint64_t)expert_id * per_expert_bytes;

  std::vector<uint8_t> buf(per_expert_bytes);
  const int fd = file.fd_for_tensor(t);
  const ssize_t r = pread(fd, buf.data(), per_expert_bytes, (off_t)expert_offset);
  if (r < 0 || (uint64_t)r != per_expert_bytes) {
    std::cerr << "pread failed: got " << r << " of " << per_expert_bytes << " bytes\n";
    return 1;
  }

  std::ofstream out(out_path, std::ios::binary);
  out.write(reinterpret_cast<const char *>(buf.data()), (std::streamsize)buf.size());
  std::cout << tensor_name << " expert " << expert_id
            << ": type=" << static_cast<uint32_t>(t.type) << " "
            << per_expert_bytes << " bytes -> " << out_path << "\n";
  return 0;
}

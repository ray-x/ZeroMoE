// list_tensors.cpp — print all tensor names/shapes from a GGUF file.
// No CUDA dependency; builds with the system C++ compiler.
#include "gguf_loader.hpp"
#include <algorithm>
#include <iostream>

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model.gguf|dir>\n";
    return 1;
  }
  GGUFFile file;
  if (!file.load_header(argv[1])) {
    std::cerr << "Failed to load GGUF: " << argv[1] << "\n";
    return 1;
  }

  std::vector<std::pair<std::string, const GGUFTensor *>> sorted;
  sorted.reserve(file.tensors.size());
  for (const auto &[name, t] : file.tensors)
    sorted.push_back({name, &t});
  std::sort(sorted.begin(), sorted.end());

  for (const auto &[name, t] : sorted) {
    std::cout << name << " type=" << static_cast<uint32_t>(t->type)
              << " shape=[";
    for (size_t i = 0; i < t->shape.size(); ++i) {
      if (i)
        std::cout << ",";
      std::cout << t->shape[i];
    }
    std::cout << "] bytes=" << t->size_bytes << "\n";
  }
  return 0;
}

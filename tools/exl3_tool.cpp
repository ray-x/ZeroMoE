#include "safetensors_adapter.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

const char *codebook_name(Exl3Codebook codebook) {
  switch (codebook) {
  case Exl3Codebook::MCG:
    return "mcg";
  case Exl3Codebook::MUL1:
    return "mul1";
  case Exl3Codebook::Default:
    return "3inst";
  }
  return "unknown";
}

bool load_model(const std::string &path, SafetensorsModelInfo &model) {
  namespace fs = std::filesystem;
  std::error_code ec;
  if (fs::is_directory(path, ec))
    return SafetensorsAdapter::parse_directory(path, model);
  return SafetensorsAdapter::parse_file(path, model);
}

bool verify_shard_set(const std::string &path, int &present, int &expected) {
  namespace fs = std::filesystem;
  std::error_code ec;
  if (!fs::is_directory(path, ec))
    return true;

  const std::regex pattern(R"(.*-([0-9]+)-of-([0-9]+)\.safetensors$)");
  std::set<int> indices;
  for (const auto &entry : fs::directory_iterator(path, ec)) {
    if (ec)
      return false;
    std::smatch match;
    const std::string name = entry.path().filename().string();
    bool regular = entry.is_regular_file(ec);
    if (!regular && !ec && entry.is_symlink(ec)) {
      ec.clear();
      regular = fs::is_regular_file(entry.status(ec));
    }
    if (!regular || ec || !std::regex_match(name, match, pattern))
      continue;
    indices.insert(std::stoi(match[1].str()));
    expected = std::max(expected, std::stoi(match[2].str()));
  }
  present = static_cast<int>(indices.size());
  if (expected == 0)
    return true;
  bool complete = true;
  std::ostringstream missing;
  for (int index = 1; index <= expected; ++index) {
    if (indices.find(index) == indices.end()) {
      missing << " " << index;
      complete = false;
    }
  }
  if (!complete)
    std::cerr << "[EXL3] missing shards:" << missing.str() << " of "
              << expected << " in " << path << "\n";
  return complete && present == expected;
}

bool payload_smoke_read(const SafetensorsModelInfo &model) {
  std::map<std::string, std::set<std::string>> selected;
  for (const auto &entry : model.tensors) {
    const std::string &name = entry.first;
    std::string kind;
    if (name.ends_with(".trellis"))
      kind = "trellis";
    else if (name.ends_with(".suh"))
      kind = "suh";
    else if (name.ends_with(".svh"))
      kind = "svh";
    else
      continue;
    const auto &info = entry.second;
    if (info.file_path.empty() || info.length == 0)
      return false;
    if (!selected[info.file_path].insert(kind).second)
      continue;
    std::ifstream file(info.file_path, std::ios::binary);
    if (!file)
      return false;
    file.seekg(static_cast<std::streamoff>(info.offset));
    const size_t bytes = static_cast<size_t>(std::min<std::uint64_t>(info.length, 4096));
    std::vector<char> buffer(bytes);
    file.read(buffer.data(), static_cast<std::streamsize>(bytes));
    if (file.gcount() != static_cast<std::streamsize>(bytes))
      return false;
  }
  return !selected.empty();
}

int verify(const std::string &path) {
  int present_shards = 0;
  int expected_shards = 0;
  const bool shards_complete = verify_shard_set(path, present_shards, expected_shards);
  SafetensorsModelInfo model;
  if (!load_model(path, model)) {
    std::cerr << "[EXL3] failed to parse safetensors model: " << path << "\n";
    return 1;
  }
  if (!SafetensorsAdapter::is_exl3_model(model)) {
    std::cerr << "[EXL3] no .trellis tensors found: " << path << "\n";
    return 1;
  }

  const Exl3Contract contract = SafetensorsAdapter::map_exl3_contract(model);
  const bool payload_ok = payload_smoke_read(model);
  std::map<double, int> rates;
  std::map<std::string, int> codebooks;
  int invalid_linears = 0;
  for (const auto &linear : contract.linears) {
    if (!linear.valid) {
      ++invalid_linears;
      if (invalid_linears <= 8)
        std::cerr << "[EXL3] invalid tensor group: " << linear.key << "\n";
    }
    rates[linear.bits_per_weight]++;
    codebooks[codebook_name(linear.codebook)]++;
  }

  std::cout << "[EXL3] tensors=" << model.tensors.size()
            << " linears=" << contract.linears.size()
            << " layers=" << contract.n_layers
            << " experts=" << contract.n_experts << "\n";
  if (expected_shards > 0)
    std::cout << "[EXL3] shards=" << present_shards << "/" << expected_shards
              << "\n";
  std::cout << "[EXL3] payload_smoke=" << (payload_ok ? "passed" : "failed")
            << "\n";
  std::cout << "[EXL3] complete=" << (contract.complete ? "yes" : "no")
            << "\n";
  if (invalid_linears > 0)
    std::cout << "[EXL3] invalid_groups=" << invalid_linears << "\n";
  std::cout << "[EXL3] rates:";
  for (const auto &entry : rates)
    std::cout << " " << entry.first << "bpw=" << entry.second;
  std::cout << "\n[EXL3] codebooks:";
  for (const auto &entry : codebooks)
    std::cout << " " << entry.first << "=" << entry.second;
  std::cout << "\n";
  return contract.complete && shards_complete && payload_ok ? 0 : 1;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3 || std::string(argv[1]) != "verify") {
    std::cerr << "Usage: " << argv[0] << " verify <model.safetensors|model_dir>\n";
    return 2;
  }
  return verify(argv[2]);
}

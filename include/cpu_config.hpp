// cpu_config.hpp — Minimal runtime configuration for moestream_cpu.
//
// This is a deliberately stripped-down, CUDA-free config loader for the
// CPU-only inference binary.  It reuses the same TOML syntax as the full
// EngineConfig but only parses the fields that matter when no GPU is
// present.
#pragma once
#include "logger.hpp"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace cpu_toml {

inline std::string trim(const std::string &s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  size_t b = s.find_last_not_of(" \t\r\n");
  return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
}

inline std::string strip_comment(const std::string &line) {
  bool in_string = false;
  for (size_t i = 0; i < line.size(); ++i) {
    if (line[i] == '"')
      in_string = !in_string;
    if (line[i] == '#' && !in_string)
      return line.substr(0, i);
  }
  return line;
}

struct Value {
  std::string str;
  double num = 0;
  bool boolean = false;
  std::vector<int> int_array;
  enum Type { String, Number, Bool, IntArray } type = String;
};

struct Table {
  std::unordered_map<std::string, Value> values;

  bool has(const std::string &key) const { return values.count(key) > 0; }

  std::string get_str(const std::string &key,
                      const std::string &def = "") const {
    auto it = values.find(key);
    return (it != values.end()) ? it->second.str : def;
  }
  int get_int(const std::string &key, int def = 0) const {
    auto it = values.find(key);
    return (it != values.end() && it->second.type == Value::Number)
               ? (int)it->second.num
               : def;
  }
  double get_double(const std::string &key, double def = 0.0) const {
    auto it = values.find(key);
    return (it != values.end() && it->second.type == Value::Number)
               ? it->second.num
               : def;
  }
  float get_float(const std::string &key, float def = 0.0f) const {
    return (float)get_double(key, def);
  }
  bool get_bool(const std::string &key, bool def = false) const {
    auto it = values.find(key);
    return (it != values.end() && it->second.type == Value::Bool)
               ? it->second.boolean
               : def;
  }
  std::vector<int> get_int_array(const std::string &key) const {
    auto it = values.find(key);
    return (it != values.end() && it->second.type == Value::IntArray)
               ? it->second.int_array
               : std::vector<int>{};
  }
};

struct Document {
  Table root;
  std::vector<Table> gpu_tables;
};

inline Value parse_value(const std::string &raw) {
  Value v;
  std::string s = trim(raw);
  if (s.empty())
    return v;

  if (s.front() == '"' && s.back() == '"') {
    v.type = Value::String;
    v.str = s.substr(1, s.size() - 2);
    return v;
  }
  if (s == "true" || s == "false") {
    v.type = Value::Bool;
    v.boolean = (s == "true");
    v.str = s;
    return v;
  }
  if (s.front() == '[' && s.back() == ']') {
    v.type = Value::IntArray;
    v.str = s;
    std::string inner = s.substr(1, s.size() - 2);
    std::istringstream elems(inner);
    std::string tok;
    while (std::getline(elems, tok, ',')) {
      std::string t = trim(tok);
      if (t.empty())
        continue;
      try {
        v.int_array.push_back(std::stoi(t));
      } catch (...) {
      }
    }
    return v;
  }
  try {
    v.num = std::stod(s);
    v.type = Value::Number;
    v.str = s;
  } catch (...) {
    v.type = Value::String;
    v.str = s;
  }
  return v;
}

inline Document parse(const std::string &content) {
  Document doc;
  Table *current = &doc.root;

  std::istringstream stream(content);
  std::string line;
  while (std::getline(stream, line)) {
    line = trim(strip_comment(line));
    if (line.empty())
      continue;

    if (line == "[[gpus]]") {
      doc.gpu_tables.emplace_back();
      current = &doc.gpu_tables.back();
      continue;
    }
    if (line.front() == '[' && line.back() == ']') {
      current = &doc.root;
      continue;
    }
    size_t eq = line.find('=');
    if (eq == std::string::npos)
      continue;
    std::string key = trim(line.substr(0, eq));
    std::string val_str = trim(line.substr(eq + 1));
    current->values[key] = parse_value(val_str);
  }
  return doc;
}

inline Document parse_file(const std::string &path) {
  std::ifstream f(path);
  if (!f.is_open()) {
    std::cerr << "[CPU-CONFIG] Cannot open: " << path << "\n";
    return {};
  }
  std::string content((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
  return parse(content);
}

} // namespace cpu_toml

struct CPUConfig {
  std::string model_path;
  int context_size = 4096;
  int max_new_tokens = 512;
  float temperature = 0.7f;
  float top_p = 0.95f;

  // Host-side expert cache (the pin + LRU working set).
  double host_ram_gb = 4.0;
  int promote_hit_threshold = 2;
  std::string pin_file = ".moestream_usage";
  int pin_threshold = 4;
  int pin_topk_per_layer = 2;
  int expert_fetch_threads = 4;
  int omp_max_threads = 0;

  // Tokenizer / chat template behavior.
  bool raw_prompt = false;
  // Optional system-role message, prepended ahead of the user turn when a
  // chat template is in use (see GGUFTokenizer::has_chat_template /
  // has_deepseek_chat_template). Empty by default -- MoEStream never
  // synthesizes a system prompt on its own, matching what the real
  // DeepSeek-V4 jinja template produces when no "system" message is passed
  // in (confirmed byte-for-byte against
  // models/templates/deepseek-ai-DeepSeek-V4-Flash-0731.jinja in a real
  // llama.cpp checkout: no system role -> ns.system_prompt stays "" ->
  // {bos}<｜User｜>{msg}<｜Assistant｜></think>, exactly what this engine
  // already sent). A base model with no language-steering instruction will
  // happily answer an ambiguous/short English prompt in Chinese -- this is
  // normal, well-documented behavior for DeepSeek/Qwen/GLM-family models,
  // not evidence of a missing/broken template. Set this (or --system-prompt)
  // to add one, e.g. "Respond in English unless asked otherwise."
  std::string system_prompt;
  std::string log_level = "INFO";
  std::string log_path;

  bool load(const std::string &path) {
    auto doc = cpu_toml::parse_file(path);
    if (doc.root.values.empty() && doc.gpu_tables.empty())
      return false;

    model_path = doc.root.get_str("model_path", model_path);
    context_size = doc.root.get_int("context_size", context_size);
    max_new_tokens = doc.root.get_int("max_new_tokens", max_new_tokens);
    temperature = doc.root.get_float("temperature", temperature);
    top_p = doc.root.get_float("top_p", top_p);
    host_ram_gb = doc.root.get_double("host_ram_gb", host_ram_gb);
    promote_hit_threshold =
        doc.root.get_int("promote_hit_threshold", promote_hit_threshold);
    pin_file = doc.root.get_str("pin_file", pin_file);
    pin_threshold = doc.root.get_int("pin_threshold", pin_threshold);
    pin_topk_per_layer =
        doc.root.get_int("pin_topk_per_layer", pin_topk_per_layer);
    expert_fetch_threads =
        doc.root.get_int("expert_fetch_threads", expert_fetch_threads);
    omp_max_threads = doc.root.get_int("omp_max_threads", omp_max_threads);
    raw_prompt = doc.root.get_bool("raw_prompt", raw_prompt);
    system_prompt = doc.root.get_str("system_prompt", system_prompt);
    log_level = doc.root.get_str("log_level", log_level);
    log_path = doc.root.get_str("log_path", log_path);

    if (model_path.empty()) {
      std::cerr << "[CPU-CONFIG] Error: model_path is required.\n";
      return false;
    }
    return true;
  }

  void print() const {
    LOG_STREAM_INFO() << "[CPU-CONFIG] Model: " << model_path << "\n";
    LOG_STREAM_INFO() << "[CPU-CONFIG] Context: " << context_size
                      << " tokens | Max gen: " << max_new_tokens << "\n";
    LOG_STREAM_INFO() << "[CPU-CONFIG] Temperature: " << temperature
                      << " | Top-p: " << top_p << "\n";
    LOG_STREAM_INFO() << "[CPU-CONFIG] Host RAM budget: " << host_ram_gb
                      << " GB\n";
    LOG_STREAM_INFO() << "[CPU-CONFIG] Pin: top-" << pin_topk_per_layer
                      << "/layer from '" << pin_file << "'\n";
    LOG_STREAM_INFO() << "[CPU-CONFIG] OMP max threads: "
                      << (omp_max_threads > 0 ? std::to_string(omp_max_threads)
                                              : "auto")
                      << "\n";
  }
};

#pragma once
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace dsv41_trace {
inline std::string &directory_storage() {
  static std::string value;
  return value;
}
inline int &selected_layer_storage() {
  static int value = -1;
  return value;
}
inline int &selected_pos_storage() {
  static int value = -1;
  return value;
}
inline void configure(const std::string &dir, int layer, int pos) {
  if (layer < -1 || pos < -1)
    throw std::invalid_argument("DeepSeek trace layer/position must be nonnegative");
  if (!dir.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec)
      throw std::runtime_error("cannot create DeepSeek trace directory '" + dir +
                               "': " + ec.message());
  }
  directory_storage() = dir;
  selected_layer_storage() = layer;
  selected_pos_storage() = pos;
}
inline const char *directory() {
  const auto &value = directory_storage();
  return value.empty() ? nullptr : value.c_str();
}
inline int selected_layer() { return selected_layer_storage(); }
inline int selected_pos() { return selected_pos_storage(); }
inline bool enabled(int layer, int pos) {
  return directory() && (selected_layer() < 0 || selected_layer() == layer) &&
         (selected_pos() < 0 || selected_pos() == pos);
}
inline void save(const char *name, int layer, int pos, const float *data,
                 size_t count) {
  const char *dir = directory();
  if (!dir || !data || count == 0 || !enabled(layer, pos)) return;
  char suffix[96];
  std::snprintf(suffix, sizeof(suffix), "/l%02d-p%d-%s.f32", layer, pos,
                name);
  std::ofstream out(std::string(dir) + suffix,
                    std::ios::binary | std::ios::trunc);
  if (out) out.write(reinterpret_cast<const char *>(data),
                     static_cast<std::streamsize>(count * sizeof(float)));
}
} // namespace dsv41_trace

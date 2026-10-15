#pragma once

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

namespace dsv41_trace {
inline const char *directory() {
  static const char *value = std::getenv("MOESTREAM_DEEPSEEK_V41_TRACE_DIR");
  return value && value[0] ? value : nullptr;
}
inline int selected_layer() {
  static const int value = [] {
    const char *s = std::getenv("MOESTREAM_DEEPSEEK_V41_TRACE_LAYER");
    return s && s[0] ? std::atoi(s) : -1;
  }();
  return value;
}
inline int selected_pos() {
  static const int value = [] {
    const char *s = std::getenv("MOESTREAM_DEEPSEEK_V41_TRACE_POS");
    return s && s[0] ? std::atoi(s) : -1;
  }();
  return value;
}
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

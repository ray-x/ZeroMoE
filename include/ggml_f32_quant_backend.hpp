#pragma once
#include <cstdint>
#if defined(MOESTREAM_USE_GGML_CUDA)
// Optional GGML CUDA extension. Reads existing quantized tensor pointers and
// F32 activations directly; never allocates or copies expert weights.
extern "C" bool moestream_install_f32_quant_graph_hook();
extern "C" uint64_t moestream_f32_quant_graph_launches();
#else
inline bool moestream_install_f32_quant_graph_hook() { return false; }
inline uint64_t moestream_f32_quant_graph_launches() { return 0; }
#endif

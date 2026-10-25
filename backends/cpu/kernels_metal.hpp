#pragma once

// Apple Metal / MPS dispatch layer for the CPU-only backend.
//
// This header exposes a thin wrapper around Metal Performance Shaders (MPS).
// It is only meaningful on Apple platforms; on other platforms the entire
// file is a no-op.
//
// Design contract:
//   * Every function returns false on any failure.  Callers MUST fall back
//     to the NEON (or scalar) CPU path when that happens.
//   * Functions are synchronous: the result is in CPU memory when they return.
//   * Initialization is lazy and thread-safe (C++11 static-local init).
//
// ANE note: MPS targets the Apple GPU.  The Apple Neural Engine (ANE) is
// primarily accessible through Core ML, not through a direct public API.
// Metal may opportunistically schedule some MPS work on ANE-capable hardware
// at the OS's discretion, but this layer does not guarantee ANE usage.
// A future Core ML backend would be required for explicit ANE dispatch.

#ifdef __APPLE__

namespace zeromoe_metal {

// Initialize the Metal device and command queue.  Safe to call multiple times.
bool init();

// Return true if a Metal device is available and initialized.
bool available();

// Synchronous F32 matmul using MPSMatrixMultiplication:
//   y[S,O] = x[S,I] * W^T, where W is [O,I] row-major.
// Returns false on failure (caller should fall back to NEON/CPU).
bool matmul_f32(float *y, const float *x, const float *W, int S, int I, int O);

} // namespace zeromoe_metal

#endif // __APPLE__

// src/kernels_metal.mm — Apple Metal Performance Shaders dispatch.
//
// Compiled as Objective-C++ only when ZEROMOE_METAL=ON on Apple Silicon.
// Uses ARC (-fobjc-arc).  All entry points are synchronous and return false
// on failure so the caller can fall back to NEON.

#include "kernels_metal.hpp"

#ifdef __APPLE__

#include "logger.hpp"

#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <atomic>
#include <cstring>
#include <mutex>

namespace zeromoe_metal {

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static std::mutex g_mutex;

// Retry init once per process.  We do not want to keep probing if the device
// is unavailable.
static std::atomic<bool> g_init_attempted{false};

bool init() {
  @autoreleasepool {
    if (g_device)
      return true;
    if (g_init_attempted.exchange(true))
      return g_device != nil;

    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device)
      return false;

    g_queue = [g_device newCommandQueue];
    if (!g_queue) {
      g_device = nil;
      return false;
    }
    LOG_INFO("[METAL] MPS matmul dispatch initialized on '%s'\n",
             [[g_device name] UTF8String]);
    return true;
  }
}

bool available() {
  static bool ok = init();
  return ok;
}

bool matmul_f32(float *y, const float *x, const float *W, int S, int I, int O) {
  if (!available())
    return false;
  if (S <= 0 || I <= 0 || O <= 0)
    return false;

  // Tiny matmuls are dominated by GPU dispatch overhead; the NEON path is
  // faster for these.  The threshold is a heuristic (256K FLOPs).
  const int64_t flop = static_cast<int64_t>(S) * I * O;
  if (flop < 256 * 1024)
    return false;

  // Serialize submission for now.  Creating multiple command queues for
  // concurrent submission is a future optimization.
  std::lock_guard<std::mutex> lock(g_mutex);

  @autoreleasepool {
    const size_t x_bytes = static_cast<size_t>(S) * I * sizeof(float);
    const size_t w_bytes = static_cast<size_t>(O) * I * sizeof(float);
    const size_t y_bytes = static_cast<size_t>(S) * O * sizeof(float);

    id<MTLBuffer> x_buf =
        [g_device newBufferWithBytes:x
                              length:x_bytes
                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> w_buf =
        [g_device newBufferWithBytes:W
                              length:w_bytes
                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> y_buf =
        [g_device newBufferWithLength:y_bytes
                              options:MTLResourceStorageModeShared];
    if (!x_buf || !w_buf || !y_buf)
      return false;

    MPSMatrixDescriptor *descA = [MPSMatrixDescriptor
        matrixDescriptorWithRows:static_cast<NSUInteger>(S)
                         columns:static_cast<NSUInteger>(I)
                        rowBytes:static_cast<NSUInteger>(I * sizeof(float))
                        dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descB = [MPSMatrixDescriptor
        matrixDescriptorWithRows:static_cast<NSUInteger>(O)
                         columns:static_cast<NSUInteger>(I)
                        rowBytes:static_cast<NSUInteger>(I * sizeof(float))
                        dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descC = [MPSMatrixDescriptor
        matrixDescriptorWithRows:static_cast<NSUInteger>(S)
                         columns:static_cast<NSUInteger>(O)
                        rowBytes:static_cast<NSUInteger>(O * sizeof(float))
                        dataType:MPSDataTypeFloat32];
    if (!descA || !descB || !descC)
      return false;

    MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:x_buf descriptor:descA];
    MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:w_buf descriptor:descB];
    MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:y_buf descriptor:descC];
    if (!matA || !matB || !matC)
      return false;

    MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc]
         initWithDevice:g_device
          transposeLeft:NO
         transposeRight:YES
             resultRows:static_cast<NSUInteger>(S)
          resultColumns:static_cast<NSUInteger>(O)
        interiorColumns:static_cast<NSUInteger>(I)
                  alpha:1.0
                   beta:0.0];
    if (!mm)
      return false;

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    if (!cmd)
      return false;

    [mm encodeToCommandBuffer:cmd
                   leftMatrix:matA
                  rightMatrix:matB
                 resultMatrix:matC];
    [cmd commit];
    [cmd waitUntilCompleted];

    if (cmd.error)
      return false;

    std::memcpy(y, y_buf.contents, y_bytes);
    return true;
  }
}

} // namespace zeromoe_metal

#endif // __APPLE__

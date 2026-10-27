#pragma once

// Scoped NVTX range for nsys timelines. A no-op unless a profiler is attached (the NVTX calls
// resolve to stubs), so the ranges stay in release builds.
//   ZE_NVTX("mtp.draft");   // range lasts until the end of the enclosing scope
#if defined(ZEROMOE_USE_GGML_CUDA)
#include <nvtx3/nvToolsExt.h>
struct ZeNvtxRange {
  explicit ZeNvtxRange(const char *name) { nvtxRangePushA(name); }
  ~ZeNvtxRange() { nvtxRangePop(); }
  ZeNvtxRange(const ZeNvtxRange &) = delete;
  ZeNvtxRange &operator=(const ZeNvtxRange &) = delete;
};
#define ZE_NVTX_CAT2(a, b) a##b
#define ZE_NVTX_CAT(a, b) ZE_NVTX_CAT2(a, b)
#define ZE_NVTX(name) ZeNvtxRange ZE_NVTX_CAT(ze_nvtx_, __LINE__)(name)
#else
#define ZE_NVTX(name) ((void)0)
#endif

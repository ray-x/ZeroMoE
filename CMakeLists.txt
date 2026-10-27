cmake_minimum_required(VERSION 3.25)
project(ZeroMoE CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ZEROMOE_WRITE_BUILD_PROVENANCE
       "Write a source/binary provenance manifest after building zeromoe" ON)

# Resolver/default/binding tests need neither CUDA nor model weights.
enable_testing()
add_executable(graph_settings_test tests/test_graph_settings.cpp)
target_include_directories(graph_settings_test PRIVATE core models/graph backends/cpu)
add_test(NAME graph_settings COMMAND graph_settings_test
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/graph_settings_defaults.snapshot")
add_executable(graph_settings_binding_test tests/test_graph_settings_binding.cpp)
target_include_directories(graph_settings_binding_test PRIVATE core models/graph)
find_package(Threads REQUIRED)
target_link_libraries(graph_settings_binding_test PRIVATE Threads::Threads)
add_test(NAME graph_settings_binding COMMAND graph_settings_binding_test)
set_tests_properties(graph_settings_binding PROPERTIES ENVIRONMENT_MODIFICATION
    "ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_LAYERS=unset:;ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_TRAILING=unset:;ZEROMOE_PHASE_CACHE_MIB=unset:")

add_executable(mtp_graph_contract_test EXCLUDE_FROM_ALL
    tests/test_mtp_graph_contract.cpp)
target_include_directories(mtp_graph_contract_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/models/graph)

add_executable(mtp_speculation_test EXCLUDE_FROM_ALL
    tests/test_mtp_speculation.cpp)
target_include_directories(mtp_speculation_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/models/graph)

# ---------------------------------------------------------------------------
# Build options
# ---------------------------------------------------------------------------
option(ZEROMOE_CPU_ONLY "Disable CUDA and build CPU-only components" OFF)
option(ZEROMOE_USE_GGML_CUDA
       "Build/link ggml-cuda for the experimental MUL_MAT_ID prefill backend"
       OFF)
# ggml's own default for this is OFF outside llama.cpp (its option() comment
# literally says "llama.cpp only") -- a plain ZEROMOE_USE_GGML_CUDA=ON build
# silently never defines GGML_CUDA_USE_GRAPHS, so ggml-cuda's CUDA graph
# capture/replay path (ggml_backend_cuda_graph_compute in ggml-cuda.cu) is
# compiled out entirely: every decode-graph node gets its own individual
# cudaLaunchKernel instead of one batched cudaGraphLaunch per captured split.
# Measured real-world cost of missing this on the full-graph Qwen4Exp decode
# path: full-graph decode dropped from a verified ~20 tok/s (graphs enabled)
# to ~8 tok/s (graphs silently absent) on identical hardware/model/config --
# this was mistaken for a correctness or scheduling regression for an entire
# investigation before the missing compile define was found. Default ON here
# so any ZEROMOE_USE_GGML_CUDA build gets it without the caller having to
# know this ggml-specific gotcha; still overridable with -DGGML_CUDA_GRAPHS=OFF.
option(ZEROMOE_ENABLE_CUDA_GRAPHS
       "Enable ggml CUDA graph capture/replay (GGML_CUDA_GRAPHS) for full-graph decode/prefill"
       ON)
option(ZEROMOE_ENABLE_AVX512_VNNI
       "Compile ZeroMoE and ggml CPU kernels with AVX-512 VNNI (set OFF on CPUs without VNNI)"
       ON)
option(ZEROMOE_GGML_OPENMP
       "Build ggml's CPU backend with OpenMP (OFF uses its persistent threadpool)"
       ON)
option(ZEROMOE_ENABLE_GGML_HOST_NODE_PATCHES
       "Apply GGML CUDA/CPUOffload host-node integration patches"
       ON)
option(ZEROMOE_GGML_DEBUG_PATCHES
       "Also apply the debug/profiling ggml patches in patches/debug (not part of the main series)"
       OFF)
option(ZEROMOE_METAL "Enable Apple Metal GPU backend (requires ggml)" OFF)

# Resolve the architecture list before add_subdirectory(ggml): ggml-cuda
# creates its CUDA targets during that call, so detecting architectures later
# would affect ZeroMoE's kernels but leave ggml-cuda with a different target.
if(NOT ZEROMOE_CPU_ONLY AND
   (NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR
    CMAKE_CUDA_ARCHITECTURES STREQUAL "" OR
    CMAKE_CUDA_ARCHITECTURES STREQUAL "native"))
    execute_process(
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/scripts/detect_cuda_arch.sh"
        RESULT_VARIABLE _zeromoe_detect_arch_result
        OUTPUT_VARIABLE _zeromoe_detected_archs
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_VARIABLE _zeromoe_detect_arch_error)
    if(_zeromoe_detect_arch_result EQUAL 0 AND _zeromoe_detected_archs)
        set(CMAKE_CUDA_ARCHITECTURES "${_zeromoe_detected_archs}")
        message(STATUS "ZeroMoE auto-detected CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    else()
        set(CMAKE_CUDA_ARCHITECTURES native)
        message(STATUS "ZeroMoE CUDA architecture detection unavailable; using CMake native")
        if(_zeromoe_detect_arch_error)
            string(STRIP "${_zeromoe_detect_arch_error}" _zeromoe_detect_arch_error)
            message(STATUS "ZeroMoE CUDA architecture detection detail: ${_zeromoe_detect_arch_error}")
        endif()
    endif()
endif()

# On macOS the CPU-only binary is the only usable engine, and real MoE models
# are almost always quantized.
if(APPLE)
    set(ZEROMOE_CPU_ONLY ON CACHE BOOL "macOS has no CUDA SDK; force CPU-only build" FORCE)
    message(STATUS "ZeroMoE: Apple platform detected — forcing ZEROMOE_CPU_ONLY=ON")
endif()

# ZeroMoE always links ggml for reference dequantizers. The default fetches a
# stable llama.cpp release and builds it in-tree; point ZEROMOE_GGML_ROOT or
# ZEROMOE_GGML_PREBUILT_DIR at an existing tree to reuse it instead.

# Detect Apple Silicon (M1/M2/M3/M4/M5 all share arm64; M3+ specific flags are
# applied below when the compiler supports them).
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    set(ZEROMOE_APPLE_SILICON ON)
    message(STATUS "ZeroMoE: Apple Silicon (arm64) detected")
else()
    set(ZEROMOE_APPLE_SILICON OFF)
endif()

# ---------------------------------------------------------------------------
# AVX-512/F16C compile flags (x86 only). kernels_cpu.cpp, attention_layer.cpp,
# and execution_layer.cpp use these intrinsics unconditionally with no
# __attribute__((target(...))) of their own, so the compiling TARGET must
# supply matching flags or GCC either silently changes the calling
# convention for __m512 params/returns (-Wpsabi) or, on stricter versions,
# hard-errors on always_inline intrinsics (_cvtsh_ss, _mm512_storeu_ps,
# _mm512_xor_ps, ...) refusing to inline into a function that isn't itself
# compiled with a matching target -- exactly the "target specific option
# mismatch" build failure this was added to fix. F/BW/VL alone turned out
# insufficient: _mm512_xor_ps and _mm512_insertf32x8/_mm512_inserti32x8
# (kernels_cpu.cpp's IQ2/IQ3 sign-unpacking) are AVX512DQ, not F/BW/VL, and
# only surfaced as a SEPARATE build failure one intrinsic at a time. Rather
# than continue that one-header-at-a-time discovery, this uses the
# standard "Skylake-X baseline" bundle (F+CD+BW+DQ+VL) -- the full set
# every mainstream AVX-512 CPU since Skylake-X ships together as one unit,
# unlike the genuinely-later, genuinely-optional extensions (VNNI, VBMI,
# ...) that arrived piecemeal in Cascade Lake/Ice Lake+ and aren't safe to
# assume. Checked as one combined flag string (not assumed) so a
# non-AVX-512 x86 host still configures cleanly; CLAUDE.md documents
# AVX-512 as a hard requirement for these files, so
# ZEROMOE_AVX512_FLAGS staying empty here just means the build fails
# later at these exact source files with a clearer compiler error instead
# of a silent miscompile. Deliberately NOT -mavx512vnni: the source
# already compile-time-gates its VNNI fast path (#if
# defined(__AVX512VNNI__)).  Keep VNNI opt-in because forcing that flag here
# would risk a SIGILL at runtime on a real AVX-512 CPU that lacks VNNI
# specifically (some early Skylake-X/Cascade Lake parts).
set(ZEROMOE_AVX512_FLAGS "")
if(NOT ZEROMOE_APPLE_SILICON)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl -mf16c" ZEROMOE_HAS_AVX512_FLAGS)
    if(ZEROMOE_HAS_AVX512_FLAGS)
        set(ZEROMOE_AVX512_FLAGS -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl -mf16c)
    else()
        message(WARNING "ZeroMoE: compiler does not accept AVX-512/F16C flags -- "
                         "kernels_cpu.cpp/attention_layer.cpp/execution_layer.cpp require "
                         "AVX-512 (see CLAUDE.md) and may fail to build without them")
    endif()
    if(ZEROMOE_ENABLE_AVX512_VNNI)
        check_cxx_compiler_flag("-mavx512vnni" ZEROMOE_HAS_AVX512_VNNI_FLAG)
        if(NOT ZEROMOE_HAS_AVX512_VNNI_FLAG)
            message(FATAL_ERROR
                    "ZeroMoE: ZEROMOE_ENABLE_AVX512_VNNI=ON, but the compiler "
                    "does not accept -mavx512vnni")
        endif()
        list(APPEND ZEROMOE_AVX512_FLAGS -mavx512vnni)
    endif()
endif()

# ---------------------------------------------------------------------------
# OpenMP (used by CPU kernels and some reference loops)
# ---------------------------------------------------------------------------
find_package(OpenMP)

# ---------------------------------------------------------------------------
# ggml integration
# ---------------------------------------------------------------------------
# Default to a stable, verified ggml tag in the ggerganov/ggml repo.
# (The old "b3923" string was not a valid tag in either ggml or llama.cpp.)
set(ZEROMOE_GGML_COMMIT "8baabd53b77d1c48c6ef8d3b620436292f3f6231" CACHE STRING "llama.cpp commit whose ggml/ directory is built (used when the third_party/llama.cpp submodule is absent)")
set(ZEROMOE_GGML_ROOT "" CACHE PATH "Path to a local ggml checkout (fetched automatically if empty)")

# Provide a deprecated alias for configs that still set it.
if(ZEROMOE_LLAMA_CPP_ROOT AND NOT ZEROMOE_GGML_ROOT)
    set(ZEROMOE_GGML_ROOT "${ZEROMOE_LLAMA_CPP_ROOT}" CACHE PATH "" FORCE)
    message(WARNING "ZEROMOE_LLAMA_CPP_ROOT is deprecated; use ZEROMOE_GGML_ROOT")
endif()

# Disable everything we do not need for a CPU decode assistant, except the
# explicit experimental CUDA backend option. GGML_CUDA must be set BEFORE
# the subproject is configured.
set(GGML_CUDA        ${ZEROMOE_USE_GGML_CUDA} CACHE BOOL "" FORCE)
set(GGML_CUDA_GRAPHS ${ZEROMOE_ENABLE_CUDA_GRAPHS} CACHE BOOL "" FORCE)
# Loud, impossible-to-miss configure-time confirmation of the exact
# setting that silently regressed full-graph decode from ~20 tok/s to
# ~8 tok/s once already (see ZEROMOE_ENABLE_CUDA_GRAPHS's own comment
# above). A default is not a guarantee -- print the resolved value on
# every configure, and fail outright if CUDA is on but graphs got
# turned off without an explicit, informed override.
if(ZEROMOE_USE_GGML_CUDA AND NOT ZEROMOE_ENABLE_CUDA_GRAPHS)
    message(WARNING
        "ZEROMOE_USE_GGML_CUDA=ON but ZEROMOE_ENABLE_CUDA_GRAPHS=OFF: "
        "full-graph decode will silently run WITHOUT CUDA graph capture/"
        "replay (every node becomes an individual cudaLaunchKernel). "
        "This previously cost ~20 tok/s -> ~8 tok/s and was mistaken for "
        "a correctness/scheduling bug for an entire investigation. Only "
        "disable this deliberately for a real A/B; the runtime "
        "[CUDA-GRAPHS-CHECK] log line will also confirm this at every "
        "process start regardless of this setting.")
else()
    message(STATUS "ZeroMoE: GGML_CUDA_GRAPHS=${GGML_CUDA_GRAPHS} "
                   "(verified again at runtime via [CUDA-GRAPHS-CHECK])")
endif()
set(GGML_HIPBLAS     OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN      OFF CACHE BOOL "" FORCE)
set(GGML_SYCL        OFF CACHE BOOL "" FORCE)
if(ZEROMOE_GGML_OPENMP AND NOT OpenMP_CXX_FOUND)
    message(WARNING "ZEROMOE_GGML_OPENMP=ON requested, but OpenMP was not found; ggml CPU backend will use its native threadpool")
endif()
set(GGML_OPENMP      ${ZEROMOE_GGML_OPENMP} CACHE BOOL "" FORCE)
message(STATUS "ZeroMoE: GGML_OPENMP=${GGML_OPENMP} (ZEROMOE_GGML_OPENMP=${ZEROMOE_GGML_OPENMP})")
# Keep ggml's CPU backend on the same ISA choice as ZeroMoE's CPU
# kernels.  This is the single opt-out for older CPUs: setting
# ZEROMOE_ENABLE_AVX512_VNNI=OFF also disables VNNI in ggml, including
# when reconfiguring a build directory that previously had it enabled.
set(GGML_AVX512_VNNI ${ZEROMOE_ENABLE_AVX512_VNNI} CACHE BOOL "" FORCE)
if(ZEROMOE_ENABLE_AVX512_VNNI)
    set(GGML_AVX512       ON CACHE BOOL "" FORCE)
endif()
message(STATUS
    "ZeroMoE: AVX-512 VNNI=${ZEROMOE_ENABLE_AVX512_VNNI} (ZeroMoE and ggml CPU kernels)")

if(ZEROMOE_METAL AND APPLE)
    # ggml's Metal backend requires the offline shader compiler (metal,
    # metallib), which Apple only ships inside the full Xcode.app.  Our
    # own MPS dispatch uses the runtime Metal Performance Shaders API and
    # does NOT need those tools.  If the compiler is missing, keep our MPS
    # path alive but turn off ggml Metal to avoid a configure/build failure.
    find_program(_ZEROMOE_METALC metal)
    find_program(_ZEROMOE_METALLIB metallib)
    if(_ZEROMOE_METALC AND _ZEROMOE_METALLIB)
        set(GGML_METAL ON CACHE BOOL "" FORCE)
        message(STATUS "ZeroMoE: ggml Metal backend enabled (metal/metallib found)")
    else()
        set(GGML_METAL OFF CACHE BOOL "" FORCE)
        message(STATUS "ZeroMoE: ggml Metal backend disabled (full Xcode not installed); "
                        "custom MPS dispatch still enabled")
    endif()
else()
    set(GGML_METAL OFF CACHE BOOL "" FORCE)
endif()

if(NOT ZEROMOE_GGML_ROOT)
    # Default: ggml/ from the pinned llama.cpp submodule (third_party/llama.cpp, commit ZEROMOE_GGML_COMMIT).
    # It is copied into the build tree before the patch series is applied, so the submodule is never modified.
    set(_zeromoe_ggml_src "${CMAKE_BINARY_DIR}/_deps/ggml-src")
    set(_zeromoe_ggml_submodule "${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml")
    if(NOT EXISTS "${_zeromoe_ggml_src}/CMakeLists.txt")
        if(EXISTS "${_zeromoe_ggml_submodule}/CMakeLists.txt")
            message(STATUS "ZeroMoE: copying ggml from third_party/llama.cpp/ggml into the build tree")
            file(COPY "${_zeromoe_ggml_submodule}/" DESTINATION "${_zeromoe_ggml_src}"
                 PATTERN ".git" EXCLUDE)
        else()
            # Submodule not initialised: fetch the same commit into the build tree.
            message(STATUS "ZeroMoE: third_party/llama.cpp missing; fetching ${ZEROMOE_GGML_COMMIT} ...")
            set(_zeromoe_fetch "${CMAKE_BINARY_DIR}/_deps/llama-fetch")
            file(MAKE_DIRECTORY "${_zeromoe_fetch}")
            file(REMOVE_RECURSE "${_zeromoe_fetch}")
            file(MAKE_DIRECTORY "${_zeromoe_fetch}")
            # Shallow, blob-less, sparse: only ggml/ is downloaded (about 5 MB, not the 390 MB whole tree).
            foreach(_cmd "init -q"
                          "remote add origin https://github.com/ggml-org/llama.cpp"
                          "config extensions.partialClone origin"
                          "config remote.origin.promisor true"
                          "config remote.origin.partialclonefilter blob:none"
                          "sparse-checkout init --no-cone"
                          "sparse-checkout set /ggml/"
                          "fetch -q --depth 1 --filter=blob:none origin ${ZEROMOE_GGML_COMMIT}"
                          "checkout -q FETCH_HEAD")
                separate_arguments(_args UNIX_COMMAND "${_cmd}")
                execute_process(COMMAND git ${_args} WORKING_DIRECTORY "${_zeromoe_fetch}"
                    RESULT_VARIABLE _rc ERROR_VARIABLE _err OUTPUT_QUIET)
                if(NOT _rc EQUAL 0)
                    message(FATAL_ERROR
                        "ggml source not found: third_party/llama.cpp is empty and the fetch failed (${_err}). "
                        "Run: scripts/init_ggml.sh (fetches only ggml/, about 5 MB), or set ZEROMOE_GGML_ROOT.")
                endif()
            endforeach()
            file(COPY "${_zeromoe_fetch}/ggml/" DESTINATION "${_zeromoe_ggml_src}")
            endif()
        endif()
    set(ZEROMOE_GGML_ROOT "${_zeromoe_ggml_src}")
endif()

# Apply ZeroMoE's ggml patch series (patches/README.md).  Idempotent:
# already-applied patches are detected by a reverse apply check.
# Link an already-built ggml instead of compiling it in-tree. The prebuilt tree must come
# from the same ZEROMOE_GGML_ROOT (its headers are used) and ggml patches are NOT re-applied.
set(ZEROMOE_GGML_PREBUILT_DIR "" CACHE PATH "Directory holding prebuilt libggml*.so (skips building ggml)")
if(ZEROMOE_GGML_PREBUILT_DIR)
    foreach(_lib ggml ggml-base ggml-cpu)
        add_library(${_lib} SHARED IMPORTED GLOBAL)
        set_target_properties(${_lib} PROPERTIES
            IMPORTED_LOCATION "${ZEROMOE_GGML_PREBUILT_DIR}/src/lib${_lib}.so"
            INTERFACE_INCLUDE_DIRECTORIES "${ZEROMOE_GGML_ROOT}/include")
    endforeach()
    set_property(TARGET ggml PROPERTY INTERFACE_LINK_LIBRARIES "ggml-cpu;ggml-base")
    set_property(TARGET ggml-cpu PROPERTY INTERFACE_LINK_LIBRARIES "ggml-base")
    add_library(ggml-cuda SHARED IMPORTED GLOBAL)
    set_target_properties(ggml-cuda PROPERTIES
        IMPORTED_LOCATION "${ZEROMOE_GGML_PREBUILT_DIR}/src/ggml-cuda/libggml-cuda.so"
        INTERFACE_INCLUDE_DIRECTORIES "${ZEROMOE_GGML_ROOT}/include"
        INTERFACE_LINK_LIBRARIES "ggml-base")
    message(STATUS "ZeroMoE: linking PREBUILT ggml from ${ZEROMOE_GGML_PREBUILT_DIR} (no ggml compile)")
else()
    include(cmake/ZeroMoEGgmlPatches.cmake)
    zeromoe_apply_ggml_patches("${ZEROMOE_GGML_ROOT}")
    # The common build-tree copy has no .git directory. Without an
    # explicit revision, ggml's version probe can walk up to ZeroMoE's
    # repository and stamp the application commit as GGML_COMMIT.
    set(_zeromoe_ggml_build_commit "")
    if(EXISTS "${ZEROMOE_GGML_ROOT}/.git")
        execute_process(
            COMMAND git -C "${ZEROMOE_GGML_ROOT}" rev-parse --short HEAD
            OUTPUT_VARIABLE _zeromoe_ggml_build_commit
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
    endif()
    if(NOT _zeromoe_ggml_build_commit AND ZEROMOE_GGML_COMMIT)
        set(_zeromoe_ggml_build_commit "${ZEROMOE_GGML_COMMIT}")
    endif()
    if(NOT _zeromoe_ggml_build_commit)
        set(_zeromoe_ggml_build_commit "unknown")
    endif()
    set(GGML_BUILD_COMMIT "${_zeromoe_ggml_build_commit}")
    message(STATUS "ZeroMoE: GGML source revision=${GGML_BUILD_COMMIT}")
    add_subdirectory("${ZEROMOE_GGML_ROOT}" ggml-build)
endif()
message(STATUS "ZeroMoE: using ggml at ${ZEROMOE_GGML_ROOT}")

# ggml-quants.h lives in ggml/src/, not in the public include/ directory.
# Expose it to our bridge translation unit.
set(ZEROMOE_GGML_SRC_DIR "${ZEROMOE_GGML_ROOT}/src")

# ---------------------------------------------------------------------------
# Common include path
# ---------------------------------------------------------------------------
# Source layout (core/ memory/ models/* backends/* kernels/*): every header directory is on
# the include path, so sources keep flat `#include "name.hpp"` lines and a file can move
# between these directories without touching its includes. File names are unique across
# them (checked by layout_map.py).
set(ZEROMOE_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/memory
    ${CMAKE_CURRENT_SOURCE_DIR}/models/base
    ${CMAKE_CURRENT_SOURCE_DIR}/models/graph
    ${CMAKE_CURRENT_SOURCE_DIR}/models/qwen
    ${CMAKE_CURRENT_SOURCE_DIR}/models/glm
    ${CMAKE_CURRENT_SOURCE_DIR}/models/mimo
    ${CMAKE_CURRENT_SOURCE_DIR}/models/deepseek
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/cpu
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/ggml
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/exl3)
include_directories(${ZEROMOE_INCLUDE_DIRS})

# ---------------------------------------------------------------------------
# Full engine build (Linux + CUDA Toolkit)
# ---------------------------------------------------------------------------
if(NOT ZEROMOE_CPU_ONLY)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    find_package(CUDAToolkit REQUIRED)
    if(ZEROMOE_WRITE_BUILD_PROVENANCE)
        find_package(Python3 COMPONENTS Interpreter REQUIRED)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tools/build_provenance.py"
                    snapshot --repo "${CMAKE_CURRENT_SOURCE_DIR}"
            RESULT_VARIABLE _zeromoe_snapshot_result
            OUTPUT_VARIABLE _zeromoe_source_snapshot
            ERROR_VARIABLE _zeromoe_snapshot_error
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(NOT _zeromoe_snapshot_result EQUAL 0)
            message(FATAL_ERROR "Could not snapshot source for build provenance: ${_zeromoe_snapshot_error}")
        endif()
        string(JSON ZEROMOE_CONFIGURED_SOURCE_COMMIT GET "${_zeromoe_source_snapshot}" source_commit)
        string(JSON ZEROMOE_CONFIGURED_SOURCE_SNAPSHOT_SHA256 GET "${_zeromoe_source_snapshot}" source_snapshot_sha256)
    endif()

    # Engine executable
    # CUDA kernel sources (cuda/*.cu) and the GPU Qwen MoE backend
    # (models/qwen/qwen_moe.cpp, "compiled into the main zeromoe binary (CUDA
    # builds only)" per its own header comment) were not referenced by
    # ANY target here -- zeromoe/zeromoe_tests linked without them and
    # failed at link time with undefined references to every launch_*
    # kernel wrapper (cuda_manager.hpp) and QwenMoEBackend's vtable. Shared
    # between both executables the same way models/base/model_factory.cpp
    # already is, since both need the identical GPU dispatch surface.
    set(ZEROMOE_CUDA_SRCS
        backends/exl3/exl3_aux.cu
        kernels/math/attn_score_softmax.cu
        kernels/models/dsv4_score_softmax.cu
        kernels/quant/dequant_bf16.cu
        kernels/math/expert_accum.cu
        kernels/math/expert_activation.cu
        kernels/math/gdn_conv1d.cu
        kernels/models/glm5next_kda.cu
        kernels/models/glm5next_router.cu
        kernels/math/ggml_moe_pack.cu
        kernels/quant/iq1_dequant.cu
        kernels/quant/iq2_matmul.cu
        kernels/quant/iq3_matmul.cu
        kernels/quant/iq4xs_matmul.cu
        kernels/quant/kquant_matmul.cu
        kernels/math/matmul.cu
        kernels/math/mla_attention.cu
        kernels/quant/mxfp4_matmul.cu
        kernels/quant/q5_1_matmul.cu
        kernels/models/qwen4exp_hc_q8.cu
        kernels/models/qwen_stage_kernels.cu
        kernels/quant/q8_0_matmul.cu
        kernels/quant/q2_0_matmul.cu
        kernels/math/rmsnorm.cu
        kernels/quant/wuv_int8_dequant.cu
    )
    # The Layer-13 predictive staging spike is an optional worktree feature.
    # Keep a master checkout buildable when its files are absent, while still
    # compiling them automatically on the preserved spike branch.
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cuda/qwen4exp_layer13_staging.cu")
        list(APPEND ZEROMOE_CUDA_SRCS cuda/qwen4exp_layer13_staging.cu)
    endif()

    # Upstream ExLlamaV3 EXL3 kernels behind the launch_exl3_mul1_* API.
    # Runtime opt-out: ZEROMOE_EXL3_UPSTREAM=0 (legacy kernels).
    include(cmake/ZeroMoEExl3Upstream.cmake)
    if(TARGET zeromoe_exl3_upstream)
        list(APPEND ZEROMOE_CUDA_SRCS backends/exl3/exl3_upstream_ops.cu)
        set_source_files_properties(backends/exl3/exl3_upstream_ops.cu PROPERTIES
            INCLUDE_DIRECTORIES "${ZEROMOE_EXLLAMAV3_SRC}")
        set_source_files_properties(backends/exl3/exl3_aux.cu backends/exl3/exl3_gpu.cpp
            backends/exl3/exl3_ggml_ffn.cpp backends/exl3/exl3_upstream_ops.cu
            PROPERTIES COMPILE_DEFINITIONS ZEROMOE_EXL3_UPSTREAM=1)
        set(ZEROMOE_EXL3_UPSTREAM_LIB zeromoe_exl3_upstream)
        set(ZEROMOE_EXL3_UPSTREAM_OPS backends/exl3/exl3_upstream_ops.cu)
    endif()

    set(ZEROMOE_OPTIONAL_SRCS)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/qwen4exp_layer13_staging.cpp")
        list(APPEND ZEROMOE_OPTIONAL_SRCS src/qwen4exp_layer13_staging.cpp)
    endif()

    add_executable(zeromoe
        core/main.cpp
        core/engine_runner.cpp
        core/engine_runner_model.cpp
        core/engine_runner_cache.cpp
        core/engine_runner_dense.cpp
        core/engine_runner_ram.cpp
        core/engine_runner_preload.cpp
        core/engine_runner_session.cpp
        core/engine_runner_api.cpp
        core/perf_report_summary.cpp
        core/perf_report_dump_a.cpp
        core/perf_report_dump_b.cpp
        core/perf_report_stats.cpp
        core/execution_layer.cpp
        backends/exl3/exl3_cpu.cpp
        backends/exl3/exl3_model_loader.cpp
        backends/exl3/exl3_decode.cpp
        backends/exl3/exl3_gpu.cpp
        backends/cpu/cpu_dense.cpp
        backends/cpu/cpu_engine.cpp
        backends/cpu/cpu_forward.cpp
        backends/cpu/cpu_model_backend.cpp
        models/qwen/cpu_qwen_moe.cpp
        models/qwen/cpu_qwen4exp.cpp
        memory/scratch_buffers.cpp
        backends/cuda/cuda_kernel_dispatcher.cpp
        core/gpu_expert_executor.cpp
        core/cpu_expert_executor.cpp
        core/moe_router.cpp
        core/moe_pipeline.cpp
        core/moe_pipeline_orchestrator.cpp
        core/moe_router_service.cpp
        core/prefill_warmup_manager.cpp
        core/batched_dispatch_pipeline.cpp
        core/token_dispatch_pipeline.cpp
        core/shared_expert_executor.cpp
        ${ZEROMOE_OPTIONAL_SRCS}
        core/attention_layer.cpp
        models/base/graph_initializer.cpp
        core/hyper_connections.cpp
        models/qwen/ple_ngram.cpp
        models/base/safetensors_adapter.cpp
        models/deepseek/deepseek41_engram.cpp
        core/gpu_topology.cpp
        backends/cpu/ggml_dequant.cpp
        backends/cpu/ggml_grids.cpp
        backends/ggml/ggml_moe_backend.cpp
        models/graph/ggml_moe_graph_builder.cpp
        models/graph/mtp_graph_runtime.cpp
        models/qwen/qwen4exp_mtp_graph.cpp
        backends/ggml/ggml_moe_transfer_manager.cpp
        backends/ggml/ggml_moe_runner.cpp
        backends/ggml/ggml_router_runner.cpp
        models/qwen/ggml_qwen_shared_runner.cpp
        models/qwen/qwen_cpu_offload_ring.cpp
        models/qwen/ggml_qwen_decode_scheduler.cpp
        models/qwen/ggml_qwen_cpu_moe.cpp
        models/qwen/ggml_qwen4exp_graph.cpp
        models/qwen/ggml_qwen4exp_graph_arena.cpp
        models/qwen/ggml_qwen4exp_graph_build_blocks.cpp
        models/qwen/ggml_qwen4exp_graph_build.cpp
        models/qwen/ggml_qwen4exp_graph_build_moe.cpp
        models/qwen/ggml_qwen4exp_graph_decode_prepare.cpp
        models/qwen/ggml_qwen4exp_graph_diag.cpp
        models/qwen/ggml_qwen4exp_graph_checkpoint.cpp
        models/qwen/ggml_qwen4exp_graph_expert_bind.cpp
        models/qwen/ggml_qwen4exp_graph_host_node.cpp
        models/qwen/ggml_qwen4exp_graph_placement.cpp
        models/qwen/ggml_qwen4exp_graph_run.cpp
        models/qwen/ggml_qwen4exp_graph_window.cpp
        models/qwen/ggml_qwen4exp_graph_settings.cpp
        models/glm/ggml_glm5next_graph.cpp
        models/glm/ggml_glm5next_graph_arena.cpp
        models/glm/ggml_glm5next_graph_build.cpp
        models/glm/ggml_glm5next_graph_build_blocks.cpp
        models/glm/ggml_glm5next_graph_decode.cpp
        models/glm/ggml_bailingmoe3_graph.cpp
        models/glm/ggml_glm5next_graph_offload.cpp
        models/glm/ggml_glm5next_graph_partial.cpp
        models/glm/ggml_glm5next_graph_run.cpp
        models/glm/ggml_glm5next_graph_settings.cpp
        models/glm/ggml_glm5next_graph_stage.cpp
        models/glm/ggml_glm5next_graph_weights.cpp
        models/graph/ggml_graph_vram_headroom.cpp
        models/deepseek/ggml_deepseek_v41_graph.cpp
        models/deepseek/ggml_deepseek_v41_graph_decode.cpp
        backends/ggml/ggml_hc_backend.cpp
        backends/ggml/ggml_gdn_proj_backend.cpp
        models/glm/ggml_glm5next_proj_backend.cpp
        core/mtp_speculative.cpp
        core/qwen4exp_graph_mtp.cpp
        backends/cpu/kernels_cpu.cpp
        models/qwen/qwen_moe.cpp
        models/base/model_factory.cpp
        models/base/family_ops.cpp
        models/mimo/ggml_mimo2_graph.cpp
        models/graph/full_graph_models.cpp
        ${ZEROMOE_CUDA_SRCS}
    )
    target_include_directories(zeromoe PRIVATE
        ${ZEROMOE_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/models
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda
    )
    target_compile_options(zeromoe PRIVATE ${ZEROMOE_AVX512_FLAGS})
    target_compile_definitions(zeromoe PRIVATE ZEROMOE_EXL3_CUDA=1)
    set_target_properties(zeromoe PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
    target_link_libraries(zeromoe PRIVATE ${ZEROMOE_EXL3_UPSTREAM_LIB}
        CUDA::cudart
        CUDA::cublas
        CUDA::cublasLt
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(zeromoe PRIVATE OpenMP::OpenMP_CXX)
    endif()
    target_link_libraries(zeromoe PRIVATE ggml)
    if(ZEROMOE_USE_GGML_CUDA)
        target_link_libraries(zeromoe PRIVATE ggml-cuda)
        # Gates backends/ggml/ggml_moe_backend.hpp's real implementation. Without
        # it that file still compiles, but every entry point returns false
        # and execution_layer.cpp falls back to its own dispatch.
        target_compile_definitions(zeromoe PRIVATE ZEROMOE_USE_GGML_CUDA=1)
    endif()
    target_include_directories(zeromoe PRIVATE ${ZEROMOE_GGML_SRC_DIR})
    if(ZEROMOE_USE_GGML_CUDA)
        target_sources(zeromoe PRIVATE
            backends/exl3/exl3_ggml_backend.cpp
            backends/exl3/exl3_ggml_ffn.cpp
            backends/exl3/exl3_ggml_model.cpp)
        target_include_directories(zeromoe PRIVATE ${ZEROMOE_GGML_SRC_DIR}/src)
    endif()

    # Synthetic parity: legacy EXL3 kernel vs upstream exl3_gemm (no model needed).
    if(TARGET zeromoe_exl3_upstream)
        add_executable(exl3_upstream_parity EXCLUDE_FROM_ALL
            tests/exl3_upstream_parity.cu
            tests/exl3_reference_matmul.cu)
        target_include_directories(exl3_upstream_parity PRIVATE
            include "${ZEROMOE_EXLLAMAV3_SRC}")
        target_link_libraries(exl3_upstream_parity PRIVATE
            zeromoe_exl3_upstream CUDA::cudart)
        set_target_properties(exl3_upstream_parity PROPERTIES CUDA_STANDARD 17)
    endif()

    # Native test harness
    add_executable(zeromoe_tests
        tests/native/main_test.cpp
        tests/native/tests_basic_cuda.cpp
        tests/native/tests_cpu_quant.cpp
        tests/native/tests_cpu_moe.cpp
        tests/native/tests_gpu_attention.cpp
        tests/native/tests_quant_proto.cpp
        tests/native/tests_expert_cache.cpp
        tests/native/tests_gpu_kernels.cpp
        tests/native/tests_config.cpp
        backends/exl3/exl3_cpu.cpp
        backends/exl3/exl3_gpu.cpp
        core/execution_layer.cpp
        memory/scratch_buffers.cpp
        backends/cuda/cuda_kernel_dispatcher.cpp
        core/gpu_expert_executor.cpp
        core/cpu_expert_executor.cpp
        core/moe_router.cpp
        core/moe_pipeline.cpp
        core/moe_pipeline_orchestrator.cpp
        core/moe_router_service.cpp
        core/prefill_warmup_manager.cpp
        core/batched_dispatch_pipeline.cpp
        core/token_dispatch_pipeline.cpp
        core/shared_expert_executor.cpp
        ${ZEROMOE_OPTIONAL_SRCS}
        core/attention_layer.cpp
        models/base/graph_initializer.cpp
        core/hyper_connections.cpp
        models/qwen/ple_ngram.cpp
        models/deepseek/deepseek41_engram.cpp
        core/gpu_topology.cpp
        backends/cpu/ggml_dequant.cpp
        backends/cpu/ggml_grids.cpp
        backends/ggml/ggml_moe_backend.cpp
        models/graph/ggml_moe_graph_builder.cpp
        models/graph/mtp_graph_runtime.cpp
        models/qwen/qwen4exp_mtp_graph.cpp
        backends/ggml/ggml_moe_transfer_manager.cpp
        backends/ggml/ggml_moe_runner.cpp
        backends/ggml/ggml_router_runner.cpp
        models/qwen/ggml_qwen_shared_runner.cpp
        models/qwen/qwen_cpu_offload_ring.cpp
        models/qwen/ggml_qwen_decode_scheduler.cpp
        models/qwen/ggml_qwen_cpu_moe.cpp
        models/qwen/ggml_qwen4exp_graph.cpp
        models/qwen/ggml_qwen4exp_graph_arena.cpp
        models/qwen/ggml_qwen4exp_graph_build_blocks.cpp
        models/qwen/ggml_qwen4exp_graph_build.cpp
        models/qwen/ggml_qwen4exp_graph_build_moe.cpp
        models/qwen/ggml_qwen4exp_graph_decode_prepare.cpp
        models/qwen/ggml_qwen4exp_graph_diag.cpp
        models/qwen/ggml_qwen4exp_graph_checkpoint.cpp
        models/qwen/ggml_qwen4exp_graph_expert_bind.cpp
        models/qwen/ggml_qwen4exp_graph_host_node.cpp
        models/qwen/ggml_qwen4exp_graph_placement.cpp
        models/qwen/ggml_qwen4exp_graph_run.cpp
        models/qwen/ggml_qwen4exp_graph_window.cpp
        models/qwen/ggml_qwen4exp_graph_settings.cpp
        models/glm/ggml_glm5next_graph.cpp
        models/glm/ggml_glm5next_graph_arena.cpp
        models/glm/ggml_glm5next_graph_build.cpp
        models/glm/ggml_glm5next_graph_build_blocks.cpp
        models/glm/ggml_glm5next_graph_decode.cpp
        models/glm/ggml_bailingmoe3_graph.cpp
        models/glm/ggml_glm5next_graph_offload.cpp
        models/glm/ggml_glm5next_graph_partial.cpp
        models/glm/ggml_glm5next_graph_run.cpp
        models/glm/ggml_glm5next_graph_settings.cpp
        models/glm/ggml_glm5next_graph_stage.cpp
        models/glm/ggml_glm5next_graph_weights.cpp
        models/graph/ggml_graph_vram_headroom.cpp
        models/deepseek/ggml_deepseek_v41_graph.cpp
        models/deepseek/ggml_deepseek_v41_graph_decode.cpp
        backends/ggml/ggml_hc_backend.cpp
        backends/ggml/ggml_gdn_proj_backend.cpp
        models/glm/ggml_glm5next_proj_backend.cpp
        core/mtp_speculative.cpp
        core/qwen4exp_graph_mtp.cpp
        backends/cpu/kernels_cpu.cpp
        models/qwen/qwen_moe.cpp
        models/base/model_factory.cpp
        models/base/family_ops.cpp
        models/mimo/ggml_mimo2_graph.cpp
        models/graph/full_graph_models.cpp
        ${ZEROMOE_CUDA_SRCS}
    )
    target_include_directories(zeromoe_tests PRIVATE
        ${ZEROMOE_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/models
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda
    )
    target_compile_options(zeromoe_tests PRIVATE ${ZEROMOE_AVX512_FLAGS})
    set_target_properties(zeromoe_tests PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_ARCHITECTURES 86
    )
    target_link_libraries(zeromoe_tests PRIVATE ${ZEROMOE_EXL3_UPSTREAM_LIB}
        CUDA::cudart
        CUDA::cublas
        CUDA::cublasLt
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(zeromoe_tests PRIVATE OpenMP::OpenMP_CXX)
    endif()
    target_link_libraries(zeromoe_tests PRIVATE ggml)
    if(ZEROMOE_USE_GGML_CUDA)
        target_link_libraries(zeromoe_tests PRIVATE ggml-cuda)
        target_compile_definitions(zeromoe_tests PRIVATE ZEROMOE_USE_GGML_CUDA=1)
    endif()
    target_include_directories(zeromoe_tests PRIVATE ${ZEROMOE_GGML_SRC_DIR})
    if(ZEROMOE_USE_GGML_CUDA)
        target_sources(zeromoe_tests PRIVATE
            backends/exl3/exl3_ggml_backend.cpp
            backends/exl3/exl3_ggml_ffn.cpp
            backends/exl3/exl3_ggml_model.cpp
            backends/exl3/exl3_model_loader.cpp
            models/base/safetensors_adapter.cpp)
        target_include_directories(zeromoe_tests PRIVATE ${ZEROMOE_GGML_SRC_DIR}/src)
    endif()

    if(ZEROMOE_USE_GGML_CUDA)
        # Reuse engine linkage without changing its production entry point.
        get_target_property(MTP_ENGINE_SOURCES zeromoe SOURCES)
        list(REMOVE_ITEM MTP_ENGINE_SOURCES core/main.cpp)
        add_executable(qwen4exp_mtp_parity EXCLUDE_FROM_ALL
            tests/mtp/zeromoe_mtp_parity.cpp ${MTP_ENGINE_SOURCES})
        target_include_directories(qwen4exp_mtp_parity PRIVATE
            $<TARGET_PROPERTY:zeromoe,INCLUDE_DIRECTORIES>)
        target_compile_definitions(qwen4exp_mtp_parity PRIVATE
            $<TARGET_PROPERTY:zeromoe,COMPILE_DEFINITIONS>)
        target_compile_options(qwen4exp_mtp_parity PRIVATE
            $<TARGET_PROPERTY:zeromoe,COMPILE_OPTIONS>)
        target_link_libraries(qwen4exp_mtp_parity PRIVATE
            $<TARGET_PROPERTY:zeromoe,LINK_LIBRARIES>)
        set_target_properties(qwen4exp_mtp_parity PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF CUDA_ARCHITECTURES 86)
    endif()

    # Pure-header lifecycle contract test for the R2 strategy migration. It has
    # no ggml/GPU dependencies; build it explicitly with:
    #   cmake --build <build-dir> --target strategy_lifecycle_test
    add_executable(strategy_lifecycle_test EXCLUDE_FROM_ALL
        tests/test_strategy_lifecycle.cpp)
    target_include_directories(strategy_lifecycle_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/models/graph)

    # R2.1 gate: family graphs may not own expert execution.
    add_custom_target(moe_conformance
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_moe_conformance.sh
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        VERBATIM)

    if(ZEROMOE_USE_GGML_CUDA)
        add_executable(phase_alias_numerics_test EXCLUDE_FROM_ALL tests/test_phase_alias_numerics.cpp)
        target_include_directories(phase_alias_numerics_test PRIVATE ${ZEROMOE_GGML_ROOT}/include ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(phase_alias_numerics_test PRIVATE ggml ggml-cuda CUDA::cudart)
    add_executable(real_tensor_numerics_test EXCLUDE_FROM_ALL tests/test_real_tensor_numerics.cpp)
    target_include_directories(real_tensor_numerics_test PRIVATE ${ZEROMOE_GGML_ROOT}/include ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(real_tensor_numerics_test PRIVATE ggml ggml-cuda CUDA::cudart)
    endif()
    add_executable(layer_plan_test EXCLUDE_FROM_ALL tests/test_layer_plan.cpp)
    target_include_directories(layer_plan_test PRIVATE ${ZEROMOE_INCLUDE_DIRS}
        ${ZEROMOE_GGML_SRC_DIR})
    target_include_directories(layer_plan_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(layer_plan_test PRIVATE ZEROMOE_USE_GGML_CUDA=1)
    target_link_libraries(layer_plan_test PRIVATE ggml CUDA::cudart)
    add_executable(quant_table_test EXCLUDE_FROM_ALL tests/test_quant_table.cpp)
    target_include_directories(quant_table_test PRIVATE ${ZEROMOE_INCLUDE_DIRS}
        ${ZEROMOE_GGML_SRC_DIR})
    target_link_libraries(quant_table_test PRIVATE ggml)
    target_include_directories(graph_settings_test PRIVATE ${ZEROMOE_INCLUDE_DIRS})
    add_executable(resident_memory_policy_test EXCLUDE_FROM_ALL tests/test_resident_memory_policy.cpp)
    target_include_directories(resident_memory_policy_test PRIVATE ${ZEROMOE_INCLUDE_DIRS})
    add_executable(phase_route_test EXCLUDE_FROM_ALL tests/test_phase_routes.cpp)
    target_include_directories(phase_route_test PRIVATE ${ZEROMOE_INCLUDE_DIRS})

    add_executable(mimo2_spec_test EXCLUDE_FROM_ALL tests/test_mimo2_spec.cpp)
    target_include_directories(mimo2_spec_test PRIVATE ${ZEROMOE_INCLUDE_DIRS})

    # Exact F32 parity for DeepSeek's quantized grouped output projection.
    # Uses the real CPU dot and GGML quantizer/dequantizer as the oracle.
    if(ZEROMOE_USE_GGML_CUDA)
        add_executable(dsv41_grouped_outproj_test EXCLUDE_FROM_ALL
            tests/test_dsv41_grouped_outproj.cpp
            backends/cpu/kernels_cpu.cpp
            backends/cpu/ggml_grids.cpp
            kernels/quant/kquant_matmul.cu)
        target_include_directories(dsv41_grouped_outproj_test PRIVATE
            ${ZEROMOE_GGML_SRC_DIR})
        target_compile_options(dsv41_grouped_outproj_test PRIVATE
            ${ZEROMOE_AVX512_FLAGS})
        set_target_properties(dsv41_grouped_outproj_test PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF CUDA_ARCHITECTURES 86)
        target_link_libraries(dsv41_grouped_outproj_test PRIVATE
            CUDA::cudart ggml)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(dsv41_grouped_outproj_test PRIVATE
                OpenMP::OpenMP_CXX)
        endif()
    endif()

    if(ZEROMOE_USE_GGML_CUDA)
        add_executable(dsv41_cache_graph_test EXCLUDE_FROM_ALL
            tests/test_dsv41_cache_graph.cpp kernels/quant/kquant_matmul.cu)
        target_compile_definitions(dsv41_cache_graph_test PRIVATE
            ZEROMOE_USE_GGML_CUDA=1)
        set_target_properties(dsv41_cache_graph_test PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF CUDA_ARCHITECTURES 86)
        target_include_directories(dsv41_cache_graph_test PRIVATE
            ${ZEROMOE_GGML_SRC_DIR})
        target_link_libraries(dsv41_cache_graph_test PRIVATE
            CUDA::cudart ggml ggml-cuda)
    endif()

    # Unit test for backends/cuda/cuda_raii.hpp (RAII device/pinned memory, events, streams).
    add_executable(cuda_raii_test EXCLUDE_FROM_ALL tests/test_cuda_raii.cpp)
    target_include_directories(cuda_raii_test PRIVATE ${ZEROMOE_INCLUDE_DIRS})
    target_link_libraries(cuda_raii_test PRIVATE CUDA::cudart)

    # GGML Qwen CPU MoE timing probe.  The same source is also compiled
    # by bench/run_qwen_cpu_moe_compare.sh against llama.cpp's GGML shared
    # libraries, giving a layer-graph comparison without loading a model.
    add_executable(qwen_cpu_moe_ggml_bench
        bench/qwen_cpu_moe_ggml_bench.cpp)
    target_include_directories(qwen_cpu_moe_ggml_bench PRIVATE
        ${ZEROMOE_GGML_SRC_DIR})
    target_compile_options(qwen_cpu_moe_ggml_bench PRIVATE
        ${ZEROMOE_AVX512_FLAGS})
    target_link_libraries(qwen_cpu_moe_ggml_bench PRIVATE ggml)

    add_executable(exl3_kernel_bench EXCLUDE_FROM_ALL
        bench/exl3_kernel_bench.cu
        backends/exl3/exl3_aux.cu
        ${ZEROMOE_EXL3_UPSTREAM_OPS})
    target_include_directories(exl3_kernel_bench PRIVATE
        ${ZEROMOE_INCLUDE_DIRS})
    set_target_properties(exl3_kernel_bench PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_ARCHITECTURES 86)
    target_link_libraries(exl3_kernel_bench PRIVATE CUDA::cudart)

    if(ZEROMOE_USE_GGML_CUDA)
        add_executable(exl3_ggml_custom_bench EXCLUDE_FROM_ALL
            bench/exl3_ggml_custom_bench.cpp
            backends/exl3/exl3_aux.cu
            ${ZEROMOE_EXL3_UPSTREAM_OPS})
        target_include_directories(exl3_ggml_custom_bench PRIVATE
            ${ZEROMOE_INCLUDE_DIRS}
            ${ZEROMOE_GGML_SRC_DIR})
        target_compile_definitions(exl3_ggml_custom_bench PRIVATE
            ZEROMOE_USE_GGML_CUDA=1)
        set_target_properties(exl3_ggml_custom_bench PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF CUDA_ARCHITECTURES 86)
        target_link_libraries(exl3_ggml_custom_bench PRIVATE
            CUDA::cudart CUDA::cublas CUDA::cublasLt ggml ggml-cuda)

        add_executable(exl3_ggml_cuda_adapter_bench EXCLUDE_FROM_ALL
            bench/exl3_ggml_cuda_adapter_bench.cpp
            backends/exl3/exl3_ggml_backend.cpp
            backends/exl3/exl3_aux.cu
            ${ZEROMOE_EXL3_UPSTREAM_OPS})
        target_include_directories(exl3_ggml_cuda_adapter_bench PRIVATE
            ${ZEROMOE_INCLUDE_DIRS}
            ${ZEROMOE_GGML_SRC_DIR}
            ${ZEROMOE_GGML_SRC_DIR}/src)
        target_compile_definitions(exl3_ggml_cuda_adapter_bench PRIVATE
            ZEROMOE_USE_GGML_CUDA=1)
        set_target_properties(exl3_ggml_cuda_adapter_bench PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF CUDA_ARCHITECTURES 86)
        target_link_libraries(exl3_ggml_cuda_adapter_bench PRIVATE
            CUDA::cudart CUDA::cublas CUDA::cublasLt ggml ggml-cuda)

        add_executable(exl3_ggml_cuda_real_bench EXCLUDE_FROM_ALL
            bench/exl3_ggml_cuda_real_bench.cpp
            backends/exl3/exl3_ggml_backend.cpp
            backends/exl3/exl3_model_loader.cpp
            models/base/safetensors_adapter.cpp
            backends/exl3/exl3_cpu.cpp
            backends/cpu/kernels_cpu.cpp
            backends/cpu/ggml_grids.cpp
            backends/exl3/exl3_aux.cu
            ${ZEROMOE_EXL3_UPSTREAM_OPS})
        target_include_directories(exl3_ggml_cuda_real_bench PRIVATE
            ${ZEROMOE_INCLUDE_DIRS}
            ${ZEROMOE_GGML_SRC_DIR}
            ${ZEROMOE_GGML_SRC_DIR}/src)
        target_compile_options(exl3_ggml_cuda_real_bench PRIVATE
            ${ZEROMOE_AVX512_FLAGS})
        target_compile_definitions(exl3_ggml_cuda_real_bench PRIVATE
            ZEROMOE_USE_GGML_CUDA=1)
        set_target_properties(exl3_ggml_cuda_real_bench PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF CUDA_ARCHITECTURES 86)
        target_link_libraries(exl3_ggml_cuda_real_bench PRIVATE
            CUDA::cudart CUDA::cublas CUDA::cublasLt ggml ggml-cuda)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(exl3_ggml_cuda_real_bench PRIVATE
                OpenMP::OpenMP_CXX)
        endif()

        add_executable(exl3_ggml_cuda_real_ffn_bench EXCLUDE_FROM_ALL
            bench/exl3_ggml_cuda_real_ffn_bench.cpp
            backends/exl3/exl3_ggml_backend.cpp
            backends/exl3/exl3_ggml_ffn.cpp
            backends/exl3/exl3_gpu.cpp
            backends/exl3/exl3_model_loader.cpp
            models/base/safetensors_adapter.cpp
            backends/exl3/exl3_cpu.cpp
            backends/cpu/kernels_cpu.cpp
            backends/cpu/ggml_grids.cpp
            backends/exl3/exl3_aux.cu
            ${ZEROMOE_EXL3_UPSTREAM_OPS})
        target_include_directories(exl3_ggml_cuda_real_ffn_bench PRIVATE
            ${ZEROMOE_INCLUDE_DIRS}
            ${ZEROMOE_GGML_SRC_DIR}
            ${ZEROMOE_GGML_SRC_DIR}/src)
        target_compile_options(exl3_ggml_cuda_real_ffn_bench PRIVATE
            ${ZEROMOE_AVX512_FLAGS})
        target_compile_definitions(exl3_ggml_cuda_real_ffn_bench PRIVATE
            ZEROMOE_USE_GGML_CUDA=1
            ZEROMOE_EXL3_CUDA=1)
        set_target_properties(exl3_ggml_cuda_real_ffn_bench PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF CUDA_ARCHITECTURES 86)
        target_link_libraries(exl3_ggml_cuda_real_ffn_bench PRIVATE
            CUDA::cudart CUDA::cublas CUDA::cublasLt ggml ggml-cuda)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(exl3_ggml_cuda_real_ffn_bench PRIVATE
                OpenMP::OpenMP_CXX)
        endif()

        add_executable(exl3_ggml_cuda_real_routed_ffn_bench EXCLUDE_FROM_ALL
            bench/exl3_ggml_cuda_real_routed_ffn_bench.cpp
            backends/exl3/exl3_ggml_backend.cpp
            backends/exl3/exl3_ggml_ffn.cpp
            backends/exl3/exl3_gpu.cpp
            backends/exl3/exl3_model_loader.cpp
            models/base/safetensors_adapter.cpp
            backends/exl3/exl3_cpu.cpp
            backends/cpu/kernels_cpu.cpp
            backends/cpu/ggml_grids.cpp
            backends/exl3/exl3_aux.cu
            ${ZEROMOE_EXL3_UPSTREAM_OPS})
        target_include_directories(exl3_ggml_cuda_real_routed_ffn_bench PRIVATE
            ${ZEROMOE_INCLUDE_DIRS}
            ${ZEROMOE_GGML_SRC_DIR}
            ${ZEROMOE_GGML_SRC_DIR}/src)
        target_compile_options(exl3_ggml_cuda_real_routed_ffn_bench PRIVATE
            ${ZEROMOE_AVX512_FLAGS})
        target_compile_definitions(exl3_ggml_cuda_real_routed_ffn_bench PRIVATE
            ZEROMOE_USE_GGML_CUDA=1
            ZEROMOE_EXL3_CUDA=1)
        set_target_properties(exl3_ggml_cuda_real_routed_ffn_bench PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF CUDA_ARCHITECTURES 86)
        target_link_libraries(exl3_ggml_cuda_real_routed_ffn_bench PRIVATE
            CUDA::cudart CUDA::cublas CUDA::cublasLt ggml ggml-cuda)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(exl3_ggml_cuda_real_routed_ffn_bench PRIVATE
                OpenMP::OpenMP_CXX)
        endif()

        add_executable(exl3_ggml_cuda_cache_graph_bench EXCLUDE_FROM_ALL
            bench/exl3_ggml_cuda_cache_graph_bench.cpp
            backends/exl3/exl3_ggml_backend.cpp
            backends/exl3/exl3_ggml_ffn.cpp
            backends/exl3/exl3_gpu.cpp
            backends/exl3/exl3_model_loader.cpp
            models/base/safetensors_adapter.cpp
            backends/exl3/exl3_cpu.cpp
            backends/cpu/kernels_cpu.cpp
            backends/cpu/ggml_grids.cpp
            backends/exl3/exl3_aux.cu
            ${ZEROMOE_EXL3_UPSTREAM_OPS}
            ${ZEROMOE_EXL3_UPSTREAM_OPS})
        target_include_directories(exl3_ggml_cuda_cache_graph_bench PRIVATE
            ${ZEROMOE_INCLUDE_DIRS}
            ${ZEROMOE_GGML_SRC_DIR}
            ${ZEROMOE_GGML_SRC_DIR}/src)
        target_compile_options(exl3_ggml_cuda_cache_graph_bench PRIVATE
            ${ZEROMOE_AVX512_FLAGS})
        target_compile_definitions(exl3_ggml_cuda_cache_graph_bench PRIVATE
            ZEROMOE_USE_GGML_CUDA=1
            ZEROMOE_EXL3_CUDA=1)
        set_target_properties(exl3_ggml_cuda_cache_graph_bench PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF CUDA_ARCHITECTURES 86)
        target_link_libraries(exl3_ggml_cuda_cache_graph_bench PRIVATE
            CUDA::cudart CUDA::cublas CUDA::cublasLt ggml ggml-cuda
            ${ZEROMOE_EXL3_UPSTREAM_LIB})
        if(OpenMP_CXX_FOUND)
            target_link_libraries(exl3_ggml_cuda_cache_graph_bench PRIVATE
                OpenMP::OpenMP_CXX)
        endif()

        add_executable(exl3_ggml_cuda_fused_ffn_bench EXCLUDE_FROM_ALL
            bench/exl3_ggml_cuda_fused_ffn_bench.cpp
            backends/exl3/exl3_aux.cu
            ${ZEROMOE_EXL3_UPSTREAM_OPS})
        target_include_directories(exl3_ggml_cuda_fused_ffn_bench PRIVATE
            ${ZEROMOE_INCLUDE_DIRS}
            ${ZEROMOE_GGML_SRC_DIR}
            ${ZEROMOE_GGML_SRC_DIR}/src)
        target_compile_definitions(exl3_ggml_cuda_fused_ffn_bench PRIVATE
            ZEROMOE_USE_GGML_CUDA=1)
        set_target_properties(exl3_ggml_cuda_fused_ffn_bench PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF CUDA_ARCHITECTURES 86)
        target_link_libraries(exl3_ggml_cuda_fused_ffn_bench PRIVATE
            CUDA::cudart CUDA::cublas CUDA::cublasLt ggml ggml-cuda)

        add_executable(exl3_ggml_cuda_moe_bench EXCLUDE_FROM_ALL
            bench/exl3_ggml_cuda_moe_bench.cpp
            backends/exl3/exl3_aux.cu
            ${ZEROMOE_EXL3_UPSTREAM_OPS})
        target_include_directories(exl3_ggml_cuda_moe_bench PRIVATE
            ${ZEROMOE_INCLUDE_DIRS}
            ${ZEROMOE_GGML_SRC_DIR}
            ${ZEROMOE_GGML_SRC_DIR}/src)
        target_compile_definitions(exl3_ggml_cuda_moe_bench PRIVATE
            ZEROMOE_USE_GGML_CUDA=1)
        set_target_properties(exl3_ggml_cuda_moe_bench PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF CUDA_ARCHITECTURES 86)
        target_link_libraries(exl3_ggml_cuda_moe_bench PRIVATE
            CUDA::cudart CUDA::cublas CUDA::cublasLt ggml ggml-cuda)
    endif()

    if(OpenMP_CXX_FOUND)
        add_executable(exl3_cpu_bench EXCLUDE_FROM_ALL
            bench/exl3_cpu_bench.cpp
            backends/exl3/exl3_cpu.cpp
            backends/cpu/kernels_cpu.cpp
            backends/cpu/ggml_grids.cpp)
        target_include_directories(exl3_cpu_bench PRIVATE
            ${ZEROMOE_INCLUDE_DIRS}
            ${ZEROMOE_GGML_SRC_DIR}
            ${ZEROMOE_GGML_SRC_DIR}/src)
        target_compile_options(exl3_cpu_bench PRIVATE ${ZEROMOE_AVX512_FLAGS})
        target_link_libraries(exl3_cpu_bench PRIVATE ggml OpenMP::OpenMP_CXX)

        add_executable(exl3_hc_bench EXCLUDE_FROM_ALL
            bench/exl3_hc_bench.cpp
            backends/exl3/exl3_gpu.cpp
            backends/exl3/exl3_cpu.cpp
            backends/cpu/kernels_cpu.cpp
            backends/cpu/ggml_grids.cpp
            backends/exl3/exl3_aux.cu
            ${ZEROMOE_EXL3_UPSTREAM_OPS})
        target_include_directories(exl3_hc_bench PRIVATE
            ${ZEROMOE_INCLUDE_DIRS}
            ${ZEROMOE_GGML_SRC_DIR}
            ${ZEROMOE_GGML_SRC_DIR}/src)
        target_compile_options(exl3_hc_bench PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:${ZEROMOE_AVX512_FLAGS}>")
        set_target_properties(exl3_hc_bench PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF CUDA_ARCHITECTURES 86)
        target_link_libraries(exl3_hc_bench PRIVATE
            ggml OpenMP::OpenMP_CXX CUDA::cudart CUDA::cublas)
    endif()

    # One-block IQ2_XXS CPU-vs-CUDA diagnostic.  This is deliberately a
    # separate opt-in executable: it reports the CPU Q8_K and CUDA Q8_1
    # activation quantization contracts plus the actual ggml_mul_mat_id
    # result, without changing the production dispatch path.
    if(ZEROMOE_USE_GGML_CUDA)
        add_executable(qwen_iq2_cpu_cuda_probe
            bench/qwen_iq2_cpu_cuda_probe.cpp
            backends/cpu/ggml_grids.cpp)
        target_include_directories(qwen_iq2_cpu_cuda_probe PRIVATE
            ${ZEROMOE_GGML_SRC_DIR}
            ${ZEROMOE_GGML_SRC_DIR}/src)
        target_compile_options(qwen_iq2_cpu_cuda_probe PRIVATE
            ${ZEROMOE_AVX512_FLAGS})
        target_link_libraries(qwen_iq2_cpu_cuda_probe PRIVATE
            ggml ggml-cuda CUDA::cudart)
        target_compile_definitions(qwen_iq2_cpu_cuda_probe PRIVATE
            ZEROMOE_USE_GGML_CUDA=1)
    endif()

    # Layer-13 pointer-indirection Phase-1 laboratory.  This target links
    # only the real IQ1_M/IQ4_NL production kernels, the device SwiGLU
    # activation, and the project's GGML lookup tables.  It intentionally
    # does not link the engine or mutate runtime dispatch; see the benchmark
    # source for the VRAM/mapped/H2D-stage conditions and M=1/3/6 contention
    # protocol.
    if(EXISTS
       "${CMAKE_CURRENT_SOURCE_DIR}/bench/qwen4exp_pointer_spike.cu")
        add_executable(qwen4exp_pointer_spike
            bench/qwen4exp_pointer_spike.cu
            kernels/quant/iq1_dequant.cu
            kernels/quant/iq4xs_matmul.cu
            kernels/math/expert_activation.cu
            backends/cpu/ggml_grids.cpp)
        target_include_directories(qwen4exp_pointer_spike PRIVATE
            ${ZEROMOE_INCLUDE_DIRS}
            ${CMAKE_CURRENT_SOURCE_DIR}/cuda
            ${ZEROMOE_GGML_SRC_DIR})
        set_target_properties(qwen4exp_pointer_spike PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF
            CUDA_ARCHITECTURES 86)
        target_link_libraries(qwen4exp_pointer_spike PRIVATE CUDA::cudart)

        # Layer-13 Phase-2 graph/staging laboratory.  This deliberately
        # remains separate from the engine: it validates device route IDs,
        # the constant descriptor table, staged pointer gathering, and the
        # indirect production kernels before any runtime dispatch change.
        add_executable(qwen4exp_staging_spike
            bench/qwen4exp_staging_spike.cu
            kernels/quant/iq1_dequant.cu
            kernels/quant/iq4xs_matmul.cu
            kernels/math/expert_activation.cu
            backends/cpu/ggml_grids.cpp)
        target_include_directories(qwen4exp_staging_spike PRIVATE
            ${ZEROMOE_INCLUDE_DIRS}
            ${CMAKE_CURRENT_SOURCE_DIR}/cuda
            ${ZEROMOE_GGML_SRC_DIR})
        set_target_properties(qwen4exp_staging_spike PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF
            CUDA_ARCHITECTURES 86)
        target_link_libraries(qwen4exp_staging_spike PRIVATE CUDA::cudart)
    endif()

    # CPUOffload Milestone-C synthetic graph sandbox.  This target is
    # deliberately independent of the model engine and GGML: it validates
    # only the serialized-vs-branched D2H -> synchronous host-node -> H2D
    # topology and the dependent device combine.
    add_executable(qwen_cpu_offload_graph_sandbox
        bench/qwen_cpu_offload_graph_sandbox.cu)
    set_target_properties(qwen_cpu_offload_graph_sandbox PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_ARCHITECTURES 86)
    target_link_libraries(qwen_cpu_offload_graph_sandbox PRIVATE CUDA::cudart)

    # CPUOffload ring wait-latency microbenchmark (Tier 1). Exercises the
    # REAL QwenCpuOffloadRing class (models/qwen/qwen_cpu_offload_ring.cpp) with a
    # synthetic CPU worker thread standing in for GgmlQwenCpuMoeRunner, to
    # isolate ZEROMOE_QWEN4EXP_RING_SPIN_US's effect on wait latency from
    # decode-loop noise. No GGML/model dependency.
    add_executable(qwen_cpu_offload_ring_spin_bench
        bench/qwen_cpu_offload_ring_spin_bench.cpp
        models/qwen/qwen_cpu_offload_ring.cpp)
    set_target_properties(qwen_cpu_offload_ring_spin_bench PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_ARCHITECTURES 86)
    target_link_libraries(qwen_cpu_offload_ring_spin_bench PRIVATE
        CUDA::cudart)

    # Tools are built by the subdirectory CMakeLists as well, but ensure the
    # root can also drive them.
    add_subdirectory(tools)

# ---------------------------------------------------------------------------
# CPU-only build (macOS, or Linux without CUDA)
# ---------------------------------------------------------------------------
else()
    message(STATUS "ZeroMoE: CPU-only build path selected")

    # Choose the right CPU kernel backend.
    set(ZEROMOE_CPU_KERNEL_SRCS)
    # models/base/safetensors_adapter.cpp was never registered here even though
    # tests/test_components.cpp's test_safetensors_adapter() calls it
    # directly -- test_components (and any other target linking
    # kernels_cpu) failed to link with undefined SafetensorsAdapter symbols
    # regardless of platform, unrelated to Apple Silicon vs x86 below.
    list(APPEND ZEROMOE_CPU_KERNEL_SRCS
        backends/exl3/exl3_cpu.cpp
        backends/exl3/exl3_model_loader.cpp
        models/base/safetensors_adapter.cpp)
    if(ZEROMOE_APPLE_SILICON)
        list(APPEND ZEROMOE_CPU_KERNEL_SRCS backends/cpu/kernels_cpu_neon.cpp)
        message(STATUS "ZeroMoE: using ARM NEON CPU kernel backend")
        if(ZEROMOE_METAL)
            list(APPEND ZEROMOE_CPU_KERNEL_SRCS backends/cpu/kernels_metal.mm)
            message(STATUS "ZeroMoE: Metal/MPS dispatch enabled (falls back to NEON)")
        endif()
    else()
        list(APPEND ZEROMOE_CPU_KERNEL_SRCS backends/cpu/kernels_cpu.cpp)
        message(STATUS "ZeroMoE: using x86 AVX-512 CPU kernel backend")
    endif()

    add_library(kernels_cpu STATIC ${ZEROMOE_CPU_KERNEL_SRCS})
    target_compile_options(kernels_cpu PRIVATE ${ZEROMOE_AVX512_FLAGS})
    target_include_directories(kernels_cpu PUBLIC
        ${ZEROMOE_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/models
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(kernels_cpu PUBLIC OpenMP::OpenMP_CXX)
    endif()

    # Apple Silicon tuning: prefer the Apple-specific CPU tuning when the
    # toolchain supports it.  This benefits M3/M4/M5 (and is harmless on M1/M2).
    if(ZEROMOE_APPLE_SILICON)
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("-mcpu=apple-m3" ZEROMOE_HAS_APPLE_M3_FLAG)
        if(ZEROMOE_HAS_APPLE_M3_FLAG)
            target_compile_options(kernels_cpu PRIVATE -mcpu=apple-m3)
            message(STATUS "ZeroMoE: applied -mcpu=apple-m3 to CPU kernels")
        endif()
        if(ZEROMOE_METAL)
            set_source_files_properties(backends/cpu/kernels_metal.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
            find_library(METAL_FRAMEWORK Metal REQUIRED)
            find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
            target_link_libraries(kernels_cpu PUBLIC ${METAL_FRAMEWORK} ${MPS_FRAMEWORK})
            target_compile_definitions(kernels_cpu PUBLIC ZEROMOE_METAL=1)
        endif()
    endif()

    target_sources(kernels_cpu PRIVATE
        backends/cpu/ggml_dequant.cpp
        backends/cpu/ggml_grids.cpp)
    target_link_libraries(kernels_cpu PUBLIC ggml)
    target_include_directories(kernels_cpu PUBLIC ${ZEROMOE_GGML_SRC_DIR})

    # CPU-only component tests (cross-platform, no CUDA)
    add_executable(test_components tests/test_components.cpp)
    target_link_libraries(test_components PRIVATE kernels_cpu)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(test_components PRIVATE OpenMP::OpenMP_CXX)
    endif()

    # CPU-only inference binary (zeromoe_cpu)
    add_library(zeromoe_cpu_runtime STATIC
        backends/exl3/exl3_decode.cpp
        backends/cpu/cpu_dense.cpp
        backends/cpu/cpu_engine.cpp
        backends/cpu/cpu_forward.cpp
        backends/cpu/cpu_model_backend.cpp
        models/qwen/cpu_qwen_moe.cpp
        models/qwen/cpu_qwen4exp.cpp
    )
    target_include_directories(zeromoe_cpu_runtime PUBLIC
        ${ZEROMOE_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/models
    )
    target_link_libraries(zeromoe_cpu_runtime PUBLIC kernels_cpu)
    add_executable(zeromoe_cpu backends/cpu/cpu_main.cpp)

    # When the kernel backend contains Objective-C++ (Metal dispatch), the
    # final executable must link the Objective-C runtime / Foundation so the
    # linker resolves ARC/runtime symbols from the static library objects.
    if(ZEROMOE_METAL AND ZEROMOE_APPLE_SILICON)
        find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
        foreach(_tgt test_components zeromoe_cpu)
            target_link_libraries(${_tgt} PRIVATE ${FOUNDATION_FRAMEWORK})
        endforeach()
    endif()
    if(OpenMP_CXX_FOUND)
        target_link_libraries(zeromoe_cpu_runtime PUBLIC OpenMP::OpenMP_CXX)
    endif()

    # Tools
    add_subdirectory(tools)
endif()

message(STATUS "ZeroMoE: configure done")

enable_testing()
add_subdirectory(server)
if(TARGET zeromoe)
    target_link_libraries(zeromoe PRIVATE zeromoe_server)
else()
    add_library(zeromoe_cpu_api STATIC backends/cpu/cpu_api.cpp)
    target_link_libraries(zeromoe_cpu_api PUBLIC zeromoe_cpu_runtime zeromoe_server)
    target_link_libraries(zeromoe_cpu PRIVATE zeromoe_cpu_api)
    if(ZEROMOE_SERVER_TESTS)
        add_executable(cpu_api_tests tests/cpu_api_test.cpp)
        target_link_libraries(cpu_api_tests PRIVATE zeromoe_cpu_api)
        add_test(NAME cpu_api_tests COMMAND cpu_api_tests)
    endif()
endif()

# EXL3 bench targets compile backends/exl3/exl3_upstream_ops.cu directly and need the
# upstream kernel library plus the torch stub include path.
if(TARGET zeromoe_exl3_upstream)
    foreach(_exl3_target
            exl3_kernel_bench exl3_ggml_custom_bench exl3_ggml_cuda_adapter_bench
            exl3_ggml_cuda_real_bench exl3_ggml_cuda_real_ffn_bench
            exl3_ggml_cuda_real_routed_ffn_bench exl3_ggml_cuda_fused_ffn_bench
            exl3_ggml_cuda_moe_bench exl3_hc_bench)
        if(TARGET ${_exl3_target})
            target_link_libraries(${_exl3_target} PRIVATE zeromoe_exl3_upstream)
        endif()
    endforeach()
endif()

if(TARGET zeromoe AND ZEROMOE_WRITE_BUILD_PROVENANCE)
    set(_zeromoe_provenance_args
        stamp
        --binary "$<TARGET_FILE:zeromoe>"
        --repo "${CMAKE_CURRENT_SOURCE_DIR}"
        --build-dir "${CMAKE_BINARY_DIR}"
        --expected-commit "${ZEROMOE_CONFIGURED_SOURCE_COMMIT}"
        --expected-source-snapshot-sha256 "${ZEROMOE_CONFIGURED_SOURCE_SNAPSHOT_SHA256}")
    if(ZEROMOE_GGML_ROOT)
        list(APPEND _zeromoe_provenance_args --ggml-root "${ZEROMOE_GGML_ROOT}")
    endif()
    if(ZEROMOE_GGML_COMMIT)
        list(APPEND _zeromoe_provenance_args --ggml-base "${ZEROMOE_GGML_COMMIT}")
    endif()
    add_custom_command(TARGET zeromoe POST_BUILD
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/build_provenance.py"
                ${_zeromoe_provenance_args}
        COMMENT "Stamping source and GGML provenance for zeromoe"
        VERBATIM COMMAND_EXPAND_LISTS)
    set_property(TARGET zeromoe APPEND PROPERTY LINK_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/build_provenance.py")
endif()

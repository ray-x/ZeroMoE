cmake_minimum_required(VERSION 3.25)
project(MoEStream CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Build options
# ---------------------------------------------------------------------------
option(MOESTREAM_CPU_ONLY "Disable CUDA and build CPU-only components" OFF)
option(MOESTREAM_USE_GGML "Use ggml (via llama.cpp source tree) for quantized CPU kernels" OFF)
option(MOESTREAM_USE_GGML_CUDA
       "Build/link ggml-cuda for the experimental MUL_MAT_ID prefill backend"
       OFF)
# ggml's own default for this is OFF outside llama.cpp (its option() comment
# literally says "llama.cpp only") -- a plain MOESTREAM_USE_GGML_CUDA=ON build
# silently never defines GGML_CUDA_USE_GRAPHS, so ggml-cuda's CUDA graph
# capture/replay path (ggml_backend_cuda_graph_compute in ggml-cuda.cu) is
# compiled out entirely: every decode-graph node gets its own individual
# cudaLaunchKernel instead of one batched cudaGraphLaunch per captured split.
# Measured real-world cost of missing this on the full-graph Qwen4Exp decode
# path: full-graph decode dropped from a verified ~20 tok/s (graphs enabled)
# to ~8 tok/s (graphs silently absent) on identical hardware/model/config --
# this was mistaken for a correctness or scheduling regression for an entire
# investigation before the missing compile define was found. Default ON here
# so any MOESTREAM_USE_GGML_CUDA build gets it without the caller having to
# know this ggml-specific gotcha; still overridable with -DGGML_CUDA_GRAPHS=OFF.
option(MOESTREAM_ENABLE_CUDA_GRAPHS
       "Enable ggml CUDA graph capture/replay (GGML_CUDA_GRAPHS) for full-graph decode/prefill"
       ON)
option(MOESTREAM_ENABLE_AVX512_VNNI
       "Compile MoEStream and ggml CPU kernels with AVX-512 VNNI (set OFF on CPUs without VNNI)"
       ON)
option(MOESTREAM_GGML_OPENMP
       "Build ggml's CPU backend with OpenMP (OFF uses its persistent threadpool)"
       ON)
option(MOESTREAM_ENABLE_GGML_HOST_NODE_PATCHES
       "Apply GGML CUDA/CPUOffload host-node integration patches"
       ON)
option(MOESTREAM_GGML_DEBUG_PATCHES
       "Also apply the debug/profiling ggml patches in patches/debug (not part of the main series)"
       OFF)
option(MOESTREAM_USE_LLAMA_CPP "Deprecated alias for MOESTREAM_USE_GGML" OFF)
option(MOESTREAM_METAL "Enable Apple Metal GPU backend (requires ggml)" OFF)

# On macOS the CPU-only binary is the only usable engine, and real MoE models
# are almost always quantized.  Default ggml fetch to ON here; users can still
# override with -DMOESTREAM_USE_GGML=OFF.
if(APPLE)
    set(MOESTREAM_USE_GGML ON CACHE BOOL "Fetch ggml for quantized CPU decode" FORCE)
    message(STATUS "MoEStream: Apple platform auto-enabling MOESTREAM_USE_GGML=ON")
endif()

# macOS has no NVIDIA CUDA SDK, so the full engine cannot build there.
if(APPLE)
    set(MOESTREAM_CPU_ONLY ON CACHE BOOL "macOS has no CUDA SDK; force CPU-only build" FORCE)
    message(STATUS "MoEStream: Apple platform detected — forcing MOESTREAM_CPU_ONLY=ON")
endif()

# The full engine links ggml for reference dequantizers.  If you already have
# a ggml install somewhere else, point CMake to it and set this OFF; otherwise
# setting this ON fetches a stable llama.cpp release and builds ggml in-tree.
if(MOESTREAM_USE_LLAMA_CPP AND NOT MOESTREAM_USE_GGML)
    set(MOESTREAM_USE_GGML ON CACHE BOOL "" FORCE)
    message(WARNING "MOESTREAM_USE_LLAMA_CPP is deprecated; use MOESTREAM_USE_GGML=ON")
endif()
if(NOT MOESTREAM_CPU_ONLY AND NOT MOESTREAM_USE_GGML)
    message(FATAL_ERROR
        "Full engine builds require ggml. Either set MOESTREAM_USE_GGML=ON "
        "to fetch ggml automatically, or provide your own ggml and link it.")
endif()

# Detect Apple Silicon (M1/M2/M3/M4/M5 all share arm64; M3+ specific flags are
# applied below when the compiler supports them).
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    set(MOESTREAM_APPLE_SILICON ON)
    message(STATUS "MoEStream: Apple Silicon (arm64) detected")
else()
    set(MOESTREAM_APPLE_SILICON OFF)
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
# MOESTREAM_AVX512_FLAGS staying empty here just means the build fails
# later at these exact source files with a clearer compiler error instead
# of a silent miscompile. Deliberately NOT -mavx512vnni: the source
# already compile-time-gates its VNNI fast path (#if
# defined(__AVX512VNNI__)).  Keep VNNI opt-in because forcing that flag here
# would risk a SIGILL at runtime on a real AVX-512 CPU that lacks VNNI
# specifically (some early Skylake-X/Cascade Lake parts).
set(MOESTREAM_AVX512_FLAGS "")
if(NOT MOESTREAM_APPLE_SILICON)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl -mf16c" MOESTREAM_HAS_AVX512_FLAGS)
    if(MOESTREAM_HAS_AVX512_FLAGS)
        set(MOESTREAM_AVX512_FLAGS -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl -mf16c)
    else()
        message(WARNING "MoEStream: compiler does not accept AVX-512/F16C flags -- "
                         "kernels_cpu.cpp/attention_layer.cpp/execution_layer.cpp require "
                         "AVX-512 (see CLAUDE.md) and may fail to build without them")
    endif()
    if(MOESTREAM_ENABLE_AVX512_VNNI)
        check_cxx_compiler_flag("-mavx512vnni" MOESTREAM_HAS_AVX512_VNNI_FLAG)
        if(NOT MOESTREAM_HAS_AVX512_VNNI_FLAG)
            message(FATAL_ERROR
                    "MoEStream: MOESTREAM_ENABLE_AVX512_VNNI=ON, but the compiler "
                    "does not accept -mavx512vnni")
        endif()
        list(APPEND MOESTREAM_AVX512_FLAGS -mavx512vnni)
    endif()
endif()

# ---------------------------------------------------------------------------
# OpenMP (used by CPU kernels and some reference loops)
# ---------------------------------------------------------------------------
find_package(OpenMP)

# ---------------------------------------------------------------------------
# ggml integration (optional)
# ---------------------------------------------------------------------------
if(MOESTREAM_USE_GGML)
    # Default to a stable, verified ggml tag in the ggerganov/ggml repo.
    # (The old "b3923" string was not a valid tag in either ggml or llama.cpp.)
    set(MOESTREAM_GGML_TAG "v0.9.11" CACHE STRING "ggml git tag to fetch")
    set(MOESTREAM_GGML_ROOT "" CACHE PATH "Path to a local ggml checkout (fetched automatically if empty)")

    # Provide a deprecated alias for configs that still set it.
    if(MOESTREAM_LLAMA_CPP_ROOT AND NOT MOESTREAM_GGML_ROOT)
        set(MOESTREAM_GGML_ROOT "${MOESTREAM_LLAMA_CPP_ROOT}" CACHE PATH "" FORCE)
        message(WARNING "MOESTREAM_LLAMA_CPP_ROOT is deprecated; use MOESTREAM_GGML_ROOT")
    endif()
    if(MOESTREAM_LLAMA_CPP_TAG)
        set(MOESTREAM_GGML_TAG "${MOESTREAM_LLAMA_CPP_TAG}" CACHE STRING "" FORCE)
        message(WARNING "MOESTREAM_LLAMA_CPP_TAG is deprecated; use MOESTREAM_GGML_TAG")
    endif()

    # Disable everything we do not need for a CPU decode assistant, except the
    # explicit experimental CUDA backend option. GGML_CUDA must be set BEFORE
    # the subproject is configured.
    set(GGML_CUDA        ${MOESTREAM_USE_GGML_CUDA} CACHE BOOL "" FORCE)
    set(GGML_CUDA_GRAPHS ${MOESTREAM_ENABLE_CUDA_GRAPHS} CACHE BOOL "" FORCE)
    # Loud, impossible-to-miss configure-time confirmation of the exact
    # setting that silently regressed full-graph decode from ~20 tok/s to
    # ~8 tok/s once already (see MOESTREAM_ENABLE_CUDA_GRAPHS's own comment
    # above). A default is not a guarantee -- print the resolved value on
    # every configure, and fail outright if CUDA is on but graphs got
    # turned off without an explicit, informed override.
    if(MOESTREAM_USE_GGML_CUDA AND NOT MOESTREAM_ENABLE_CUDA_GRAPHS)
        message(WARNING
            "MOESTREAM_USE_GGML_CUDA=ON but MOESTREAM_ENABLE_CUDA_GRAPHS=OFF: "
            "full-graph decode will silently run WITHOUT CUDA graph capture/"
            "replay (every node becomes an individual cudaLaunchKernel). "
            "This previously cost ~20 tok/s -> ~8 tok/s and was mistaken for "
            "a correctness/scheduling bug for an entire investigation. Only "
            "disable this deliberately for a real A/B; the runtime "
            "[CUDA-GRAPHS-CHECK] log line will also confirm this at every "
            "process start regardless of this setting.")
    else()
        message(STATUS "MoEStream: GGML_CUDA_GRAPHS=${GGML_CUDA_GRAPHS} "
                       "(verified again at runtime via [CUDA-GRAPHS-CHECK])")
    endif()
    set(GGML_HIPBLAS     OFF CACHE BOOL "" FORCE)
    set(GGML_VULKAN      OFF CACHE BOOL "" FORCE)
    set(GGML_SYCL        OFF CACHE BOOL "" FORCE)
    if(MOESTREAM_GGML_OPENMP AND NOT OpenMP_CXX_FOUND)
        message(WARNING "MOESTREAM_GGML_OPENMP=ON requested, but OpenMP was not found; ggml CPU backend will use its native threadpool")
    endif()
    set(GGML_OPENMP      ${MOESTREAM_GGML_OPENMP} CACHE BOOL "" FORCE)
    message(STATUS "MoEStream: GGML_OPENMP=${GGML_OPENMP} (MOESTREAM_GGML_OPENMP=${MOESTREAM_GGML_OPENMP})")
    # Keep ggml's CPU backend on the same ISA choice as MoEStream's CPU
    # kernels.  This is the single opt-out for older CPUs: setting
    # MOESTREAM_ENABLE_AVX512_VNNI=OFF also disables VNNI in ggml, including
    # when reconfiguring a build directory that previously had it enabled.
    set(GGML_AVX512_VNNI ${MOESTREAM_ENABLE_AVX512_VNNI} CACHE BOOL "" FORCE)
    if(MOESTREAM_ENABLE_AVX512_VNNI)
        set(GGML_AVX512       ON CACHE BOOL "" FORCE)
    endif()
    message(STATUS
        "MoEStream: AVX-512 VNNI=${MOESTREAM_ENABLE_AVX512_VNNI} (MoEStream and ggml CPU kernels)")

    if(MOESTREAM_METAL AND APPLE)
        # ggml's Metal backend requires the offline shader compiler (metal,
        # metallib), which Apple only ships inside the full Xcode.app.  Our
        # own MPS dispatch uses the runtime Metal Performance Shaders API and
        # does NOT need those tools.  If the compiler is missing, keep our MPS
        # path alive but turn off ggml Metal to avoid a configure/build failure.
        find_program(_MOESTREAM_METALC metal)
        find_program(_MOESTREAM_METALLIB metallib)
        if(_MOESTREAM_METALC AND _MOESTREAM_METALLIB)
            set(GGML_METAL ON CACHE BOOL "" FORCE)
            message(STATUS "MoEStream: ggml Metal backend enabled (metal/metallib found)")
        else()
            set(GGML_METAL OFF CACHE BOOL "" FORCE)
            message(STATUS "MoEStream: ggml Metal backend disabled (full Xcode not installed); "
                            "custom MPS dispatch still enabled")
        endif()
    else()
        set(GGML_METAL OFF CACHE BOOL "" FORCE)
    endif()

    if(NOT MOESTREAM_GGML_ROOT)
        # Use a direct shell clone instead of FetchContent's ExternalProject.
        # Some sandboxed CMake environments fail to detect git version and
        # abort before cloning; running git from the shell works around that.
        set(_moestream_ggml_src "${CMAKE_BINARY_DIR}/_deps/ggml-src")
        if(NOT EXISTS "${_moestream_ggml_src}/CMakeLists.txt")
            message(STATUS "MoEStream: cloning ggml tag ${MOESTREAM_GGML_TAG} ...")
            # Avoid sandbox/gitconfig permission issues on some hosts.
            set(ENV{GIT_CONFIG_GLOBAL} "/dev/null")
            set(ENV{GIT_CONFIG_NOSYSTEM} "1")
            execute_process(
                COMMAND git clone --depth 1 --branch ${MOESTREAM_GGML_TAG}
                        https://github.com/ggerganov/ggml.git
                        "${_moestream_ggml_src}"
                RESULT_VARIABLE _moestream_ggml_clone_result
                ERROR_VARIABLE  _moestream_ggml_clone_err
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_STRIP_TRAILING_WHITESPACE
            )
            if(NOT _moestream_ggml_clone_result EQUAL 0)
                message(FATAL_ERROR
                    "Failed to clone ggml (git returned ${_moestream_ggml_clone_result}).\n"
                    "Error: ${_moestream_ggml_clone_err}\n"
                    "Set MOESTREAM_GGML_ROOT to a local checkout, or verify network/git access.")
            endif()
        endif()
        set(MOESTREAM_GGML_ROOT "${_moestream_ggml_src}")
    endif()

    # Apply MoEStream's ggml patch series (patches/README.md).  Idempotent:
    # already-applied patches are detected by a reverse apply check.
    # Link an already-built ggml instead of compiling it in-tree. The prebuilt tree must come
    # from the same MOESTREAM_GGML_ROOT (its headers are used) and ggml patches are NOT re-applied.
    set(MOESTREAM_GGML_PREBUILT_DIR "" CACHE PATH "Directory holding prebuilt libggml*.so (skips building ggml)")
    if(MOESTREAM_GGML_PREBUILT_DIR)
        foreach(_lib ggml ggml-base ggml-cpu)
            add_library(${_lib} SHARED IMPORTED GLOBAL)
            set_target_properties(${_lib} PROPERTIES
                IMPORTED_LOCATION "${MOESTREAM_GGML_PREBUILT_DIR}/src/lib${_lib}.so"
                INTERFACE_INCLUDE_DIRECTORIES "${MOESTREAM_GGML_ROOT}/include")
        endforeach()
        set_property(TARGET ggml PROPERTY INTERFACE_LINK_LIBRARIES "ggml-cpu;ggml-base")
        set_property(TARGET ggml-cpu PROPERTY INTERFACE_LINK_LIBRARIES "ggml-base")
        add_library(ggml-cuda SHARED IMPORTED GLOBAL)
        set_target_properties(ggml-cuda PROPERTIES
            IMPORTED_LOCATION "${MOESTREAM_GGML_PREBUILT_DIR}/src/ggml-cuda/libggml-cuda.so"
            INTERFACE_INCLUDE_DIRECTORIES "${MOESTREAM_GGML_ROOT}/include"
            INTERFACE_LINK_LIBRARIES "ggml-base")
        message(STATUS "MoEStream: linking PREBUILT ggml from ${MOESTREAM_GGML_PREBUILT_DIR} (no ggml compile)")
    else()
        include(cmake/MoEStreamGgmlPatches.cmake)
        moestream_apply_ggml_patches("${MOESTREAM_GGML_ROOT}")
        add_subdirectory("${MOESTREAM_GGML_ROOT}" ggml-build)
    endif()
    message(STATUS "MoEStream: using ggml at ${MOESTREAM_GGML_ROOT}")

    # ggml-quants.h lives in ggml/src/, not in the public include/ directory.
    # Expose it to our bridge translation unit.
    set(MOESTREAM_GGML_SRC_DIR "${MOESTREAM_GGML_ROOT}/src")
endif()

# ---------------------------------------------------------------------------
# Common include path
# ---------------------------------------------------------------------------
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---------------------------------------------------------------------------
# Full engine build (Linux + CUDA Toolkit)
# ---------------------------------------------------------------------------
if(NOT MOESTREAM_CPU_ONLY)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    find_package(CUDAToolkit REQUIRED)

    # Engine executable
    # CUDA kernel sources (cuda/*.cu) and the GPU Qwen MoE backend
    # (src/qwen_moe.cpp, "compiled into the main moestream binary (CUDA
    # builds only)" per its own header comment) were not referenced by
    # ANY target here -- moestream/moestream_tests linked without them and
    # failed at link time with undefined references to every launch_*
    # kernel wrapper (cuda_manager.hpp) and QwenMoEBackend's vtable. Shared
    # between both executables the same way models/model_factory.cpp
    # already is, since both need the identical GPU dispatch surface.
    set(MOESTREAM_CUDA_SRCS
        cuda/exl3_aux.cu
        cuda/attn_score_softmax.cu
        cuda/dsv4_score_softmax.cu
        cuda/dequant_bf16.cu
        cuda/expert_accum.cu
        cuda/expert_activation.cu
        cuda/gdn_conv1d.cu
        cuda/glm5next_kda.cu
        cuda/glm5next_router.cu
        cuda/ggml_moe_pack.cu
        cuda/iq1_dequant.cu
        cuda/iq2_matmul.cu
        cuda/iq3_matmul.cu
        cuda/iq4xs_matmul.cu
        cuda/kquant_matmul.cu
        cuda/matmul.cu
        cuda/mla_attention.cu
        cuda/mxfp4_matmul.cu
        cuda/q5_1_matmul.cu
        cuda/qwen4exp_master_graph.cu
        cuda/qwen4exp_hc_q8.cu
        cuda/q8_0_matmul.cu
        cuda/q2_0_matmul.cu
        cuda/rmsnorm.cu
        cuda/wuv_int8_dequant.cu
    )
    # The Layer-13 predictive staging spike is an optional worktree feature.
    # Keep a master checkout buildable when its files are absent, while still
    # compiling them automatically on the preserved spike branch.
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cuda/qwen4exp_layer13_staging.cu")
        list(APPEND MOESTREAM_CUDA_SRCS cuda/qwen4exp_layer13_staging.cu)
    endif()

    # Upstream ExLlamaV3 EXL3 kernels behind the launch_exl3_mul1_* API.
    # Runtime opt-out: MOESTREAM_EXL3_UPSTREAM=0 (legacy kernels).
    include(cmake/MoEStreamExl3Upstream.cmake)
    if(TARGET moestream_exl3_upstream)
        list(APPEND MOESTREAM_CUDA_SRCS src/exl3_upstream_ops.cu)
        set_source_files_properties(src/exl3_upstream_ops.cu PROPERTIES
            INCLUDE_DIRECTORIES "${MOESTREAM_EXLLAMAV3_SRC}")
        set_source_files_properties(cuda/exl3_aux.cu src/exl3_gpu.cpp
            src/exl3_ggml_ffn.cpp src/exl3_upstream_ops.cu
            PROPERTIES COMPILE_DEFINITIONS MOESTREAM_EXL3_UPSTREAM=1)
        set(MOESTREAM_EXL3_UPSTREAM_LIB moestream_exl3_upstream)
        set(MOESTREAM_EXL3_UPSTREAM_OPS src/exl3_upstream_ops.cu)
    endif()

    set(MOESTREAM_OPTIONAL_SRCS)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/qwen4exp_layer13_staging.cpp")
        list(APPEND MOESTREAM_OPTIONAL_SRCS src/qwen4exp_layer13_staging.cpp)
    endif()

    add_executable(moestream
        src/main.cpp
        src/execution_layer.cpp
        src/exl3_cpu.cpp
        src/exl3_model_loader.cpp
        src/exl3_decode.cpp
        src/exl3_gpu.cpp
        src/cpu_dense.cpp
        src/cpu_engine.cpp
        src/cpu_forward.cpp
        src/cpu_model_backend.cpp
        src/cpu_qwen_moe.cpp
        src/cpu_qwen4exp.cpp
        src/scratch_buffers.cpp
        src/cuda_kernel_dispatcher.cpp
        src/gpu_expert_executor.cpp
        src/cpu_expert_executor.cpp
        src/moe_router.cpp
        src/moe_pipeline.cpp
        src/moe_pipeline_orchestrator.cpp
        src/moe_router_service.cpp
        src/prefill_warmup_manager.cpp
        src/batched_dispatch_pipeline.cpp
        src/token_dispatch_pipeline.cpp
        src/shared_expert_executor.cpp
        ${MOESTREAM_OPTIONAL_SRCS}
        src/attention_layer.cpp
        src/graph_initializer.cpp
        src/hyper_connections.cpp
        src/ple_ngram.cpp
        src/safetensors_adapter.cpp
        src/deepseek41_engram.cpp
        src/gpu_topology.cpp
        src/ggml_dequant.cpp
        src/ggml_grids.cpp
        src/ggml_moe_backend.cpp
        src/ggml_moe_graph_builder.cpp
        src/ggml_moe_transfer_manager.cpp
        src/ggml_moe_runner.cpp
        src/ggml_router_runner.cpp
        src/ggml_qwen_shared_runner.cpp
        src/qwen_cpu_offload_ring.cpp
        src/ggml_qwen_decode_scheduler.cpp
        src/ggml_qwen_cpu_moe.cpp
        src/ggml_qwen4exp_graph.cpp
        src/ggml_glm5next_graph.cpp
        src/ggml_hc_backend.cpp
        src/ggml_gdn_proj_backend.cpp
        src/ggml_glm5next_proj_backend.cpp
        src/mtp_speculative.cpp
        src/kernels_cpu.cpp
        src/qwen_moe.cpp
        models/model_factory.cpp
        models/family_ops.cpp
        ${MOESTREAM_CUDA_SRCS}
    )
    target_include_directories(moestream PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/models
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda
    )
    target_compile_options(moestream PRIVATE ${MOESTREAM_AVX512_FLAGS})
    target_compile_definitions(moestream PRIVATE MOESTREAM_EXL3_CUDA=1)
    set_target_properties(moestream PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_ARCHITECTURES 86        # Ampere; adjust via -DCMAKE_CUDA_ARCHITECTURES
    )
    target_link_libraries(moestream PRIVATE ${MOESTREAM_EXL3_UPSTREAM_LIB}
        CUDA::cudart
        CUDA::cublas
        CUDA::cublasLt
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(moestream PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(MOESTREAM_USE_GGML)
        target_link_libraries(moestream PRIVATE ggml)
        if(MOESTREAM_USE_GGML_CUDA)
            target_link_libraries(moestream PRIVATE ggml-cuda)
            # Gates include/ggml_moe_backend.hpp's real implementation. Without
            # it that file still compiles, but every entry point returns false
            # and execution_layer.cpp falls back to its own dispatch.
            target_compile_definitions(moestream PRIVATE MOESTREAM_USE_GGML_CUDA=1)
        endif()
        target_compile_definitions(moestream PRIVATE MOESTREAM_USE_GGML=1)
        target_include_directories(moestream PRIVATE ${MOESTREAM_GGML_SRC_DIR})
        if(MOESTREAM_USE_GGML_CUDA)
            target_sources(moestream PRIVATE
                src/exl3_ggml_backend.cpp
                src/exl3_ggml_ffn.cpp
                src/exl3_ggml_model.cpp)
            target_include_directories(moestream PRIVATE ${MOESTREAM_GGML_SRC_DIR}/src)
        endif()
    endif()

    # Synthetic parity: legacy EXL3 kernel vs upstream exl3_gemm (no model needed).
    if(TARGET moestream_exl3_upstream)
        add_executable(exl3_upstream_parity EXCLUDE_FROM_ALL
            tests/exl3_upstream_parity.cu
            tests/exl3_reference_matmul.cu)
        target_include_directories(exl3_upstream_parity PRIVATE
            include "${MOESTREAM_EXLLAMAV3_SRC}")
        target_link_libraries(exl3_upstream_parity PRIVATE
            moestream_exl3_upstream CUDA::cudart)
        set_target_properties(exl3_upstream_parity PROPERTIES CUDA_STANDARD 17)
    endif()

    # Native test harness
    add_executable(moestream_tests
        src/main_test.cpp
        src/exl3_cpu.cpp
        src/exl3_gpu.cpp
        src/execution_layer.cpp
        src/scratch_buffers.cpp
        src/cuda_kernel_dispatcher.cpp
        src/gpu_expert_executor.cpp
        src/cpu_expert_executor.cpp
        src/moe_router.cpp
        src/moe_pipeline.cpp
        src/moe_pipeline_orchestrator.cpp
        src/moe_router_service.cpp
        src/prefill_warmup_manager.cpp
        src/batched_dispatch_pipeline.cpp
        src/token_dispatch_pipeline.cpp
        src/shared_expert_executor.cpp
        ${MOESTREAM_OPTIONAL_SRCS}
        src/attention_layer.cpp
        src/graph_initializer.cpp
        src/hyper_connections.cpp
        src/ple_ngram.cpp
        src/deepseek41_engram.cpp
        src/gpu_topology.cpp
        src/ggml_dequant.cpp
        src/ggml_grids.cpp
        src/ggml_moe_backend.cpp
        src/ggml_moe_graph_builder.cpp
        src/ggml_moe_transfer_manager.cpp
        src/ggml_moe_runner.cpp
        src/ggml_router_runner.cpp
        src/ggml_qwen_shared_runner.cpp
        src/qwen_cpu_offload_ring.cpp
        src/ggml_qwen_decode_scheduler.cpp
        src/ggml_qwen_cpu_moe.cpp
        src/ggml_qwen4exp_graph.cpp
        src/ggml_glm5next_graph.cpp
        src/ggml_hc_backend.cpp
        src/ggml_gdn_proj_backend.cpp
        src/ggml_glm5next_proj_backend.cpp
        src/mtp_speculative.cpp
        src/kernels_cpu.cpp
        src/qwen_moe.cpp
        models/model_factory.cpp
        models/family_ops.cpp
        ${MOESTREAM_CUDA_SRCS}
    )
    target_include_directories(moestream_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/models
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda
    )
    target_compile_options(moestream_tests PRIVATE ${MOESTREAM_AVX512_FLAGS})
    set_target_properties(moestream_tests PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_ARCHITECTURES 86
    )
    target_link_libraries(moestream_tests PRIVATE ${MOESTREAM_EXL3_UPSTREAM_LIB}
        CUDA::cudart
        CUDA::cublas
        CUDA::cublasLt
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(moestream_tests PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(MOESTREAM_USE_GGML)
        target_link_libraries(moestream_tests PRIVATE ggml)
        if(MOESTREAM_USE_GGML_CUDA)
            target_link_libraries(moestream_tests PRIVATE ggml-cuda)
            target_compile_definitions(moestream_tests PRIVATE MOESTREAM_USE_GGML_CUDA=1)
        endif()
        target_compile_definitions(moestream_tests PRIVATE MOESTREAM_USE_GGML=1)
        target_include_directories(moestream_tests PRIVATE ${MOESTREAM_GGML_SRC_DIR})
        if(MOESTREAM_USE_GGML_CUDA)
            target_sources(moestream_tests PRIVATE
                src/exl3_ggml_backend.cpp
                src/exl3_ggml_ffn.cpp
                src/exl3_ggml_model.cpp
                src/exl3_model_loader.cpp
                src/safetensors_adapter.cpp)
            target_include_directories(moestream_tests PRIVATE ${MOESTREAM_GGML_SRC_DIR}/src)
        endif()
    endif()

    # GGML-only Qwen CPU MoE timing probe.  The same source is also compiled
    # by bench/run_qwen_cpu_moe_compare.sh against llama.cpp's GGML shared
    # libraries, giving a layer-graph comparison without loading a model.
    if(MOESTREAM_USE_GGML)
        add_executable(qwen_cpu_moe_ggml_bench
            bench/qwen_cpu_moe_ggml_bench.cpp)
        target_include_directories(qwen_cpu_moe_ggml_bench PRIVATE
            ${MOESTREAM_GGML_SRC_DIR})
        target_compile_options(qwen_cpu_moe_ggml_bench PRIVATE
            ${MOESTREAM_AVX512_FLAGS})
        target_link_libraries(qwen_cpu_moe_ggml_bench PRIVATE ggml)
    endif()

    add_executable(exl3_kernel_bench EXCLUDE_FROM_ALL
        bench/exl3_kernel_bench.cu
        cuda/exl3_aux.cu
        ${MOESTREAM_EXL3_UPSTREAM_OPS})
    target_include_directories(exl3_kernel_bench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set_target_properties(exl3_kernel_bench PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_ARCHITECTURES 86)
    target_link_libraries(exl3_kernel_bench PRIVATE CUDA::cudart)

    if(MOESTREAM_USE_GGML AND MOESTREAM_USE_GGML_CUDA)
        add_executable(exl3_ggml_custom_bench EXCLUDE_FROM_ALL
            bench/exl3_ggml_custom_bench.cpp
            cuda/exl3_aux.cu
            ${MOESTREAM_EXL3_UPSTREAM_OPS})
        target_include_directories(exl3_ggml_custom_bench PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${MOESTREAM_GGML_SRC_DIR})
        target_compile_definitions(exl3_ggml_custom_bench PRIVATE
            MOESTREAM_USE_GGML=1 MOESTREAM_USE_GGML_CUDA=1)
        set_target_properties(exl3_ggml_custom_bench PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF CUDA_ARCHITECTURES 86)
        target_link_libraries(exl3_ggml_custom_bench PRIVATE
            CUDA::cudart CUDA::cublas CUDA::cublasLt ggml ggml-cuda)

        add_executable(exl3_ggml_cuda_adapter_bench EXCLUDE_FROM_ALL
            bench/exl3_ggml_cuda_adapter_bench.cpp
            src/exl3_ggml_backend.cpp
            cuda/exl3_aux.cu
            ${MOESTREAM_EXL3_UPSTREAM_OPS})
        target_include_directories(exl3_ggml_cuda_adapter_bench PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${MOESTREAM_GGML_SRC_DIR}
            ${MOESTREAM_GGML_SRC_DIR}/src)
        target_compile_definitions(exl3_ggml_cuda_adapter_bench PRIVATE
            MOESTREAM_USE_GGML=1 MOESTREAM_USE_GGML_CUDA=1)
        set_target_properties(exl3_ggml_cuda_adapter_bench PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF CUDA_ARCHITECTURES 86)
        target_link_libraries(exl3_ggml_cuda_adapter_bench PRIVATE
            CUDA::cudart CUDA::cublas CUDA::cublasLt ggml ggml-cuda)

        add_executable(exl3_ggml_cuda_real_bench EXCLUDE_FROM_ALL
            bench/exl3_ggml_cuda_real_bench.cpp
            src/exl3_ggml_backend.cpp
            src/exl3_model_loader.cpp
            src/safetensors_adapter.cpp
            src/exl3_cpu.cpp
            src/kernels_cpu.cpp
            src/ggml_grids.cpp
            cuda/exl3_aux.cu
            ${MOESTREAM_EXL3_UPSTREAM_OPS})
        target_include_directories(exl3_ggml_cuda_real_bench PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${MOESTREAM_GGML_SRC_DIR}
            ${MOESTREAM_GGML_SRC_DIR}/src)
        target_compile_options(exl3_ggml_cuda_real_bench PRIVATE
            ${MOESTREAM_AVX512_FLAGS})
        target_compile_definitions(exl3_ggml_cuda_real_bench PRIVATE
            MOESTREAM_USE_GGML=1 MOESTREAM_USE_GGML_CUDA=1)
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
            src/exl3_ggml_backend.cpp
            src/exl3_ggml_ffn.cpp
            src/exl3_gpu.cpp
            src/exl3_model_loader.cpp
            src/safetensors_adapter.cpp
            src/exl3_cpu.cpp
            src/kernels_cpu.cpp
            src/ggml_grids.cpp
            cuda/exl3_aux.cu
            ${MOESTREAM_EXL3_UPSTREAM_OPS})
        target_include_directories(exl3_ggml_cuda_real_ffn_bench PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${MOESTREAM_GGML_SRC_DIR}
            ${MOESTREAM_GGML_SRC_DIR}/src)
        target_compile_options(exl3_ggml_cuda_real_ffn_bench PRIVATE
            ${MOESTREAM_AVX512_FLAGS})
        target_compile_definitions(exl3_ggml_cuda_real_ffn_bench PRIVATE
            MOESTREAM_USE_GGML=1 MOESTREAM_USE_GGML_CUDA=1
            MOESTREAM_EXL3_CUDA=1)
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
            src/exl3_ggml_backend.cpp
            src/exl3_ggml_ffn.cpp
            src/exl3_gpu.cpp
            src/exl3_model_loader.cpp
            src/safetensors_adapter.cpp
            src/exl3_cpu.cpp
            src/kernels_cpu.cpp
            src/ggml_grids.cpp
            cuda/exl3_aux.cu
            ${MOESTREAM_EXL3_UPSTREAM_OPS})
        target_include_directories(exl3_ggml_cuda_real_routed_ffn_bench PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${MOESTREAM_GGML_SRC_DIR}
            ${MOESTREAM_GGML_SRC_DIR}/src)
        target_compile_options(exl3_ggml_cuda_real_routed_ffn_bench PRIVATE
            ${MOESTREAM_AVX512_FLAGS})
        target_compile_definitions(exl3_ggml_cuda_real_routed_ffn_bench PRIVATE
            MOESTREAM_USE_GGML=1 MOESTREAM_USE_GGML_CUDA=1
            MOESTREAM_EXL3_CUDA=1)
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
            src/exl3_ggml_backend.cpp
            src/exl3_ggml_ffn.cpp
            src/exl3_gpu.cpp
            src/exl3_model_loader.cpp
            src/safetensors_adapter.cpp
            src/exl3_cpu.cpp
            src/kernels_cpu.cpp
            src/ggml_grids.cpp
            cuda/exl3_aux.cu
            ${MOESTREAM_EXL3_UPSTREAM_OPS}
            ${MOESTREAM_EXL3_UPSTREAM_OPS})
        target_include_directories(exl3_ggml_cuda_cache_graph_bench PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${MOESTREAM_GGML_SRC_DIR}
            ${MOESTREAM_GGML_SRC_DIR}/src)
        target_compile_options(exl3_ggml_cuda_cache_graph_bench PRIVATE
            ${MOESTREAM_AVX512_FLAGS})
        target_compile_definitions(exl3_ggml_cuda_cache_graph_bench PRIVATE
            MOESTREAM_USE_GGML=1 MOESTREAM_USE_GGML_CUDA=1
            MOESTREAM_EXL3_CUDA=1)
        set_target_properties(exl3_ggml_cuda_cache_graph_bench PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF CUDA_ARCHITECTURES 86)
        target_link_libraries(exl3_ggml_cuda_cache_graph_bench PRIVATE
            CUDA::cudart CUDA::cublas CUDA::cublasLt ggml ggml-cuda
            ${MOESTREAM_EXL3_UPSTREAM_LIB})
        if(OpenMP_CXX_FOUND)
            target_link_libraries(exl3_ggml_cuda_cache_graph_bench PRIVATE
                OpenMP::OpenMP_CXX)
        endif()

        add_executable(exl3_ggml_cuda_fused_ffn_bench EXCLUDE_FROM_ALL
            bench/exl3_ggml_cuda_fused_ffn_bench.cpp
            cuda/exl3_aux.cu
            ${MOESTREAM_EXL3_UPSTREAM_OPS})
        target_include_directories(exl3_ggml_cuda_fused_ffn_bench PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${MOESTREAM_GGML_SRC_DIR}
            ${MOESTREAM_GGML_SRC_DIR}/src)
        target_compile_definitions(exl3_ggml_cuda_fused_ffn_bench PRIVATE
            MOESTREAM_USE_GGML=1 MOESTREAM_USE_GGML_CUDA=1)
        set_target_properties(exl3_ggml_cuda_fused_ffn_bench PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF CUDA_ARCHITECTURES 86)
        target_link_libraries(exl3_ggml_cuda_fused_ffn_bench PRIVATE
            CUDA::cudart CUDA::cublas CUDA::cublasLt ggml ggml-cuda)

        add_executable(exl3_ggml_cuda_moe_bench EXCLUDE_FROM_ALL
            bench/exl3_ggml_cuda_moe_bench.cpp
            cuda/exl3_aux.cu
            ${MOESTREAM_EXL3_UPSTREAM_OPS})
        target_include_directories(exl3_ggml_cuda_moe_bench PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${MOESTREAM_GGML_SRC_DIR}
            ${MOESTREAM_GGML_SRC_DIR}/src)
        target_compile_definitions(exl3_ggml_cuda_moe_bench PRIVATE
            MOESTREAM_USE_GGML=1 MOESTREAM_USE_GGML_CUDA=1)
        set_target_properties(exl3_ggml_cuda_moe_bench PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF CUDA_ARCHITECTURES 86)
        target_link_libraries(exl3_ggml_cuda_moe_bench PRIVATE
            CUDA::cudart CUDA::cublas CUDA::cublasLt ggml ggml-cuda)
    endif()

    if(MOESTREAM_USE_GGML AND OpenMP_CXX_FOUND)
        add_executable(exl3_cpu_bench EXCLUDE_FROM_ALL
            bench/exl3_cpu_bench.cpp
            src/exl3_cpu.cpp
            src/kernels_cpu.cpp
            src/ggml_grids.cpp)
        target_include_directories(exl3_cpu_bench PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${MOESTREAM_GGML_SRC_DIR}
            ${MOESTREAM_GGML_SRC_DIR}/src)
        target_compile_options(exl3_cpu_bench PRIVATE ${MOESTREAM_AVX512_FLAGS})
        target_compile_definitions(exl3_cpu_bench PRIVATE MOESTREAM_USE_GGML=1)
        target_link_libraries(exl3_cpu_bench PRIVATE ggml OpenMP::OpenMP_CXX)

        add_executable(exl3_hc_bench EXCLUDE_FROM_ALL
            bench/exl3_hc_bench.cpp
            src/exl3_gpu.cpp
            src/exl3_cpu.cpp
            src/kernels_cpu.cpp
            src/ggml_grids.cpp
            cuda/exl3_aux.cu
            ${MOESTREAM_EXL3_UPSTREAM_OPS})
        target_include_directories(exl3_hc_bench PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${MOESTREAM_GGML_SRC_DIR}
            ${MOESTREAM_GGML_SRC_DIR}/src)
        target_compile_options(exl3_hc_bench PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:${MOESTREAM_AVX512_FLAGS}>")
        target_compile_definitions(exl3_hc_bench PRIVATE MOESTREAM_USE_GGML=1)
        set_target_properties(exl3_hc_bench PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF CUDA_ARCHITECTURES 86)
        target_link_libraries(exl3_hc_bench PRIVATE
            ggml OpenMP::OpenMP_CXX CUDA::cudart CUDA::cublas)
    endif()

    # One-block IQ2_XXS CPU-vs-CUDA diagnostic.  This is deliberately a
    # separate opt-in executable: it reports the CPU Q8_K and CUDA Q8_1
    # activation quantization contracts plus the actual ggml_mul_mat_id
    # result, without changing the production dispatch path.
    if(MOESTREAM_USE_GGML_CUDA)
        add_executable(qwen_iq2_cpu_cuda_probe
            bench/qwen_iq2_cpu_cuda_probe.cpp
            src/ggml_grids.cpp)
        target_include_directories(qwen_iq2_cpu_cuda_probe PRIVATE
            ${MOESTREAM_GGML_SRC_DIR}
            ${MOESTREAM_GGML_SRC_DIR}/src)
        target_compile_options(qwen_iq2_cpu_cuda_probe PRIVATE
            ${MOESTREAM_AVX512_FLAGS})
        target_link_libraries(qwen_iq2_cpu_cuda_probe PRIVATE
            ggml ggml-cuda CUDA::cudart)
        target_compile_definitions(qwen_iq2_cpu_cuda_probe PRIVATE
            MOESTREAM_USE_GGML=1 MOESTREAM_USE_GGML_CUDA=1)
    endif()

    # Layer-13 pointer-indirection Phase-1 laboratory.  This target links
    # only the real IQ1_M/IQ4_NL production kernels, the device SwiGLU
    # activation, and the project's GGML lookup tables.  It intentionally
    # does not link the engine or mutate runtime dispatch; see the benchmark
    # source for the VRAM/mapped/H2D-stage conditions and M=1/3/6 contention
    # protocol.
    if(MOESTREAM_USE_GGML AND EXISTS
       "${CMAKE_CURRENT_SOURCE_DIR}/bench/qwen4exp_pointer_spike.cu")
        add_executable(qwen4exp_pointer_spike
            bench/qwen4exp_pointer_spike.cu
            cuda/iq1_dequant.cu
            cuda/iq4xs_matmul.cu
            cuda/expert_activation.cu
            src/ggml_grids.cpp)
        target_include_directories(qwen4exp_pointer_spike PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/cuda
            ${MOESTREAM_GGML_SRC_DIR})
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
            cuda/iq1_dequant.cu
            cuda/iq4xs_matmul.cu
            cuda/expert_activation.cu
            src/ggml_grids.cpp)
        target_include_directories(qwen4exp_staging_spike PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/cuda
            ${MOESTREAM_GGML_SRC_DIR})
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
    # REAL QwenCpuOffloadRing class (src/qwen_cpu_offload_ring.cpp) with a
    # synthetic CPU worker thread standing in for GgmlQwenCpuMoeRunner, to
    # isolate MOESTREAM_QWEN4EXP_RING_SPIN_US's effect on wait latency from
    # decode-loop noise. No GGML/model dependency.
    add_executable(qwen_cpu_offload_ring_spin_bench
        bench/qwen_cpu_offload_ring_spin_bench.cpp
        src/qwen_cpu_offload_ring.cpp)
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
    message(STATUS "MoEStream: CPU-only build path selected")

    # Choose the right CPU kernel backend.
    set(MOESTREAM_CPU_KERNEL_SRCS)
    # src/safetensors_adapter.cpp was never registered here even though
    # tests/test_components.cpp's test_safetensors_adapter() calls it
    # directly -- test_components (and any other target linking
    # kernels_cpu) failed to link with undefined SafetensorsAdapter symbols
    # regardless of platform, unrelated to Apple Silicon vs x86 below.
    list(APPEND MOESTREAM_CPU_KERNEL_SRCS
        src/exl3_cpu.cpp
        src/exl3_model_loader.cpp
        src/safetensors_adapter.cpp)
    if(MOESTREAM_APPLE_SILICON)
        list(APPEND MOESTREAM_CPU_KERNEL_SRCS src/kernels_cpu_neon.cpp)
        message(STATUS "MoEStream: using ARM NEON CPU kernel backend")
        if(MOESTREAM_METAL)
            list(APPEND MOESTREAM_CPU_KERNEL_SRCS src/kernels_metal.mm)
            message(STATUS "MoEStream: Metal/MPS dispatch enabled (falls back to NEON)")
        endif()
    else()
        list(APPEND MOESTREAM_CPU_KERNEL_SRCS src/kernels_cpu.cpp)
        message(STATUS "MoEStream: using x86 AVX-512 CPU kernel backend")
    endif()

    add_library(kernels_cpu STATIC ${MOESTREAM_CPU_KERNEL_SRCS})
    target_compile_options(kernels_cpu PRIVATE ${MOESTREAM_AVX512_FLAGS})
    target_include_directories(kernels_cpu PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/models
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(kernels_cpu PUBLIC OpenMP::OpenMP_CXX)
    endif()

    # Apple Silicon tuning: prefer the Apple-specific CPU tuning when the
    # toolchain supports it.  This benefits M3/M4/M5 (and is harmless on M1/M2).
    if(MOESTREAM_APPLE_SILICON)
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("-mcpu=apple-m3" MOESTREAM_HAS_APPLE_M3_FLAG)
        if(MOESTREAM_HAS_APPLE_M3_FLAG)
            target_compile_options(kernels_cpu PRIVATE -mcpu=apple-m3)
            message(STATUS "MoEStream: applied -mcpu=apple-m3 to CPU kernels")
        endif()
        target_compile_definitions(kernels_cpu PUBLIC MOESTREAM_ARM_NEON=1)
        if(MOESTREAM_METAL)
            set_source_files_properties(src/kernels_metal.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
            find_library(METAL_FRAMEWORK Metal REQUIRED)
            find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
            target_link_libraries(kernels_cpu PUBLIC ${METAL_FRAMEWORK} ${MPS_FRAMEWORK})
            target_compile_definitions(kernels_cpu PUBLIC MOESTREAM_METAL=1)
        endif()
    endif()

    if(MOESTREAM_USE_GGML)
        target_sources(kernels_cpu PRIVATE src/ggml_dequant.cpp)
        target_link_libraries(kernels_cpu PUBLIC ggml)
        target_compile_definitions(kernels_cpu PUBLIC MOESTREAM_USE_GGML=1)
        target_include_directories(kernels_cpu PUBLIC ${MOESTREAM_GGML_SRC_DIR})
    endif()

    # CPU-only component tests (cross-platform, no CUDA)
    add_executable(test_components tests/test_components.cpp)
    target_link_libraries(test_components PRIVATE kernels_cpu)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(test_components PRIVATE OpenMP::OpenMP_CXX)
    endif()

    # CPU-only inference binary (moestream_cpu)
    add_executable(moestream_cpu
        src/cpu_main.cpp
        src/exl3_decode.cpp
        src/cpu_dense.cpp
        src/cpu_engine.cpp
        src/cpu_forward.cpp
        src/cpu_model_backend.cpp
        src/cpu_qwen_moe.cpp
        src/cpu_qwen4exp.cpp
    )
    target_include_directories(moestream_cpu PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/models
    )
    target_link_libraries(moestream_cpu PRIVATE kernels_cpu)

    # When the kernel backend contains Objective-C++ (Metal dispatch), the
    # final executable must link the Objective-C runtime / Foundation so the
    # linker resolves ARC/runtime symbols from the static library objects.
    if(MOESTREAM_METAL AND MOESTREAM_APPLE_SILICON)
        find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
        foreach(_tgt test_components moestream_cpu)
            target_link_libraries(${_tgt} PRIVATE ${FOUNDATION_FRAMEWORK})
        endforeach()
    endif()
    if(OpenMP_CXX_FOUND)
        target_link_libraries(moestream_cpu PRIVATE OpenMP::OpenMP_CXX)
    endif()

    # Tools
    add_subdirectory(tools)
endif()

message(STATUS "MoEStream: configure done")

# EXL3 bench targets compile src/exl3_upstream_ops.cu directly and need the
# upstream kernel library plus the torch stub include path.
if(TARGET moestream_exl3_upstream)
    foreach(_exl3_target
            exl3_kernel_bench exl3_ggml_custom_bench exl3_ggml_cuda_adapter_bench
            exl3_ggml_cuda_real_bench exl3_ggml_cuda_real_ffn_bench
            exl3_ggml_cuda_real_routed_ffn_bench exl3_ggml_cuda_fused_ffn_bench
            exl3_ggml_cuda_moe_bench exl3_hc_bench)
        if(TARGET ${_exl3_target})
            target_link_libraries(${_exl3_target} PRIVATE moestream_exl3_upstream)
        endif()
    endforeach()
endif()

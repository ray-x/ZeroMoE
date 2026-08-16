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
# defined(__AVX512VNNI__)), and forcing that flag here would risk a
# SIGILL at runtime on a real AVX-512 CPU that lacks VNNI specifically
# (some early Skylake-X/Cascade Lake parts) -- the source's own ifdef
# already makes it opt-in per-host, no build flag needed to unlock it.
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

    # Disable everything we do not need for a CPU decode assistant.
    # These must be set BEFORE the subproject is configured.
    set(GGML_CUDA        OFF CACHE BOOL "" FORCE)
    set(GGML_HIPBLAS     OFF CACHE BOOL "" FORCE)
    set(GGML_VULKAN      OFF CACHE BOOL "" FORCE)
    set(GGML_SYCL        OFF CACHE BOOL "" FORCE)
    set(GGML_OPENMP      ${OpenMP_CXX_FOUND} CACHE BOOL "" FORCE)

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

    add_subdirectory("${MOESTREAM_GGML_ROOT}" ggml-build)
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
        cuda/attn_score_softmax.cu
        cuda/gguf_cuda.cu
        cuda/iq1_dequant.cu
        cuda/iq2_matmul.cu
        cuda/iq2m_dequant.cu
        cuda/iq3_matmul.cu
        cuda/iq4xs_matmul.cu
        cuda/kquant_matmul.cu
        cuda/kv_quant.cu
        cuda/matmul.cu
        cuda/mla_attention.cu
        cuda/mxfp4_matmul.cu
        cuda/q8_0_matmul.cu
        cuda/rmsnorm.cu
    )

    add_executable(moestream
        src/main.cpp
        src/execution_layer.cpp
        src/attention_layer.cpp
        src/graph_initializer.cpp
        src/hyper_connections.cpp
        src/gpu_topology.cpp
        src/ggml_dequant.cpp
        src/ggml_grids.cpp
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
    set_target_properties(moestream PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_ARCHITECTURES 86        # Ampere; adjust via -DCMAKE_CUDA_ARCHITECTURES
    )
    target_link_libraries(moestream PRIVATE
        CUDA::cudart
        CUDA::cublas
        CUDA::cublasLt
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(moestream PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(MOESTREAM_USE_GGML)
        target_link_libraries(moestream PRIVATE ggml)
        target_compile_definitions(moestream PRIVATE MOESTREAM_USE_GGML=1)
        target_include_directories(moestream PRIVATE ${MOESTREAM_GGML_SRC_DIR})
    endif()

    # Native test harness
    add_executable(moestream_tests
        src/main_test.cpp
        src/execution_layer.cpp
        src/attention_layer.cpp
        src/graph_initializer.cpp
        src/hyper_connections.cpp
        src/gpu_topology.cpp
        src/ggml_dequant.cpp
        src/ggml_grids.cpp
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
    target_link_libraries(moestream_tests PRIVATE
        CUDA::cudart
        CUDA::cublas
        CUDA::cublasLt
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(moestream_tests PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(MOESTREAM_USE_GGML)
        target_link_libraries(moestream_tests PRIVATE ggml)
        target_compile_definitions(moestream_tests PRIVATE MOESTREAM_USE_GGML=1)
        target_include_directories(moestream_tests PRIVATE ${MOESTREAM_GGML_SRC_DIR})
    endif()

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
        src/cpu_engine.cpp
        src/cpu_forward.cpp
        src/cpu_model_backend.cpp
        src/cpu_qwen_moe.cpp
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

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
# OpenMP (used by CPU kernels and some reference loops)
# ---------------------------------------------------------------------------
find_package(OpenMP)

# ---------------------------------------------------------------------------
# ggml integration (optional)
# ---------------------------------------------------------------------------
if(MOESTREAM_USE_GGML)
    set(MOESTREAM_GGML_TAG "b3923" CACHE STRING "ggml git tag to fetch")
    set(MOESTREAM_GGML_ROOT "" CACHE PATH "Path to a local ggml checkout (uses FetchContent if empty)")

    # Provide a deprecated alias for configs that still set it.
    if(MOESTREAM_LLAMA_CPP_ROOT AND NOT MOESTREAM_GGML_ROOT)
        set(MOESTREAM_GGML_ROOT "${MOESTREAM_LLAMA_CPP_ROOT}" CACHE PATH "" FORCE)
        message(WARNING "MOESTREAM_LLAMA_CPP_ROOT is deprecated; use MOESTREAM_GGML_ROOT")
    endif()
    if(MOESTREAM_LLAMA_CPP_TAG AND MOESTREAM_GGML_TAG STREQUAL "b3923")
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
        set(GGML_METAL ON CACHE BOOL "" FORCE)
    else()
        set(GGML_METAL OFF CACHE BOOL "" FORCE)
    endif()

    if(MOESTREAM_GGML_ROOT)
        add_subdirectory("${MOESTREAM_GGML_ROOT}" ggml-build)
        message(STATUS "MoEStream: using local ggml at ${MOESTREAM_GGML_ROOT}")
    else()
        include(FetchContent)
        FetchContent_Declare(
            ggml
            GIT_REPOSITORY https://github.com/ggerganov/ggml.git
            GIT_TAG        ${MOESTREAM_GGML_TAG}
            GIT_SHALLOW    TRUE
            GIT_SUBMODULES ""   # avoid fragile recursive submodule handling in restricted envs
        )
        FetchContent_MakeAvailable(ggml)
        message(STATUS "MoEStream: ggml fetched and available")
    endif()
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
    add_executable(moestream
        src/main.cpp
        src/execution_layer.cpp
        src/attention_layer.cpp
        src/graph_initializer.cpp
        src/gpu_topology.cpp
        src/ggml_dequant.cpp
        src/ggml_grids.cpp
        src/kernels_cpu.cpp
        models/model_factory.cpp
    )
    target_include_directories(moestream PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/models
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda
    )
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
    endif()

    # Native test harness
    add_executable(moestream_tests
        src/main_test.cpp
        src/execution_layer.cpp
        src/attention_layer.cpp
        src/graph_initializer.cpp
        src/gpu_topology.cpp
        src/ggml_dequant.cpp
        src/ggml_grids.cpp
        src/kernels_cpu.cpp
        models/model_factory.cpp
    )
    target_include_directories(moestream_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/models
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda
    )
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
    else()
        list(APPEND MOESTREAM_CPU_KERNEL_SRCS src/kernels_cpu.cpp)
        message(STATUS "MoEStream: using x86 AVX-512 CPU kernel backend")
    endif()

    add_library(kernels_cpu STATIC ${MOESTREAM_CPU_KERNEL_SRCS})
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
    endif()

    if(MOESTREAM_USE_GGML)
        target_sources(kernels_cpu PRIVATE src/ggml_dequant.cpp)
        target_link_libraries(kernels_cpu PUBLIC ggml)
        target_compile_definitions(kernels_cpu PUBLIC MOESTREAM_USE_GGML=1)
    endif()

    # CPU-only component tests (cross-platform, no CUDA)
    add_executable(test_components tests/test_components.cpp)
    target_link_libraries(test_components PRIVATE kernels_cpu)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(test_components PRIVATE OpenMP::OpenMP_CXX)
    endif()

    # Tools
    add_subdirectory(tools)
endif()

message(STATUS "MoEStream: configure done")

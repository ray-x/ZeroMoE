cmake_minimum_required(VERSION 3.23)
project(MoEStream LANGUAGES CXX CUDA)

# Force Host C++ Compiler to C++23 (Enables std::expected)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force NVIDIA NVCC Compiler to C++17 (NVCC lacks C++23 support)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    # Optimized compilation flags with unrolling and native vectorization targets
    add_compile_options(-O3 -march=native -mavx512f -mavx512vnni -funroll-loops -Wall -Wextra)
endif()

# Multi-architecture: Ampere (3090), Ada (4090), Hopper, Blackwell (5060 Ti)
# Older CUDA toolkits will ignore unsupported architectures
set(CMAKE_CUDA_ARCHITECTURES "86;89")
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.8")
    list(APPEND CMAKE_CUDA_ARCHITECTURES "100" "120")
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler -O3")

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# ---- Import ggml (sibling llama.cpp checkout) for its reference dequantize_row_* ----
# We LINK libggml for the K-quant/IQ dequant instead of copying that code. Built
# CPU-only: MoEStream provides its own CUDA kernels, so ggml's CUDA/BLAS backends
# are disabled to avoid nvcc/arch conflicts. Only src/ggml_dequant.cpp includes
# ggml headers; the rest of the code is isolated from ggml's type names.
set(GGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/ggml" CACHE PATH "Path to ggml sources")
set(GGML_STANDALONE     OFF CACHE BOOL "" FORCE)
set(GGML_CUDA           OFF CACHE BOOL "" FORCE)
set(GGML_CUDA_F16       OFF CACHE BOOL "" FORCE)
set(GGML_BLAS           OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP         OFF CACHE BOOL "" FORCE)  # MoEStream links its own OpenMP
set(GGML_METAL          OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${GGML_DIR} ${CMAKE_BINARY_DIR}/ggml-build EXCLUDE_FROM_ALL)

add_executable(moestream
    src/main.cpp
    src/execution_layer.cpp
    src/attention_layer.cpp
    src/graph_initializer.cpp
    src/kernels_cpu.cpp
    src/ggml_dequant.cpp
    src/ggml_grids.cpp
    models/model_factory.cpp
    cuda/gguf_cuda.cu
    cuda/matmul.cu
    cuda/q8_0_matmul.cu
    cuda/kquant_matmul.cu
    cuda/iq2_matmul.cu
    cuda/iq3_matmul.cu
    cuda/rmsnorm.cu
    cuda/iq2m_dequant.cu
    cuda/iq1_dequant.cu
    cuda/mla_attention.cu
    cuda/kv_quant.cu
)

target_include_directories(moestream PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/models
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
    ${GGML_DIR}/include
    ${GGML_DIR}/src
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(moestream PRIVATE
    OpenMP::OpenMP_CXX
    CUDA::cudart
    CUDA::cublas
    ggml
)

# ---- Native Test Harness (src/main_test.cpp) ----
# Links against the REAL kernels/graph code (not a re-implementation), so it
# needs the same .cpp/.cu sources as the `moestream` binary above, minus
# src/main.cpp (which defines its own main() and would collide with
# main_test.cpp's). No global state from main.cpp is referenced by any other
# translation unit, so this split is safe. Self-skips CUDA-dependent cases at
# runtime when no device is present (see CudaManager::initialize() call
# sites in main_test.cpp).
enable_testing()

add_executable(moestream_tests
    src/main_test.cpp
    src/execution_layer.cpp
    src/attention_layer.cpp
    src/graph_initializer.cpp
    src/kernels_cpu.cpp
    src/ggml_dequant.cpp
    src/ggml_grids.cpp
    models/model_factory.cpp
    cuda/gguf_cuda.cu
    cuda/matmul.cu
    cuda/q8_0_matmul.cu
    cuda/kquant_matmul.cu
    cuda/iq2_matmul.cu
    cuda/iq3_matmul.cu
    cuda/rmsnorm.cu
    cuda/iq2m_dequant.cu
    cuda/iq1_dequant.cu
    cuda/mla_attention.cu
    cuda/kv_quant.cu
)

target_include_directories(moestream_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/models
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
    ${GGML_DIR}/include
    ${GGML_DIR}/src
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(moestream_tests PRIVATE
    OpenMP::OpenMP_CXX
    CUDA::cudart
    CUDA::cublas
    ggml
)

add_test(NAME native_test_harness COMMAND moestream_tests)

message(STATUS "MoEStream Build Configuration:")
message(STATUS "  -> Host C++ Standard: C++23 Enabled")
message(STATUS "  -> Device CUDA Standard: C++17 Enabled")
message(STATUS "  -> CPU Vectorization: AVX-512 / VNNI Enabled")
message(STATUS "  -> GPU Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  -> Multi-GPU: Topology auto-detection enabled")
message(STATUS "  -> OpenMP Threading: Enabled")
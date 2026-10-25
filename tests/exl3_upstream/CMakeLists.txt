# Standalone build of the upstream-EXL3 parity test (no ggml needed):
#   cmake -S tests/exl3_upstream -B build_exl3_upstream \
#         -DZEROMOE_EXLLAMAV3_SRC=<exllamav3>/exllamav3/exllamav3_ext -DCMAKE_CUDA_ARCHITECTURES=86
cmake_minimum_required(VERSION 3.24)
project(exl3_upstream_parity LANGUAGES CXX CUDA)
set(ZEROMOE_REPO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")
find_package(CUDAToolkit REQUIRED)
include("${ZEROMOE_REPO_DIR}/cmake/ZeroMoEExl3Upstream.cmake")
add_executable(exl3_upstream_parity
    ../exl3_upstream_parity.cu
    ../exl3_reference_matmul.cu)
target_include_directories(exl3_upstream_parity PRIVATE
    "${ZEROMOE_REPO_DIR}/backends/exl3" "${ZEROMOE_EXLLAMAV3_SRC}")
target_link_libraries(exl3_upstream_parity PRIVATE zeromoe_exl3_upstream CUDA::cudart)
set_target_properties(exl3_upstream_parity PROPERTIES CUDA_STANDARD 17)

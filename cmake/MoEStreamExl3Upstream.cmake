# Upstream ExLlamaV3 EXL3 kernels, built WITHOUT libtorch.
if(NOT MOESTREAM_REPO_DIR)
    set(MOESTREAM_REPO_DIR "${CMAKE_SOURCE_DIR}")
endif()
#
# MOESTREAM_EXLLAMAV3_SRC points at exllamav3/exllamav3_ext of an ExLlamaV3
# checkout. Its sources are compiled unmodified; third_party/exl3_torch_stub
# supplies the few ATen/c10 symbols the launchers use (non-owning tensor views,
# TORCH_CHECK, current-stream/device guards).
#
# Provides: target moestream_exl3_upstream (static, CUDA).
set(MOESTREAM_EXLLAMAV3_SRC "" CACHE PATH
    "Path to exllamav3/exllamav3_ext (enables the upstream EXL3 kernels)")

if(MOESTREAM_EXLLAMAV3_SRC AND NOT TARGET moestream_exl3_upstream)
    set(_ext "${MOESTREAM_EXLLAMAV3_SRC}")
    if(NOT EXISTS "${_ext}/quant/exl3_gemm.cu")
        message(FATAL_ERROR "MOESTREAM_EXLLAMAV3_SRC=${_ext}: quant/exl3_gemm.cu not found")
    endif()
    file(GLOB _units
        "${_ext}/quant/comp_units/exl3_comp_unit_*.cu"
        "${_ext}/quant/comp_units/exl3_gemv_half_inst.cu"
        "${_ext}/quant/comp_units/exl3_gemv_int8_inst_*.cu"
        "${_ext}/quant/comp_units/exl3_moe_coop_inst_*.cu")
    add_library(moestream_exl3_upstream STATIC
        ${_units}
        "${_ext}/quant/exl3_gemm.cu"
        "${_ext}/quant/exl3_gemv.cu"
        "${_ext}/quant/exl3_gemv_int8.cu"
        "${_ext}/quant/exl3_devctx.cu"
        "${_ext}/quant/exl3_kernel_map.cu"
        "${_ext}/quant/coop_autotune.cu"
        "${_ext}/quant/exl3_moe_coop.cu"
        "${MOESTREAM_REPO_DIR}/backends/exl3/exl3_upstream_stream.cpp")
    target_include_directories(moestream_exl3_upstream
        PUBLIC "${MOESTREAM_REPO_DIR}/third_party/exl3_torch_stub"
        PRIVATE "${_ext}")
    set_target_properties(moestream_exl3_upstream PROPERTIES
        CUDA_STANDARD 17 CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON)
    target_compile_options(moestream_exl3_upstream PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    target_link_libraries(moestream_exl3_upstream PUBLIC CUDA::cudart CUDA::cuda_driver)
    message(STATUS "MoEStream: upstream EXL3 kernels from ${_ext} (${CMAKE_CURRENT_LIST_FILE})")
endif()

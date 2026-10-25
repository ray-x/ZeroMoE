# Upstream ExLlamaV3 EXL3 kernels, built WITHOUT libtorch.
if(NOT ZEROMOE_REPO_DIR)
    set(ZEROMOE_REPO_DIR "${CMAKE_SOURCE_DIR}")
endif()
#
# ZEROMOE_EXLLAMAV3_SRC points at exllamav3/exllamav3_ext of an ExLlamaV3
# checkout. Its sources are compiled unmodified; third_party/exl3_torch_stub
# supplies the few ATen/c10 symbols the launchers use (non-owning tensor views,
# TORCH_CHECK, current-stream/device guards).
#
# Provides: target zeromoe_exl3_upstream (static, CUDA).
set(ZEROMOE_EXLLAMAV3_SRC "" CACHE PATH
    "Path to exllamav3/exllamav3_ext (enables the upstream EXL3 kernels)")

if(ZEROMOE_EXLLAMAV3_SRC AND NOT TARGET zeromoe_exl3_upstream)
    set(_ext "${ZEROMOE_EXLLAMAV3_SRC}")
    if(NOT EXISTS "${_ext}/quant/exl3_gemm.cu")
        message(FATAL_ERROR "ZEROMOE_EXLLAMAV3_SRC=${_ext}: quant/exl3_gemm.cu not found")
    endif()
    file(GLOB _units
        "${_ext}/quant/comp_units/exl3_comp_unit_*.cu"
        "${_ext}/quant/comp_units/exl3_gemv_half_inst.cu"
        "${_ext}/quant/comp_units/exl3_gemv_int8_inst_*.cu"
        "${_ext}/quant/comp_units/exl3_moe_coop_inst_*.cu")
    add_library(zeromoe_exl3_upstream STATIC
        ${_units}
        "${_ext}/quant/exl3_gemm.cu"
        "${_ext}/quant/exl3_gemv.cu"
        "${_ext}/quant/exl3_gemv_int8.cu"
        "${_ext}/quant/exl3_devctx.cu"
        "${_ext}/quant/exl3_kernel_map.cu"
        "${_ext}/quant/coop_autotune.cu"
        "${_ext}/quant/exl3_moe_coop.cu"
        "${ZEROMOE_REPO_DIR}/backends/exl3/exl3_upstream_stream.cpp")
    target_include_directories(zeromoe_exl3_upstream
        PUBLIC "${ZEROMOE_REPO_DIR}/third_party/exl3_torch_stub"
        PRIVATE "${_ext}")
    set_target_properties(zeromoe_exl3_upstream PROPERTIES
        CUDA_STANDARD 17 CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON)
    target_compile_options(zeromoe_exl3_upstream PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    target_link_libraries(zeromoe_exl3_upstream PUBLIC CUDA::cudart CUDA::cuda_driver)
    message(STATUS "ZeroMoE: upstream EXL3 kernels from ${_ext} (${CMAKE_CURRENT_LIST_FILE})")
endif()

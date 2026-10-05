# Applies MoEStream's ggml patch series to the ggml checkout.  See
# patches/README.md for the series layout and the upstream base commit.
#
# Each patch is classified by `git apply --check`:
#   applies forward      -> apply it
#   applies in reverse   -> already applied, nothing to do (idempotent)
#   neither              -> tree diverged from the patch's base; reported, not applied
#
# Patch directories (applied in filename order):
#   patches/ggml                     always; REQUIRED (warns loudly if one cannot apply)
#   patches/ggml-host-node           MOESTREAM_ENABLE_GGML_HOST_NODE_PATCHES; REQUIRED
#   patches/ggml-host-node-beellama  MOESTREAM_ENABLE_GGML_HOST_NODE_PATCHES; best effort.
#                                    These use beellama-only scheduler profiling
#                                    helpers and do not apply to plain upstream.
#   patches/debug                    MOESTREAM_GGML_DEBUG_PATCHES (default OFF); best effort
# patches/experimental and patches/obsolete are never applied.

function(_moestream_ggml_apply_dir dir required)
    file(GLOB _patches "${CMAKE_CURRENT_SOURCE_DIR}/patches/${dir}/*.patch")
    list(SORT _patches)
    foreach(_patch IN LISTS _patches)
        get_filename_component(_name "${_patch}" NAME)
        execute_process(
            COMMAND git apply --whitespace=nowarn --check ${_MOESTREAM_APPLY_ARGS} "${_patch}"
            WORKING_DIRECTORY "${_MOESTREAM_APPLY_DIR}"
            RESULT_VARIABLE _fwd
            OUTPUT_QUIET ERROR_QUIET)
        if(_fwd EQUAL 0)
            execute_process(
                COMMAND git apply --whitespace=nowarn ${_MOESTREAM_APPLY_ARGS} "${_patch}"
                WORKING_DIRECTORY "${_MOESTREAM_APPLY_DIR}"
                RESULT_VARIABLE _rc
                ERROR_VARIABLE _err)
            if(NOT _rc EQUAL 0)
                message(FATAL_ERROR "MoEStream: failed to apply ggml patch ${dir}/${_name}:\n${_err}")
            endif()
            message(STATUS "MoEStream: applied ggml patch ${dir}/${_name}")
            continue()
        endif()
        execute_process(
            COMMAND git apply --whitespace=nowarn --check --reverse ${_MOESTREAM_APPLY_ARGS} "${_patch}"
            WORKING_DIRECTORY "${_MOESTREAM_APPLY_DIR}"
            RESULT_VARIABLE _rev
            OUTPUT_QUIET ERROR_QUIET)
        if(_rev EQUAL 0)
            continue()
        endif()
        if(required)
            message(WARNING
                "MoEStream: required ggml patch ${dir}/${_name} neither applies nor is "
                "already applied -- the ggml tree differs from the series base. On plain "
                "upstream expect link errors for the symbols it provides; a fork that "
                "carries the change natively (e.g. beellama) can ignore this.")
        else()
            message(STATUS "MoEStream: skipped optional ggml patch ${dir}/${_name} "
                           "(does not apply to this ggml tree)")
        endif()
    endforeach()
endfunction()

function(moestream_apply_ggml_patches ggml_root)
    # Some ggml providers (llama.cpp, beellama) keep ggml under a nested
    # repository prefix; git apply resolves paths from the provider's git root.
    execute_process(
        COMMAND git rev-parse --show-prefix
        WORKING_DIRECTORY "${ggml_root}"
        OUTPUT_VARIABLE _prefix
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    set(_MOESTREAM_APPLY_ARGS "")
    set(_MOESTREAM_APPLY_DIR "${ggml_root}")
    if(_prefix STREQUAL "ggml/")
        set(_MOESTREAM_APPLY_ARGS --directory=ggml)
        get_filename_component(_MOESTREAM_APPLY_DIR "${ggml_root}/.." ABSOLUTE)
    endif()

    _moestream_ggml_apply_dir(ggml TRUE)
    if(MOESTREAM_ENABLE_GGML_HOST_NODE_PATCHES)
        _moestream_ggml_apply_dir(ggml-host-node TRUE)
        _moestream_ggml_apply_dir(ggml-host-node-beellama FALSE)
    endif()
    if(MOESTREAM_GGML_DEBUG_PATCHES)
        _moestream_ggml_apply_dir(debug FALSE)
    endif()
endfunction()

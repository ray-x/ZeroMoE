# Applies ZeroMoE's ggml patch series to the ggml checkout.  See
# patches/README.md for the series layout and the upstream base commit.
#
# Each patch is classified by a GNU `patch --dry-run` (fuzz 0):
#   applies forward      -> apply it
#   applies in reverse   -> already applied, nothing to do (idempotent)
#   neither              -> tree diverged from the patch's base; reported, not applied
#
# Patch directories (applied in filename order):
#   patches/ggml                     always; REQUIRED (warns loudly if one cannot apply)
#   patches/ggml-host-node           ZEROMOE_ENABLE_GGML_HOST_NODE_PATCHES; REQUIRED
#   patches/ggml-host-node-beellama  ZEROMOE_ENABLE_GGML_HOST_NODE_PATCHES; best effort.
#                                    These use beellama-only scheduler profiling
#                                    helpers and do not apply to plain upstream.
#   patches/debug                    ZEROMOE_GGML_DEBUG_PATCHES (default OFF); best effort
# patches/experimental and patches/obsolete are never applied.

function(_zeromoe_ggml_apply_dir dir required)
    file(GLOB _patches "${CMAKE_CURRENT_SOURCE_DIR}/patches/${dir}/*.patch")
    list(SORT _patches)
    foreach(_patch IN LISTS _patches)
        get_filename_component(_name "${_patch}" NAME)
        # GNU patch, fuzz 0, run from the ggml directory (patches are rooted at it: a/src/...).
        # Hunks may move (offset) but their context must match exactly, so a hunk that no longer fits is
        # reported instead of landing in the wrong function. `git apply` is not used: inside a checkout of
        # another repository it silently skips files outside the current directory.
        # Already applied? Checked first: a patch that only adds lines can still dry-run forward on a
        # tree that has it, and applying it twice would corrupt the file.
        execute_process(
            COMMAND patch -R -p1 -F0 --dry-run -s
            INPUT_FILE "${_patch}"
            WORKING_DIRECTORY "${_ZEROMOE_APPLY_DIR}"
            RESULT_VARIABLE _rev
            OUTPUT_QUIET ERROR_QUIET)
        if(_rev EQUAL 0)
            continue()  # already applied
        endif()
        execute_process(
            COMMAND patch -p1 -F0 --dry-run -s
            INPUT_FILE "${_patch}"
            WORKING_DIRECTORY "${_ZEROMOE_APPLY_DIR}"
            RESULT_VARIABLE _fwd
            OUTPUT_QUIET ERROR_QUIET)
        if(_fwd EQUAL 0)
            execute_process(
                COMMAND patch -p1 -F0 --no-backup-if-mismatch -s
                INPUT_FILE "${_patch}"
                WORKING_DIRECTORY "${_ZEROMOE_APPLY_DIR}"
                RESULT_VARIABLE _rc
                ERROR_VARIABLE _err)
            if(NOT _rc EQUAL 0)
                message(FATAL_ERROR "ZeroMoE: failed to apply ggml patch ${dir}/${_name}:\n${_err}")
            endif()
            message(STATUS "ZeroMoE: applied ggml patch ${dir}/${_name}")
            continue()
        endif()
        if(required)
            message(WARNING
                "ZeroMoE: required ggml patch ${dir}/${_name} neither applies nor is "
                "already applied -- the ggml tree differs from the series base (see "
                "docs/build-ggml.md). Expect link errors for the symbols it provides.")
        else()
            message(STATUS "ZeroMoE: skipped optional ggml patch ${dir}/${_name} "
                           "(does not apply to this ggml tree)")
        endif()
    endforeach()
endfunction()

function(zeromoe_apply_ggml_patches ggml_root)
    set(_ZEROMOE_APPLY_DIR "${ggml_root}")

    # A sequential series cannot be re-checked patch by patch (a later patch changes the context of an
    # earlier one), so the applied state is recorded: a stamp holding the hash of every patch file that
    # was applied. A tree with the same stamp is left alone; a changed series on a patched tree is an error.
    set(_stamp_files "")
    foreach(_dir IN ITEMS ggml ggml-host-node ggml-host-node-beellama debug)
        file(GLOB _p "${CMAKE_CURRENT_SOURCE_DIR}/patches/${_dir}/*.patch")
        list(SORT _p)
        list(APPEND _stamp_files ${_p})
    endforeach()
    set(_stamp "")
    foreach(_f IN LISTS _stamp_files)
        file(SHA256 "${_f}" _h)
        string(APPEND _stamp "${_h}\n")
    endforeach()
    string(APPEND _stamp "host_node=${ZEROMOE_ENABLE_GGML_HOST_NODE_PATCHES} debug=${ZEROMOE_GGML_DEBUG_PATCHES}\n")
    set(_stamp_file "${ggml_root}/.zeromoe_patches_applied")
    if(EXISTS "${_stamp_file}")
        file(READ "${_stamp_file}" _have)
        if(_have STREQUAL _stamp)
            message(STATUS "ZeroMoE: ggml patch series already applied to ${ggml_root} (stamp matches)")
            return()
        endif()
        message(FATAL_ERROR "ZeroMoE: ${ggml_root} carries a different ggml patch series than patches/. "
                            "Use a fresh ggml tree (scripts/build_ggml.sh makes one).")
    endif()

    _zeromoe_ggml_apply_dir(ggml TRUE)
    if(ZEROMOE_ENABLE_GGML_HOST_NODE_PATCHES)
        _zeromoe_ggml_apply_dir(ggml-host-node TRUE)
        _zeromoe_ggml_apply_dir(ggml-host-node-beellama FALSE)
    endif()
    if(ZEROMOE_GGML_DEBUG_PATCHES)
        _zeromoe_ggml_apply_dir(debug FALSE)
    endif()
    file(WRITE "${_stamp_file}" "${_stamp}")
endfunction()

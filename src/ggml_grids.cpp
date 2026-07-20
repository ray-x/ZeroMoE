// src/ggml_grids.cpp — expose ggml's IQ lookup-grid DATA (host pointers) so the
// CUDA fused IQ2 kernels (cuda/iq2_matmul.cu) can copy them into __constant__
// memory. We pull the table data via GGML_COMMON_IMPL_C — the same define
// ggml-quants.c uses to instantiate the grids. No structs are referenced here,
// only the arrays. (The CUDA TU deliberately does NOT include ggml-common.h, to
// avoid its DECL/IMPL include-guard tangle and the block_q4_0 name clash.)
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

extern "C" {
const void* ms_grid_iq2xxs()       { return iq2xxs_grid;   } // uint64_t[256]
const void* ms_grid_iq2xs()        { return iq2xs_grid;    } // uint64_t[512]
const void* ms_grid_iq2s()         { return iq2s_grid;     } // uint64_t[1024]
const void* ms_grid_ksigns_iq2xs() { return ksigns_iq2xs;  } // uint8_t[128]
const void* ms_grid_kmask_iq2xs()  { return kmask_iq2xs;   } // uint8_t[8]
const void* ms_grid_iq3xxs()       { return iq3xxs_grid;   } // uint32_t[256]
const void* ms_grid_kvalues_iq4nl() { return kvalues_iq4nl; } // int8_t[16] -- fixed non-linear lookup table shared by IQ4_NL/IQ4_XS
}

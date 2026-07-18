#include "engine.hpp"
#include "logger.hpp"
#include "kquant_dequant.hpp"  // gpu_type_supported
#include <cuda_runtime.h>
#include <iostream>
#include <regex>
#include <algorithm>
#include <cstring>
#include <map>
#include <vector>
#include <tuple>

class GGUFGraphInitializer {
public:
    static bool build_model_graph(StreamingEngine& engine, int max_ctx);
};

// Several call sites (execution_layer.cpp's rmsnorm_row/router-bias reads,
// attention_layer.cpp's Q/K-norm reads, dense_dispatch.hpp's rmsnorm_proj CPU
// fallback) reinterpret a norm-weight or bias tensor's host_ptr as float*
// UNCONDITIONALLY, trusting that norm/bias tensors are always stored F32 in
// the GGUF (true for virtually every real quantization scheme, but never
// actually verified against THIS file). If any one of these is quantized or
// F16 instead, every element read through that reinterpret is garbage from
// byte zero -- and since one norm weight feeds every token/every downstream
// computation for its layer, this exactly matches a real symptom hit this
// session: a layer's router weights AND every expert's FFN going NaN
// simultaneously, from a single shared input (see moe_forward_gguf's
// scratch.normed). This is a load-time, whole-model version of the runtime
// check added at that one call site -- catches EVERY layer at once, at
// startup, instead of only the specific layer a given generation run
// happens to reach.
static void verify_float_reinterpret_tensors(const StreamingEngine& engine) {
    int checked = 0, bad = 0;
    for (const auto& kv : engine.resident_layers) {
        const std::string& name = kv.first;
        bool is_norm_or_bias = name.find("norm") != std::string::npos ||
                                (name.size() >= 5 && name.compare(name.size() - 5, 5, ".bias") == 0);
        if (!is_norm_or_bias) continue;
        ++checked;
        if (kv.second.meta.type != GGUFType::F32) {
            ++bad;
            LOG_ERROR("[LOAD-CHECK] %s type=%u (NOT F32) -- this tensor is read via an "
                      "unconditional reinterpret_cast<float*>(host_ptr) somewhere in the forward "
                      "path; every value read through it will be garbage, not just imprecise.\n",
                      name.c_str(), (unsigned)kv.second.meta.type);
        }
    }
    if (bad > 0) {
        LOG_STREAM_ERROR() << "[LOAD-CHECK] " << bad << "/" << checked
                  << " norm/bias tensors are NOT F32 -- see lines above. This alone can explain "
                     "whole-layer NaN corruption with no bug in any matmul/dequant kernel at all.\n";
    } else {
        LOG_STREAM_INFO() << "[LOAD-CHECK] All " << checked
                  << " norm/bias tensors are F32 as every reinterpret_cast<float*> site assumes.\n";
    }
}

// Comprehensive load-time audit: every attention/router/FFN weight tensor's
// dimensions, as ArchitectureConfig derives them from GGUF metadata KEYS,
// checked against that tensor's REAL shape as loaded from the file (its own
// ne[] dims -- ground truth, independent of any metadata key
// interpretation). This is the generalized form of the check that actually
// found this session's real corruption bug: qk_nope was silently computed
// as 256 (reading attention.key_length_mla directly) when attn_k_b's real
// ne[0] was 192 (key_length_mla is nope+rope COMBINED, not nope alone --
// confirmed against llama.cpp's deepseek2.cpp). Any OTHER metadata key with
// a similar "doesn't mean what the name suggests" issue would silently
// produce the exact same class of corruption -- a dense_proj/dequant call
// reading the wrong number of elements from that tensor's real bytes, at
// every layer, every token -- so every dimension the live MLA attention/
// router path actually uses gets checked here, once, at load time, rather
// than only being discoverable by chasing bad output through a full
// generation run each time.
static void verify_tensor_dimensions(StreamingEngine& engine) {
    auto& file = engine.model_file;
    auto& c = file.architecture_config;
    if (!c.is_mla) return;  // this tensor set is MLA-specific (GLM-5.2/DeepSeek-style)

    int mismatches = 0;
    auto check = [&](const std::string& name, uint64_t expected) {
        auto it = file.tensors.find(name);
        if (it == file.tensors.end()) return;  // absence is reported elsewhere (dense manifest coverage)
        uint64_t real = 1;
        for (auto d : it->second.shape) real *= d;
        if (real != expected) {
            ++mismatches;
            std::string shape_str;
            for (auto d : it->second.shape) shape_str += std::to_string(d) + ",";
            LOG_ERROR("[DIM-CHECK] %s: real numel=%llu (shape=[%s]) != expected=%llu "
                      "(derived from ArchitectureConfig) -- a dense_proj/dequant call using "
                      "the expected size will read the WRONG number of elements from this "
                      "tensor's real bytes.\n",
                      name.c_str(), (unsigned long long)real, shape_str.c_str(),
                      (unsigned long long)expected);
        }
    };

    int D = c.hidden;
    int H = c.n_heads;
    int qh = c.qk_nope + c.qk_rope;
    int kv_dim = c.kv_lora + c.qk_rope;
    int q_lora = file.get_config_int("attention.q_lora_rank", 2048);

    check("token_embd.weight", (uint64_t)c.vocab * D);
    check("output.weight", (uint64_t)c.vocab * D);
    check("output_norm.weight", (uint64_t)D);

    // Representative MLA layer (dims should be uniform across every MLA layer)
    int l = std::min(c.first_dense, std::max(0, c.n_layers - 1));
    std::string p = "blk." + std::to_string(l) + ".";
    check(p + "attn_norm.weight", (uint64_t)D);
    check(p + "attn_q_a.weight", (uint64_t)D * q_lora);
    check(p + "attn_q_a_norm.weight", (uint64_t)q_lora);
    check(p + "attn_q_b.weight", (uint64_t)q_lora * H * qh);
    check(p + "attn_kv_a_mqa.weight", (uint64_t)D * kv_dim);
    check(p + "attn_kv_a_norm.weight", (uint64_t)c.kv_lora);
    check(p + "attn_k_b.weight", (uint64_t)c.qk_nope * c.kv_lora * H);
    check(p + "attn_v_b.weight", (uint64_t)c.kv_lora * c.v_head * H);
    check(p + "attn_output.weight", (uint64_t)H * c.v_head * D);

    // ne[0] (fastest / contiguous dim) orientation check for the two MLA
    // up-projections whose absorption matmul in mla_attention_absorbed_gguf
    // assumes a specific layout:
    //   attn_k_b ne[0] == qk_nope   (query absorption: q_absorbed[kv] = sum_nope q[nope]*W[nope,kv])
    //   attn_v_b ne[0] == kv_lora   (value projection: v[vh] = sum_kv latent[kv]*W[kv,vh])
    // The numel check() above CANNOT catch a transpose -- [a,b,c] and [b,a,c]
    // have the same product -- so a swapped orientation would silently make the
    // absorption read the transposed matrix and corrupt every MLA layer's
    // attention (finite garbage -> "random text", identical failure mode to the
    // RoPE bug). Confirmed correct vs llama.cpp deepseek2 wk_b/wv_b shapes
    // ({qk_nope,kv_lora,n_head} and {kv_lora,v_head,n_head}); this just makes a
    // future drift loud at load time instead of invisible.
    auto check_orient = [&](const std::string& name, int expected_ne0) {
        auto it = file.tensors.find(name);
        if (it == file.tensors.end() || it->second.shape.empty()) return;
        int ne0 = (int)it->second.shape[0];
        if (ne0 != expected_ne0) {
            LOG_ERROR("[ORIENT-CHECK] %s ne[0]=%d != expected %d -- the MLA absorption "
                      "matmul reads this tensor assuming ne[0]=%d; a different contiguous "
                      "dim means the query/value projection is silently TRANSPOSED and every "
                      "MLA layer's attention is garbage.\n",
                      name.c_str(), ne0, expected_ne0, expected_ne0);
        }
    };
    check_orient(p + "attn_k_b.weight", c.qk_nope);
    check_orient(p + "attn_v_b.weight", c.kv_lora);

    // Representative MoE layer's router + shared expert (first MoE layer)
    if (c.n_experts > 0 && c.first_dense < c.n_layers) {
        std::string pm = "blk." + std::to_string(c.first_dense) + ".";
        check(pm + "ffn_norm.weight", (uint64_t)D);
        check(pm + "ffn_gate_inp.weight", (uint64_t)c.n_experts * D);
        if (c.n_expert_shared > 0) {
            uint64_t shared_inter = (uint64_t)c.moe_inter * c.n_expert_shared;
            check(pm + "ffn_gate_shexp.weight", (uint64_t)D * shared_inter);
            check(pm + "ffn_up_shexp.weight", (uint64_t)D * shared_inter);
            check(pm + "ffn_down_shexp.weight", shared_inter * (uint64_t)D);
        }
    }

    // Leading dense layer's plain FFN, if this model has any (layer 0)
    if (c.first_dense > 0) {
        check("blk.0.ffn_gate.weight", (uint64_t)D * c.dense_inter);
        check("blk.0.ffn_up.weight", (uint64_t)D * c.dense_inter);
        check("blk.0.ffn_down.weight", (uint64_t)c.dense_inter * D);
    }

    if (mismatches == 0) {
        LOG_STREAM_INFO() << "[DIM-CHECK] All checked tensor dimensions match ArchitectureConfig.\n";
    } else {
        LOG_STREAM_ERROR() << "[DIM-CHECK] " << mismatches
                            << " tensor(s) have a real size mismatch -- see lines above.\n";
    }
}

// Step 4 (report): surface every dense resident tensor's GGUFType and the
// routed-expert type so unsupported quants are VISIBLE. Today's GPU dense
// kernels cover F32/F16/BF16/Q8_0; anything else will hit the loud CPU-fallback
// warning in dense_proj() (attention_layer.cpp) until its kernel lands. This is
// the guard against the silent "reinterpret quant buffer as float" garbage class
// of bug — make it loud at load time, not at first wrong token.
static void report_dense_quant_coverage(const StreamingEngine& engine) {
    std::map<uint32_t, int> dense_counts;
    for (const auto& kv : engine.resident_layers)
        dense_counts[(uint32_t)kv.second.meta.type]++;

    // Kept as its own list rather than reusing gpu_type_supported() (kquant_dequant.hpp):
    // that one models gpu_matmul_dispatch (the EXPERT-FFN fused-kernel path, which
    // excludes F32/F16/BF16 since those route through the SEPARATE cuBLAS path in
    // dense_matmul_host below) — a genuinely different capability set from "can this
    // dense tensor be GPU-dispatched at all". Must be kept in sync with
    // dense_matmul_host's switch (cuda_manager.hpp) by hand.
    auto gpu_fused = [](uint32_t t) -> bool {
        switch ((GGUFType)t) {
            case GGUFType::F32: case GGUFType::F16: case GGUFType::BF16:
            case GGUFType::Q4_0: case GGUFType::Q8_0:
            case GGUFType::Q2_K: case GGUFType::Q3_K: case GGUFType::Q4_K:
            case GGUFType::Q5_K: case GGUFType::Q6_K:
            case GGUFType::IQ2_XXS: case GGUFType::IQ2_XS: case GGUFType::IQ2_S:
            case GGUFType::IQ3_XXS:
                return true;  // GPU dispatch: cuBLAS (F32/F16/BF16) or fused dequant kernel
            default:
                return false; // IQ1/IQ3_S/IQ4: CPU dequant via libggml; IQ2_M: no ggml export
        }
    };

    LOG_STREAM_INFO() << "[QUANT] Dense tensor coverage (CPU dequant via libggml; GPU fused for "
                 "F32/F16/BF16/Q4_0/Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/IQ2_XXS/XS/S/IQ3_XXS):\n";
    for (const auto& tc : dense_counts) {
        const char* status =
            gpu_fused(tc.first) ? "GPU dispatch OK" :
            ((GGUFType)tc.first == GGUFType::IQ2_M) ? "PENDING (IQ2_M: no ggml dequant export)" :
            "CPU (libggml) — GPU fused pending";
        LOG_STREAM_INFO() << "  GGUFType " << tc.first << " : " << tc.second
                  << " tensor(s) — " << status << "\n";
    }

    // Routed-expert GPU capability, scanned across EVERY MoE layer (not just
    // blk.0 — Unsloth-style Dynamic quantization can vary quant type by
    // layer, so a single sample can be misleading). This is the direct
    // answer to "do we need to load all this into GPU at all": expert_ffn
    // only dispatches to GPU when gate/up/down are ALL GPU-supported
    // (all-or-nothing per expert), and ram_manager.hpp / execution_layer.cpp
    // / async_prefetcher.hpp all skip VRAM promotion for a layer whose
    // packed tensors aren't GPU-capable — so a layer reported "CPU-bound"
    // here will never be promoted to VRAM at runtime regardless of how hot
    // its experts get.
    {
        auto& c = engine.model_file.architecture_config;
        int gpu_capable_layers = 0, cpu_bound_layers = 0;
        // Layer COUNT per distinct (gate,up,down) combo, not just which
        // combos exist — Unsloth-style Dynamic quantization can mix several
        // combos across one model's layers (observed: 3+ distinct combos in
        // a single GLM-5.2 UD-IQ2_M file), and knowing "73 layers use this
        // one, 2 layers use that one" directly informs which missing GPU
        // kernel is actually worth writing first.
        std::map<std::tuple<uint32_t, uint32_t, uint32_t>, int> combo_counts;
        for (int l = c.first_dense; l < c.n_layers; ++l) {
            std::string p = "blk." + std::to_string(l) + ".";
            auto gE = engine.model_file.tensors.find(p + "ffn_gate_exps.weight");
            auto uE = engine.model_file.tensors.find(p + "ffn_up_exps.weight");
            auto dE = engine.model_file.tensors.find(p + "ffn_down_exps.weight");
            if (gE == engine.model_file.tensors.end() || uE == engine.model_file.tensors.end() ||
                dE == engine.model_file.tensors.end()) {
                // Legacy per-expert format — check expert 0 as a representative sample.
                gE = engine.model_file.tensors.find(p + "ffn_gate.0.weight");
                uE = engine.model_file.tensors.find(p + "ffn_up.0.weight");
                dE = engine.model_file.tensors.find(p + "ffn_down.0.weight");
                if (gE == engine.model_file.tensors.end() || uE == engine.model_file.tensors.end() ||
                    dE == engine.model_file.tensors.end())
                    continue;
            }
            bool capable = gpu_type_supported(gE->second.type) &&
                           gpu_type_supported(uE->second.type) &&
                           gpu_type_supported(dE->second.type);
            if (capable) ++gpu_capable_layers; else ++cpu_bound_layers;
            ++combo_counts[{(uint32_t)gE->second.type, (uint32_t)uE->second.type, (uint32_t)dE->second.type}];
        }

        LOG_STREAM_INFO() << "[QUANT] Routed experts: " << gpu_capable_layers
                  << " layer(s) GPU-capable, " << cpu_bound_layers
                  << " layer(s) CPU-bound (no kernel for gate/up/down quant type — "
                  << "VRAM promotion skipped for these, always compute on CPU)\n";
        // Sorted by layer count descending — the combo affecting the most
        // layers (i.e. the kernel most worth writing) appears first.
        std::vector<std::pair<std::tuple<uint32_t, uint32_t, uint32_t>, int>> sorted_combos(
            combo_counts.begin(), combo_counts.end());
        std::sort(sorted_combos.begin(), sorted_combos.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        for (const auto& [combo, count] : sorted_combos) {
            auto [g, u, d] = combo;
            GGUFType gt = (GGUFType)g, ut = (GGUFType)u, dt = (GGUFType)d;
            bool capable = gpu_type_supported(gt) && gpu_type_supported(ut) && gpu_type_supported(dt);
            LOG_STREAM_INFO() << "[QUANT]   [" << count << " layer(s)] gate=" << gpu_type_name(gt) << " ("
                      << gpu_type_bits_per_weight(gt) << " bits/weight)  up=" << gpu_type_name(ut) << " ("
                      << gpu_type_bits_per_weight(ut) << " bits/weight)  down=" << gpu_type_name(dt) << " ("
                      << gpu_type_bits_per_weight(dt) << " bits/weight) — " << (capable ? "GPU-capable" : "CPU-bound") << "\n";
            // Explain WHY, per unsupported projection — answers "is this
            // about bit-packing efficiency" directly rather than leaving the
            // reader to guess from a bare type name.
            if (!gpu_type_supported(gt)) LOG_STREAM_INFO() << "[QUANT]     gate=" << gpu_type_name(gt) << ": " << gpu_unsupported_reason(gt) << "\n";
            if (!gpu_type_supported(ut)) LOG_STREAM_INFO() << "[QUANT]     up="   << gpu_type_name(ut) << ": " << gpu_unsupported_reason(ut) << "\n";
            if (!gpu_type_supported(dt)) LOG_STREAM_INFO() << "[QUANT]     down=" << gpu_type_name(dt) << ": " << gpu_unsupported_reason(dt) << "\n";
        }
        if (cpu_bound_layers > 0 && gpu_capable_layers == 0) {
            LOG_STREAM_WARN() << "[QUANT] WARNING: NO routed-expert layer can use GPU acceleration with this "
                      "quantization — the expert VRAM cache (moe_cache_gb) provides zero compute benefit "
                      "for this model/quant combination today; see the per-type reasons above for what's "
                      "missing, or lower moe_cache_gb since it's not currently buying any speedup.\n";
        }
    }
}

// ----------------------------------------------------
// Fixed Linkage: Function defined externally to export the symbol
// ----------------------------------------------------
bool GGUFGraphInitializer::build_model_graph(StreamingEngine& engine, int max_ctx) {
    auto& file = engine.model_file;
    auto& c = file.architecture_config; 

    LOG_STREAM_INFO() << "[GRAPH] Analyzing GGUF tensor topology (" << file.tensors.size() << " tensors)...\n";

    // Dump first few tensor names for debugging
    {
        int shown = 0;
        for (const auto& [name, tensor] : file.tensors) {
            if (shown < 20) {
                LogStream ls(LogLevel::DEBUG);
                ls << "  [tensor] " << name << "  shape=[";
                for (size_t d = 0; d < tensor.shape.size(); ++d) {
                    if (d) ls << ",";
                    ls << tensor.shape[d];
                }
                ls << "]  type=" << (int)tensor.type
                   << "  shard=" << tensor.shard_idx << "\n";
                ++shown;
            }
        }
        if (file.tensors.size() > 20)
            LOG_STREAM_DEBUG() << "  ... (" << file.tensors.size() - 20 << " more tensors)\n";
    }

    // Dump KV metadata for debugging
    LOG_STREAM_DEBUG() << "[GRAPH] KV metadata (" << file.kv_uint.size() + file.kv_int.size()
              + file.kv_float.size() + file.kv_string.size() << " scalar entries):\n";
    for (const auto& [k, v] : file.kv_uint)   LOG_STREAM_DEBUG() << "  " << k << " = " << v << " (uint)\n";
    for (const auto& [k, v] : file.kv_int)    LOG_STREAM_DEBUG() << "  " << k << " = " << v << " (int)\n";
    for (const auto& [k, v] : file.kv_float)  LOG_STREAM_DEBUG() << "  " << k << " = " << v << " (float)\n";
    for (const auto& [k, v] : file.kv_string) LOG_STREAM_DEBUG() << "  " << k << " = \"" << v.substr(0, 64) << "\" (string)\n";

    // ---------- Extract architecture from KV metadata ----------
    // Standard GGUF keys (llama.cpp convention): {arch}.{param}
    c.hidden   = file.get_config_int("embedding_length", 0);
    // ROOT CAUSE (found by comparing a full per-layer trace against colibri,
    // a separately-implemented GLM-5.2 engine -- colibri's own source calls
    // this out directly: "usato dai 78 principali e dal layer MTP", i.e.
    // used by the 78 main layers AND separately by the MTP layer). GGUF's
    // block_count is n_layer_all -- the TOTAL block count INCLUDING trailing
    // NextN/MTP (speculative-decoding draft) blocks -- confirmed against
    // llama.cpp: `ml.get_key(LLM_KV_BLOCK_COUNT, hparams.n_layer_all)` and
    // `llama_hparams::n_layer() { return n_layer_all - n_layer_nextn; }` --
    // the MAIN forward pass loop uses n_layer(), never n_layer_all directly.
    // Reading block_count straight into c.n_layers (no subtraction) made
    // layers_forward_gguf's main loop run ONE EXTRA "layer" over the NextN
    // block's tensors, which are named completely differently
    // (blk.N.nextn.eh_proj/enorm/hnorm/... vs the normal
    // attn_norm/attn_q_a/ffn_gate_inp/... every real layer has) -- every
    // lookup for that block silently fails and falls back to zeroing
    // (missing attn_k_b/attn_v_b -> zeroed attention; missing
    // ffn_gate_inp.weight -> zeroed FFN), so the model's real LAST layer's
    // output got passed through one more zero-contribution residual step
    // right before the LM head, on every single token.
    int n_layer_all = file.get_config_int("block_count", 0);
    int n_layer_nextn = file.get_config_int("nextn_predict_layers", 0);
    c.n_layers = n_layer_all - n_layer_nextn;
    if (n_layer_nextn > 0) {
        LOG_STREAM_INFO() << "[GRAPH] block_count=" << n_layer_all << " nextn_predict_layers="
                           << n_layer_nextn << " -- using n_layers=" << c.n_layers
                           << " for the main forward pass (excludes NextN/MTP draft block(s), "
                              "which this engine doesn't implement speculative decoding for).\n";
    }
    c.n_heads  = file.get_config_int("attention.head_count", 0);
    c.vocab    = file.get_config_int("vocab_size", 0);
    c.n_experts = file.get_config_int("expert_count", 0);
    c.topk     = file.get_config_int("expert_used_count", 8);
    c.kv_lora  = file.get_config_int("attention.kv_lora_rank", 512);
    c.qk_rope  = file.get_config_int("rope.dimension_count",
                 file.get_config_int("attention.key_length_rope", 64));
    // ROOT CAUSE (found via a real NaN-corruption investigation this
    // session): attention.key_length_mla is the COMBINED qk head dim
    // (nope+rope), NOT the nope-only dimension the previous comment here
    // claimed -- confirmed against llama.cpp's actual loader
    // (src/models/deepseek2.cpp): `n_embd_head_qk_nope = n_embd_head_k_mla -
    // n_embd_head_qk_rope`. Reading key_length_mla straight into qk_nope
    // overcounted nope by qk_rope elements (this model: 256 instead of the
    // real 192, a 25% overrun) -- every attn_k_b/attn_v_b dequant then read
    // that many elements from host_ptr, running past attn_k_b's real data
    // into whatever tensor follows it in memory. That's what made
    // hidden_states 100% NaN from layer 0 of every single generation run.
    // (attention.value_length_mla, by contrast, IS used directly with no
    // subtraction in the reference loader -- confirmed correct, left as-is.)
    int key_length_mla = file.get_config_int("attention.key_length_mla", 0);
    c.qk_nope = (key_length_mla > 0)
        ? (key_length_mla - c.qk_rope)
        : file.get_config_int("attention.key_length_nope", 128);
    c.v_head   = file.get_config_int("attention.value_length_mla",
                 file.get_config_int("attention.value_length", 128));
    c.eps       = file.get_config_float("attention.layer_norm_rms_epsilon", 1e-5f);
    // GLM-5.2 uses "leading_dense_block_count" (not "first_k_dense_replace")
    int first_dense = file.get_config_int("leading_dense_block_count",
                      file.get_config_int("first_k_dense_replace", 0));
    c.moe_inter = file.get_config_int("expert_feed_forward_length", 0);
    c.dense_inter = file.get_config_int("feed_forward_length", 0);
    c.first_dense = first_dense;

    // Dense Q/K/V attention params (Hunyuan/Llama); harmless defaults for MLA.
    c.n_head_kv = file.get_config_int("attention.head_count_kv", c.n_heads);
    c.head_dim  = file.get_config_int("attention.key_length",
                                      c.hidden / std::max(1, c.n_heads));
    std::string arch0 = file.kv_string.count("general.architecture")
                        ? file.kv_string["general.architecture"] : "";
    c.is_mla = (arch0 == "glm-dsa" || arch0 == "deepseek2");

    // ---------- MoE router config: gating function, selection bias/scale/norm ----------
    // See llama.cpp's build_moe_ffn (llama-graph.cpp) for the reference algorithm.
    c.n_expert_shared = file.get_config_int("expert_shared_count", 0);
    c.expert_weights_scale = file.get_config_float("expert_weights_scale", 0.0f);
    c.expert_weights_norm = file.get_config_int("expert_weights_norm", 1) != 0;
    {
        // GGUF encoding (llama.cpp llama_expert_gating_func_type):
        // 0 = none/unset, 1 = softmax, 2 = sigmoid.
        int gating = file.get_config_int("expert_gating_func", 0);
        if (gating == 2) c.use_sigmoid_gate = true;
        else if (gating == 1) c.use_sigmoid_gate = false;
        else c.use_sigmoid_gate = (arch0 == "glm-dsa" || arch0 == "hy_v3");
    }
    // Original HunyuanMoE ropes Q/K before QK-norm; Hy-V3/GLM-5.2 norm first.
    c.rope_before_norm = (arch0 == "hunyuan-moe");

    // ---------- YaRN long-context RoPE scaling ----------
    // See ggml's rope_yarn/ggml_rope_yarn_corr_dims for the reference formulas.
    // ext_factor == 0 (the default, when rope.scaling.type != "yarn") reduces
    // every formula below to plain RoPE — no behavior change for models that
    // don't set YaRN metadata.
    {
        std::string scaling_type = file.get_config_string("rope.scaling.type", "none");
        c.rope_scaling_yarn = (scaling_type == "yarn");
        c.rope_ext_factor = c.rope_scaling_yarn ? 1.0f : 0.0f;
        float scale_factor = file.get_config_float("rope.scaling.factor", 1.0f);
        c.rope_freq_scale = scale_factor > 0.0f ? 1.0f / scale_factor : 1.0f;
        c.rope_attn_factor = file.get_config_float("rope.scaling.attn_factor", 1.0f);
        // GLM-DSA's own hparams loader never sets yarn_log_multiplier (only
        // DeepSeek2/DeepSeek32/Mistral3 do upstream) so this is 0.0 for
        // GLM-5.2 in practice; read generically for forward-compat.
        c.rope_yarn_log_mul = file.get_config_float("rope.scaling.yarn_log_multiplier", 0.0f);
        c.rope_n_ctx_orig = file.get_config_int("rope.scaling.original_context_length",
                             file.get_config_int("context_length", max_ctx));
        // colibri (a working GLM-5.2 reference) uses a PLAIN attn_scale =
        // 1/sqrt(qk_head) with no YaRN adjustment at all -- only correct to
        // compare against if rope_ext_factor genuinely comes out 0 here too.
        // Logged directly (not inferred) so this is checkable, not assumed.
        LOG_STREAM_INFO() << "[ROPE] scaling.type=\"" << scaling_type << "\" ext_factor=" << c.rope_ext_factor
                           << " freq_scale=" << c.rope_freq_scale << " attn_factor=" << c.rope_attn_factor
                           << " yarn_log_mul=" << c.rope_yarn_log_mul << " n_ctx_orig=" << c.rope_n_ctx_orig
                           << " -- ext_factor=0 means plain RoPE, attn_scale reduces to 1/sqrt(qk_head) "
                              "(matches colibri exactly); nonzero means the YaRN branch is live.\n";
    }

    // Fallback: detect from tensor shapes if KV didn't provide values
    if (c.hidden == 0 || c.n_layers == 0) {
        uint64_t max_layer_found = 0;
        std::regex layer_regex(R"(blk\.([0-9]+)\.)");
        for (const auto& [name, tensor] : file.tensors) {
            std::smatch match;
            if (std::regex_search(name, match, layer_regex)) {
                uint64_t layer_idx = std::stoull(match[1].str());
                if (layer_idx > max_layer_found) max_layer_found = layer_idx;
            }
            if (c.hidden == 0 && name == "token_embd.weight") {
                c.vocab = tensor.shape[0];
                c.hidden = tensor.shape[1];
            }
            // Detect hidden from any attn weight shape
            if (c.hidden == 0 && name.find("attn_output.weight") != std::string::npos) {
                c.hidden = tensor.shape[0]; // [H*vh, hidden] → shape[0] might be H*vh
            }
        }
        if (c.n_layers == 0) c.n_layers = max_layer_found + 1;
    }

    // Detect experts from packed expert tensors
    if (c.n_experts == 0) {
        for (const auto& [name, tensor] : file.tensors) {
            if (name.find("ffn_gate_exps.weight") != std::string::npos && tensor.shape.size() >= 3) {
                c.n_experts = tensor.shape[2]; // shape=[hidden, inter, n_experts]
                if (c.moe_inter == 0) c.moe_inter = tensor.shape[1];
                break;
            }
        }
    }

    if (c.n_heads == 0 && c.hidden > 0 && c.v_head > 0) {
        // For MLA: attn_output is [n_heads*v_head, hidden], so n_heads = shape[0]/v_head
        for (const auto& [name, tensor] : file.tensors) {
            if (name.find("blk.0.attn_output.weight") != std::string::npos && tensor.shape.size() >= 2) {
                c.n_heads = tensor.shape[0] / c.v_head;
                break;
            }
        }
    }

    LOG_STREAM_INFO() << "[GRAPH] Architecture discovered:\n";
    LOG_STREAM_INFO() << "  -> Hidden: " << c.hidden << " | Layers: " << c.n_layers
              << " | Heads: " << c.n_heads << " | Vocab: " << c.vocab << "\n";
    LOG_STREAM_INFO() << "  -> Experts: " << c.n_experts << " (top-" << c.topk
              << ", inter=" << c.moe_inter << ")\n";
    LOG_STREAM_INFO() << "  -> MLA: kv_lora=" << c.kv_lora << " qk_nope=" << c.qk_nope
              << " qk_rope=" << c.qk_rope << " v_head=" << c.v_head << "\n";
    LOG_STREAM_INFO() << "  -> First dense layers: " << first_dense << "\n";

    // ---------- Allocate KV cache (host + attempt GPU) ----------
    engine.max_t = max_ctx;
    size_t total_kv_bytes = 0;
    if (c.is_mla) {
        engine.Lc.resize(c.n_layers + 1, nullptr);
        engine.Rc.resize(c.n_layers + 1, nullptr);
        engine.d_Lc.resize(c.n_layers + 1, nullptr);
        engine.d_Rc.resize(c.n_layers + 1, nullptr);
        for (int i = 0; i <= c.n_layers; ++i) {
            size_t lc_bytes = (size_t)max_ctx * c.kv_lora * sizeof(float);
            size_t rc_bytes = (size_t)max_ctx * c.qk_rope * sizeof(float);
            posix_memalign((void**)&engine.Lc[i], 4096, lc_bytes);
            posix_memalign((void**)&engine.Rc[i], 4096, rc_bytes);
            std::memset(engine.Lc[i], 0, lc_bytes);
            std::memset(engine.Rc[i], 0, rc_bytes);
            total_kv_bytes += lc_bytes + rc_bytes;
        }
    } else {
        // Dense K/V cache (Hunyuan/Llama): [max_t, n_head_kv*head_dim] per layer.
        int kvd = std::max(1, c.n_head_kv) * std::max(1, c.head_dim);
        engine.Kc.resize(c.n_layers, nullptr);
        engine.Vc.resize(c.n_layers, nullptr);
        size_t kv_bytes = (size_t)max_ctx * kvd * sizeof(float);
        for (int i = 0; i < c.n_layers; ++i) {
            posix_memalign((void**)&engine.Kc[i], 4096, kv_bytes);
            posix_memalign((void**)&engine.Vc[i], 4096, kv_bytes);
            std::memset(engine.Kc[i], 0, kv_bytes);
            std::memset(engine.Vc[i], 0, kv_bytes);
        }
        total_kv_bytes = 2 * kv_bytes * c.n_layers;
        LOG_STREAM_INFO() << "[GRAPH] Dense KV cache: " << total_kv_bytes/1e9 << " GB ("
                  << c.n_head_kv << " kv heads x " << c.head_dim << ")\n";
    }

    // NOTE: this used to also mirror the latent/rope KV cache into VRAM here
    // (d_Lc/d_Rc/d_Lc_q4/d_Rc_q4, "Q4 then FP32" attempt). Removed: the live
    // MLA attention path (mla_attention_absorbed_gguf, src/attention_layer.cpp)
    // reads and writes ONLY engine.Lc[layer_idx]/engine.Rc[layer_idx] — plain
    // host pointers, via AVX-512 SIMD — and never touches d_Lc/d_Rc/d_Lc_q4/
    // d_Rc_q4 at all. There IS a real GPU kernel for this (cuda/mla_attention.cu's
    // launch_mla_attention, with mla_score_kernel/mla_softmax_kernel/
    // mla_value_aggregate_kernel behind it) but it is never called from
    // anywhere in the codebase, and the kv_on_device/kv_quantized flags that
    // would gate using it are set here but never read anywhere either. So
    // this was allocating and zeroing up to ~0.85 GB (Q4) or ~6 GB (FP32
    // fallback) of VRAM on the core GPU for a buffer that had zero effect on
    // compute — the exact same class of bug as the d_hidden/d_scratch dead
    // allocation removed earlier from gpu_topology.hpp. If launch_mla_attention
    // is ever wired up, this allocation (and the fields, still present on
    // StreamingEngine) can come back.
    if (c.is_mla) {
        LOG_STREAM_INFO() << "[GRAPH] KV cache: host-only (" << total_kv_bytes/1e9
                  << " GB) — GPU KV mirror unused by the live attention path, not allocated\n";
    }

    // ---------- Build dense preload manifest with correct GGUF tensor names ----------
    std::vector<std::string> dense_preload_manifest;
    engine.layers_experts.resize(c.n_layers);

    // MLA (absorbed) vs dense Q/K/V attention, by architecture string.
    std::string arch = file.kv_string.count("general.architecture")
                       ? file.kv_string["general.architecture"] : "";
    const bool is_mla = c.is_mla;
    LOG_STREAM_INFO() << "[GRAPH] Attention type: " << (is_mla ? "MLA (absorbed)" : "dense Q/K/V")
              << "  (arch=\"" << arch << "\")\n";

    for (int l = 0; l < c.n_layers; ++l) {
        std::string p = "blk." + std::to_string(l) + ".";

        // Norms (always present)
        dense_preload_manifest.push_back(p + "attn_norm.weight");
        dense_preload_manifest.push_back(p + "ffn_norm.weight");

        if (is_mla) {
            // MLA absorbed attention (GLM-5.2 / DeepSeek-V3)
            dense_preload_manifest.push_back(p + "attn_q_a.weight");
            dense_preload_manifest.push_back(p + "attn_q_a_norm.weight");
            dense_preload_manifest.push_back(p + "attn_q_b.weight");
            dense_preload_manifest.push_back(p + "attn_kv_a_mqa.weight");
            dense_preload_manifest.push_back(p + "attn_kv_a_norm.weight");
            dense_preload_manifest.push_back(p + "attn_k_b.weight");
            dense_preload_manifest.push_back(p + "attn_v_b.weight");
        } else {
            // Dense Q/K/V attention (Hunyuan / Llama / GLM-4)
            dense_preload_manifest.push_back(p + "attn_q.weight");
            dense_preload_manifest.push_back(p + "attn_k.weight");
            dense_preload_manifest.push_back(p + "attn_v.weight");
            dense_preload_manifest.push_back(p + "attn_q_norm.weight");  // HY-V3 QK-norm
            dense_preload_manifest.push_back(p + "attn_k_norm.weight");
        }
        dense_preload_manifest.push_back(p + "attn_output.weight");

        bool is_moe = (c.n_experts > 0 && l >= first_dense);
        if (is_moe) {
            // Router
            dense_preload_manifest.push_back(p + "ffn_gate_inp.weight");
            // Selection-bias tensor (DeepSeek-V3/GLM-5.2 style). GGUF name is
            // "blk.N.exp_probs_b.bias" — llama.cpp's tn(LLM_TENSOR_FFN_EXP_PROBS_B,
            // "bias", N) resolves LLM_TENSOR_FFN_EXP_PROBS_B's pattern
            // "blk.%d.exp_probs_b" then appends ".bias" (confirmed against
            // llama-arch.cpp:407 + LLM_TN_IMPL::str()). This used to look for
            // "ffn_gate_inp.e_score_correction_bias" — a name that doesn't
            // exist in any real GGUF, so this tensor was NEVER actually
            // loaded (explains the "1106/1182 tensors found" gap — 76
            // missing = exactly the MoE layer count). The router code
            // already handles the bias tensor being absent gracefully (falls
            // back to unbiased selection), so this wasn't a crash, just
            // silently wrong/incomplete routing for models that do ship it.
            dense_preload_manifest.push_back(p + "exp_probs_b.bias");
            // Shared expert
            dense_preload_manifest.push_back(p + "ffn_gate_shexp.weight");
            dense_preload_manifest.push_back(p + "ffn_up_shexp.weight");
            dense_preload_manifest.push_back(p + "ffn_down_shexp.weight");
            // Packed routed experts (streamed from disk, NOT preloaded here)
            // ffn_gate_exps, ffn_up_exps, ffn_down_exps are loaded on-demand

            engine.layers_experts[l].resize(std::max(1, c.n_experts));
            for (int e = 0; e < c.n_experts; ++e) {
                engine.layers_experts[l][e].expert_id = -1;
                engine.layers_experts[l][e].slab_buffer = nullptr;
                engine.layers_experts[l][e].slab_capacity = 0;
            }
        } else {
            // Dense FFN
            dense_preload_manifest.push_back(p + "ffn_gate.weight");
            dense_preload_manifest.push_back(p + "ffn_up.weight");
            dense_preload_manifest.push_back(p + "ffn_down.weight");
        }
    }

    // Global tensors
    dense_preload_manifest.push_back("token_embd.weight");
    dense_preload_manifest.push_back("output_norm.weight");
    dense_preload_manifest.push_back("output.weight");

    // Report and load
    int found = 0;
    for (const auto& name : dense_preload_manifest) {
        if (file.tensors.find(name) != file.tensors.end()) ++found;
    }
    LOG_STREAM_INFO() << "[GRAPH] Dense manifest: " << found << " / " << dense_preload_manifest.size()
              << " tensors found in GGUF\n";
    if (found == 0) {
        LOG_STREAM_WARN() << "[GRAPH] WARNING: No manifest tensors matched. First 5 expected:\n";
        for (size_t i = 0; i < std::min((size_t)5, dense_preload_manifest.size()); ++i)
            LOG_STREAM_WARN() << "    " << dense_preload_manifest[i] << "\n";
    }

    engine.initialize_resident_memory(dense_preload_manifest);

    verify_float_reinterpret_tensors(engine);
    verify_tensor_dimensions(engine);
    report_dense_quant_coverage(engine);
    return true;
}
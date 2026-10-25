// Native test harness: tests_expert_cache (split from tests/native/main_test.cpp).
#include "native_tests.hpp"

// ------------------------------------------------------------------
// 19. sanity_check_unpacked_format()'s RELAXATION itself, exercised
//     through the real StreamingEngine methods (not a reimplementation of
//     their logic) -- reuses the same hand-built mixed-type file Test 18
//     already wrote, so this test depends on Test 18 having run first in
//     the same process (main() below preserves that order).
//
// Three things checked, all against the real engine.hpp code:
//   (a) sanity_check_unpacked_format() no longer FATAL-refuses a layer
//       whose two experts have different byte sizes (the old behavior,
//       now removed) -- it must return true.
//   (b) it correctly flags that layer in layer_has_mixed_expert_sizes.
//   (c) expert_layer_geometry() -- the "one number represents the whole
//       layer" shortcut every RAM/VRAM-sizing caller still uses, and which
//       has NOT been made per-expert-aware yet -- correctly REFUSES
//       (returns false) for this flagged layer, rather than silently
//       handing back expert 0's size as if it applied to expert 1 too.
//       This is the safety net: Phase 2/3 callers that haven't been
//       updated yet get a clean "not supported for this layer" signal
//       instead of a silent wrong-size bug.
// unpacked_expert_geometry() itself (the real per-expert lookup, which
// Phase 2 work will call directly) is deliberately NOT re-tested here --
// Test 18 already exercises it for both experts via the exact same code
// path, reading back genuinely different, correct byte counts for each.
// ------------------------------------------------------------------
bool test_unpacked_format_relaxed_check() {
    std::cout << "[Test 19] sanity_check_unpacked_format() relaxation (mixed-size layer accepted, "
                 "geometry shortcut refuses)...\n";

    const std::string mixed_path = "/tmp/zeromoe_test_mixed_layer0.gguf";
    StreamingEngine engine;
    if (!engine.model_file.load_header(mixed_path)) {
        std::cout << "  -> SKIPPED (Test 18's mixed-layer file not found -- did it run first?)\n";
        return true;
    }
    // Test 18's file carries 0 KV pairs by design (see its own comment) --
    // architecture_config is normally populated by graph_initializer.cpp's
    // discovery pass, which this test deliberately does not invoke (out of
    // scope: this test drives sanity_check_unpacked_format()/expert_layer_
    // geometry() directly, not the full model-load pipeline). Set exactly
    // what those two functions read.
    engine.model_file.architecture_config.n_layers = 1;
    engine.model_file.architecture_config.n_experts = 2;
    engine.unpacked_expert_format = true;

    bool all_ok = true;

    bool sanity_ok = engine.sanity_check_unpacked_format();
    std::cout << "  sanity_check_unpacked_format(): " << (sanity_ok ? "true (accepted)" : "false (REFUSED)") << "\n";
    if (!sanity_ok) {
        std::cout << "  -> FAILED (a genuinely mixed-size layer should be ACCEPTED now, not refused -- "
                     "the old hard-fail should be gone)\n";
        all_ok = false;
    }

    bool flagged_mixed = !engine.layer_has_mixed_expert_sizes.empty() && engine.layer_has_mixed_expert_sizes[0];
    std::cout << "  layer_has_mixed_expert_sizes[0]: " << (flagged_mixed ? "true" : "false") << "\n";
    if (!flagged_mixed) {
        std::cout << "  -> FAILED (layer 0 genuinely has two different-sized experts and must be flagged)\n";
        all_ok = false;
    }

    size_t gb, ub, db;
    const GGUFTensor *gM, *uM, *dM;
    bool geometry_refused = !engine.expert_layer_geometry(0, gb, ub, db, gM, uM, dM);
    std::cout << "  expert_layer_geometry(0, ...): " << (geometry_refused ? "false (REFUSED, correct)" : "true (returned a size)") << "\n";
    if (!geometry_refused) {
        std::cout << "  -> FAILED (this shortcut must refuse for a mixed layer -- returning expert 0's "
                     "size here would silently mis-size every RAM/VRAM caller that still assumes one "
                     "size per layer)\n";
        all_ok = false;
    }

    std::cout << "  -> " << (all_ok ? "PASSED" : "FAILED")
              << " (relaxed check accepts real mixed sizes; the layer-level shortcut correctly refuses "
                 "instead of silently mis-sizing)\n";
    return all_ok;
}

// ------------------------------------------------------------------
// 20. Phase 2: compute_expert_slab_bytes() must return the LAYER'S MAX
//     per-expert size for a mixed layer, not just expert 0's.
//
// Deliberately builds a SEPARATE file from Test 18/19's, with the two
// experts' output indices SWAPPED (small IQ1_S expert at out_expert=0,
// larger baseline expert at out_expert=1) -- Test 18/19's own file
// happens to have the larger expert already at index 0, so checking
// compute_expert_slab_bytes against it wouldn't actually distinguish the
// old "expert 0 only" behavior from the new "true max" behavior; both
// would coincidentally return the same number. This ordering is chosen
// specifically so the two behaviors disagree: the old code would return
// expert 0's (SMALLER) IQ1_S total; only the fix returns the true
// (LARGER) max.
// ------------------------------------------------------------------
bool test_compute_expert_slab_bytes_max() {
    std::cout << "[Test 20] compute_expert_slab_bytes() returns the layer's TRUE MAX for a mixed layer...\n";

    const std::string prod_path = "/home/ray/vllm-workspace/GLM-5.3-GGUF-align";
    const std::string iq1s_path = "/home/ray/vllm-workspace/GLM-5.3-GGUF-align/"
        "repacked-GLM-5.3-UD-IQ1_S-00005-of-00006-part00001-of-00001.gguf";

    struct ExpertSpec { std::string src_path; int src_layer; int src_expert; int out_expert; };
    // SWAPPED vs Test 18: small (IQ1_S) expert first, large (baseline) expert second.
    std::vector<ExpertSpec> specs = {
        {iq1s_path, 60, 7, 0},
        {prod_path, 20, 5, 1},
    };

    struct OutTensor { std::string name; GGUFType type; std::vector<uint64_t> shape; std::vector<uint8_t> data; };
    std::vector<OutTensor> out_tensors;
    size_t expert_total[2] = {0, 0};

    for (auto& spec : specs) {
        GGUFFile f;
        if (!f.load_header(spec.src_path)) {
            std::cout << "  -> SKIPPED (real source not available: " << spec.src_path << ")\n";
            return true;
        }
        for (const char* kind : {"gate", "up", "down"}) {
            std::string tn = "blk." + std::to_string(spec.src_layer) + ".ffn_" + kind + "_exps.weight";
            auto it = f.tensors.find(tn);
            if (it == f.tensors.end()) { std::cout << "  -> FAILED (tensor not found: " << tn << ")\n"; return false; }
            const GGUFTensor& t = it->second;
            uint64_t n_experts = t.shape.back();
            uint64_t bpe = t.size_bytes / n_experts;
            uint64_t eoff = t.offset + (uint64_t)spec.src_expert * bpe;
            std::vector<uint8_t> raw(bpe);
            if (pread(f.shard_fds[t.shard_idx], raw.data(), bpe, eoff) != (ssize_t)bpe) {
                std::cout << "  -> FAILED (short read)\n"; return false;
            }
            OutTensor ot;
            ot.name = "blk.0.ffn_" + std::string(kind) + "." + std::to_string(spec.out_expert) + ".weight";
            ot.type = t.type;
            ot.shape.assign(t.shape.begin(), t.shape.end() - 1);
            expert_total[spec.out_expert] += bpe;
            ot.data = std::move(raw);
            out_tensors.push_back(std::move(ot));
        }
    }

    std::string out_path = "/tmp/zeromoe_test_mixed_layer0_swapped.gguf";
    {
        FILE* out = fopen(out_path.c_str(), "wb");
        if (!out) { std::cout << "  -> FAILED (cannot write " << out_path << ")\n"; return false; }
        auto w_u32 = [&](uint32_t v) { fwrite(&v, 4, 1, out); };
        auto w_u64 = [&](uint64_t v) { fwrite(&v, 8, 1, out); };
        auto w_str = [&](const std::string& s) { uint64_t n = s.size(); fwrite(&n, 8, 1, out); fwrite(s.data(), 1, n, out); };
        fwrite("GGUF", 1, 4, out);
        w_u32(3);
        w_u64(out_tensors.size());
        w_u64(0);
        const uint64_t ALIGN = 32;
        uint64_t running = 0;
        std::vector<uint64_t> offs;
        for (auto& ot : out_tensors) {
            w_str(ot.name);
            w_u32((uint32_t)ot.shape.size());
            for (auto d : ot.shape) w_u64(d);
            w_u32((uint32_t)ot.type);
            offs.push_back(running);
            w_u64(running);
            running += ot.data.size();
            running = (running + ALIGN - 1) & ~(ALIGN - 1);
        }
        long pos = ftell(out);
        long data_start = (pos + (long)ALIGN - 1) & ~((long)ALIGN - 1);
        while (ftell(out) < data_start) fputc(0, out);
        for (size_t i = 0; i < out_tensors.size(); ++i) {
            fseek(out, data_start + (long)offs[i], SEEK_SET);
            fwrite(out_tensors[i].data.data(), 1, out_tensors[i].data.size(), out);
        }
        fclose(out);
    }

    StreamingEngine engine;
    if (!engine.model_file.load_header(out_path)) { std::cout << "  -> FAILED (reload)\n"; return false; }
    engine.model_file.architecture_config.n_layers = 1;
    engine.model_file.architecture_config.n_experts = 2;
    engine.unpacked_expert_format = true;

    size_t expected_max = std::max(expert_total[0], expert_total[1]);
    size_t got = engine.compute_expert_slab_bytes(0);

    std::cout << "  expert 0 (small, IQ1_S) total=" << expert_total[0]
              << "  expert 1 (large, baseline) total=" << expert_total[1]
              << "  expected_max=" << expected_max << "  got=" << got << "\n";

    bool pass = (got == expected_max) && (got != expert_total[0] || expert_total[0] == expert_total[1]);
    if (got == expert_total[0] && expert_total[0] != expected_max) {
        std::cout << "  -> FAILED (returned expert 0's SMALLER size -- the old 'expert 0 only' bug is still present)\n";
        pass = false;
    } else if (got != expected_max) {
        std::cout << "  -> FAILED (returned " << got << ", neither expert's real total matches this)\n";
        pass = false;
    }
    std::cout << "  -> " << (pass ? "PASSED" : "FAILED") << "\n";
    return pass;
}

// ------------------------------------------------------------------
// 20b. Compact VRAM slot planner: variable-sized layer regions must fit the
//     requested byte budget, never exceed the model expert count, and keep
//     every layer with a usable slab represented when the budget permits.
//
// This is deliberately a pure topology test. It does not allocate CUDA
// memory, touch a model file, or start an engine, so it is safe to run in the
// component test target and gives the compact layout a deterministic guard.
// ------------------------------------------------------------------
bool test_compact_vram_slot_planner() {
    std::cout << "[Test 20b] compact VRAM slot planner preserves layer regions within budget...\n";

    const std::vector<size_t> layer_bytes = {10, 20, 10};
    const std::vector<int> counts =
        GpuTopology::compact_layer_slot_counts(100, 4, layer_bytes);
    size_t used = 0;
    bool ok = counts.size() == layer_bytes.size();
    for (size_t l = 0; l < counts.size(); ++l) {
        ok = ok && counts[l] >= 1 && counts[l] <= 4;
        used += (size_t)counts[l] * layer_bytes[l];
    }
    ok = ok && used <= 100 && counts == std::vector<int>({3, 2, 3});

    const std::vector<int> sparse =
        GpuTopology::compact_layer_slot_counts(50, 2, {0, 20, 30});
    ok = ok && sparse == std::vector<int>({0, 1, 1});

    const std::vector<int> full_prefix =
        GpuTopology::compact_layer_slot_counts(100, 4, layer_bytes, 1);
    ok = ok && full_prefix == std::vector<int>({4, 2, 2});

    // Explicit CPU-affinity layers must not strand their physical cache
    // regions. The same 100-byte pool should be redistributed across the
    // two GPU-executed layers rather than leaving layer 1's share unusable.
    const std::vector<int> cpu_affinity_excluded =
        GpuTopology::compact_layer_slot_counts(
            100, 4, layer_bytes, 0, std::vector<int>{1});
    ok = ok && cpu_affinity_excluded == std::vector<int>({4, 0, 4});

    std::cout << "  counts=";
    for (int count : counts) std::cout << count << " ";
    std::cout << "used=" << used << " bytes, full-prefix=";
    for (int count : full_prefix) std::cout << count << " ";
    std::cout << "-> " << (ok ? "PASSED" : "FAILED") << "\n";
    return ok;
}

// ------------------------------------------------------------------
// 20c. RAM source lifetime: an asynchronous H2D caller must receive an
// atomic hold together with its ready-slot lookup. A separate lookup followed
// by ram_pin() leaves a real eviction/regrow window for prefetch workers.
// This stays CPU-only: it exercises the RAM state machine and does not start
// CUDA, allocate a VRAM pool, or read a model file.
// ------------------------------------------------------------------
bool test_ram_sync_promotion_capacity() {
    std::cout << "[Test] RAM sync promotion preserves reusable slab capacity...\n";
    StreamingEngine engine;
    engine.layers_experts.resize(1);
    engine.layers_experts[0].resize(1);
    RAMSafetyManager ram;
    ram.slots_per_layer = 1;
    auto &slot = engine.layers_experts[0][0];
    constexpr size_t bytes = 4096;
    slot.slab_buffer = static_cast<uint8_t *>(calloc(1, bytes));
    if (!slot.slab_buffer) return false;
    slot.slab_capacity = bytes;
    slot.expert_id = 7;
    std::memset(slot.slab_buffer, 0x5a, bytes);
    uint8_t *const original = slot.slab_buffer;
    ram.preload_ram_slot(engine, 0, 7, 0);
    ram.move_on_promotion = true;
    ram.finish_sync_promotion(slot, 0, 7, &engine.layers_experts[0]);
    bool ok = slot.expert_id == -1 && slot.slab_buffer == original &&
              slot.slab_capacity == bytes;
    engine.ensure_slab_capacity(slot, bytes);
    ok = ok && slot.slab_buffer == original && slot.slab_capacity == bytes;
    for (size_t i = 0; i < bytes; ++i) ok = ok && slot.slab_buffer[i] == 0x5a;
    slot.expert_id = 8;
    ram.preload_ram_slot(engine, 0, 8, 0);
    ram.move_on_promotion = false;
    ram.finish_sync_promotion(slot, 0, 8, &engine.layers_experts[0]);
    ok = ok && slot.expert_id == 8 && slot.also_in_vram &&
         slot.slab_buffer == original && slot.slab_capacity == bytes;
    ram.move_on_promotion = true;
    ram.ram_pin(slot);
    ram.finish_sync_promotion(slot, 0, 8, &engine.layers_experts[0]);
    ok = ok && slot.expert_id == 8 && slot.slab_capacity == bytes;
    ram.ram_unpin(slot);
    std::cout << "  -> " << (ok ? "PASSED" : "FAILED") << "\n";
    return ok;
}

bool test_ram_async_source_hold() {
    std::cout << "[Test 20c] RAM ready lookup holds an async source slab...\n";

    StreamingEngine engine;
    engine.layers_experts.resize(1);
    engine.layers_experts[0].resize(1);

    RAMSafetyManager ram;
    ram.slots_per_layer = 1;
    // Heap-allocated (not a stack array): StreamingEngine's destructor calls
    // plain free() on any slot.slab_buffer that isn't slab_from_shared_arena
    // (engine.hpp, ~StreamingEngine) -- matching real production ownership,
    // where every slab_buffer is always a malloc/posix_memalign allocation.
    // A stack array here previously made `engine`'s destructor call free()
    // on stack memory at end of scope, corrupting the heap ("double free or
    // corruption (out)") the moment this test function returned -- not a
    // production bug, just this test not giving the engine a real buffer to
    // own. No manual free() needed below; the destructor now does it correctly.
    constexpr size_t kSourceBytes = 16;
    uint8_t *source_bytes = static_cast<uint8_t *>(calloc(1, kSourceBytes));
    MoEExpertBlock &slot = engine.layers_experts[0][0];
    slot.expert_id = 7;
    slot.slab_buffer = source_bytes;
    slot.slab_capacity = kSourceBytes;
    ram.preload_ram_slot(engine, 0, 7, 0);

    const uint8_t *captured = nullptr;
    MoEExpertBlock *held = ram.find_ram_slot_if_ready_held(
        engine.layers_experts[0], 1, 0, 7, &captured, nullptr);
    bool ok = held == &slot && captured == source_bytes &&
              slot.ram_temp_pin_count == 1;
    if (held)
        ram.ram_unpin(*held);

    // Also cover the wait path used when another thread published the slot:
    // pin_on_ready must be taken under the same lock that observed Ready.
    MoEExpertBlock *waited =
        ram.wait_for_ram_slot(engine, 0, 7, /*pin_on_ready=*/true);
    ok = ok && waited == &slot && slot.ram_temp_pin_count == 1;
    if (waited)
        ram.ram_unpin(*waited);

    // The unified Hit path has the same atomic-pinning contract used by
    // get_expert(..., pin_for_caller=true).
    RamClaimResult hit = ram.claim_or_wait(
        engine, 0, 7, RamClaimCaller::MainThread, nullptr,
        /*pin_on_hit=*/true);
    ok = ok && hit.status == RamClaimStatus::Hit && hit.slot == &slot &&
         slot.ram_temp_pin_count == 1;
    if (hit.slot)
        ram.ram_unpin(*hit.slot);

    ok = ok && slot.ram_temp_pin_count == 0;
    ok = test_ram_sync_promotion_capacity() && ok;
    std::cout << "  -> " << (ok ? "PASSED" : "FAILED") << "\n";
    return ok;
}

// ------------------------------------------------------------------
// 21. Phase 3 (VRAM): the REAL GpuTopology cache path -- upload_expert +
//     find_expert + a real fused-kernel dispatch -- with two genuinely
//     different-sized real experts sharing ONE fixed-size slot pool.
//
// Unlike Test 18 (which used the simpler, direct offload_tensor_to_vram/
// dense_matmul_host helper Test 5 also uses -- correct, but not the real
// production cache path), this drives the actual GpuTopology::upload_
// expert/find_expert machinery every real decode call goes through.
//
// The real question Phase 3 needed answered: GpuTopology's VRAM pool is
// ONE fixed slot size for the whole pool (reindex_expert_vram_cache,
// offset += expert_slab_bytes per slot) -- does uploading a SMALLER
// expert into a slot sized for the LARGER one actually work correctly,
// or does it read/write the wrong number of bytes? Traced through the
// real source (gpu_topology.cpp) before writing this test: every upload
// path (upload_expert, upload_expert_async, upload_expert_active[_async])
// already computes `copy_bytes = std::min(bytes, slot->size)` -- i.e. it
// was ALREADY built to copy exactly the real caller-supplied byte count
// into a possibly-larger slot, not a fixed slot-sized copy. This predates
// per-expert mixed precision entirely -- it's what already made
// GLM-5.2-style CROSS-layer type heterogeneity (Q2_K vs Q3_K per layer,
// one pool sized to the global max) work. This test confirms the same
// mechanism, unmodified, genuinely extends to WITHIN-layer heterogeneity:
// no VRAM-tier code changes were needed for Phase 3, only real
// verification that the claim holds.
// ------------------------------------------------------------------
bool test_vram_cache_mixed_sizes() {
    std::cout << "[Test 21] Phase 3: real GpuTopology cache (upload_expert/find_expert) with two "
                 "different-sized real experts in one fixed-size slot pool...\n";

    const std::string prod_path = "/home/ray/vllm-workspace/GLM-5.3-GGUF-align";
    const std::string iq1s_path = "/home/ray/vllm-workspace/GLM-5.3-GGUF-align/"
        "repacked-GLM-5.3-UD-IQ1_S-00005-of-00006-part00001-of-00001.gguf";

    struct ExpertSpec { std::string src_path; int src_layer; int src_expert; };
    // expert 0 = large (baseline), expert 1 = small (IQ1_S) -- real sizes,
    // real bytes, same two sources every other test in this file uses.
    std::vector<ExpertSpec> specs = {
        {prod_path, 20, 5},
        {iq1s_path, 60, 7},
    };

    struct RealExpert {
        std::vector<uint8_t> slab;  // gate @ 0, up @ gb, down @ gb+ub -- matches populate_packed_slab's layout
        size_t gb = 0, ub = 0, db = 0;
        GGUFType gtype, utype, dtype;
    };
    std::vector<RealExpert> experts;

    for (auto& spec : specs) {
        GGUFFile f;
        if (!f.load_header(spec.src_path)) {
            std::cout << "  -> SKIPPED (real source not available: " << spec.src_path << ")\n";
            return true;
        }
        RealExpert re;
        GGUFType* types[3] = {&re.gtype, &re.utype, &re.dtype};
        size_t* sizes[3] = {&re.gb, &re.ub, &re.db};
        int ti = 0;
        for (const char* kind : {"gate", "up", "down"}) {
            std::string tn = "blk." + std::to_string(spec.src_layer) + ".ffn_" + kind + "_exps.weight";
            auto it = f.tensors.find(tn);
            if (it == f.tensors.end()) { std::cout << "  -> FAILED (tensor not found: " << tn << ")\n"; return false; }
            const GGUFTensor& t = it->second;
            uint64_t n_experts = t.shape.back();
            uint64_t bpe = t.size_bytes / n_experts;
            uint64_t eoff = t.offset + (uint64_t)spec.src_expert * bpe;
            size_t base = re.slab.size();
            re.slab.resize(base + bpe);
            if (pread(f.shard_fds[t.shard_idx], re.slab.data() + base, bpe, eoff) != (ssize_t)bpe) {
                std::cout << "  -> FAILED (short read)\n"; return false;
            }
            *sizes[ti] = bpe;
            *types[ti] = t.type;
            ++ti;
        }
        experts.push_back(std::move(re));
    }

    size_t slab_bytes[2] = {experts[0].slab.size(), experts[1].slab.size()};
    size_t pool_slot_size = std::max(slab_bytes[0], slab_bytes[1]);
    std::cout << "  expert 0 (baseline) slab=" << slab_bytes[0]
              << "  expert 1 (IQ1_S) slab=" << slab_bytes[1]
              << "  pool_slot_size(=max)=" << pool_slot_size << "\n";

    CudaManager gpu;
    if (!gpu.initialize()) {
        std::cout << "  -> SKIPPED (no CUDA device)\n";
        return true;
    }
    // Force the vault pool onto the SAME GPU dense_matmul_host below
    // targets (CudaManager::device_id, "Core engine GPU" -- can genuinely
    // differ from topology.vault_gpus on a real multi-GPU box, confirmed
    // on this one: upload_expert placed the slot on the real vault GPU
    // while dense_matmul_host's own cudaSetDevice(device_id) targeted the
    // core GPU, an unintended cross-device pointer access -- "illegal
    // memory access", not a bug in upload_expert or dense_matmul_host
    // individually, just this test's simplified single-device helper
    // meeting a genuinely multi-GPU cache placement it wasn't told to
    // match). Production code dispatches multi-GPU-aware; this override
    // keeps the test scoped to Phase 3's real question (does variable-size
    // upload/dispatch work on a fixed-size slot pool) without also having
    // to reimplement cross-GPU dispatch here.
    gpu.topology.vault_gpus = {gpu.device_id};
    // No config passed -- allocate_expert_cache's own documented fallback
    // for a caller without one ("safe default... e.g. tests"): grabs
    // available VRAM, sized to pool_slot_size per slot. n_experts_total=2
    // is just a hint for internal bookkeeping array sizing, not a hard cap.
    gpu.topology.allocate_expert_cache(pool_slot_size, 2, nullptr);

    bool all_ok = true;
    for (int e = 0; e < 2; ++e) {
        ExpertLocation* loc = gpu.topology.upload_expert(0, e, experts[e].slab.data(), experts[e].slab.size());
        if (!loc) { std::cout << "  -> FAILED (upload_expert returned null for expert " << e << ")\n"; return false; }
        if (loc->size != pool_slot_size) {
            std::cout << "  -> FAILED (slot size=" << loc->size << ", expected the pool's fixed " << pool_slot_size << ")\n";
            all_ok = false;
        }

        ExpertLocation* found = gpu.topology.find_expert(0, e);
        if (found != loc) { std::cout << "  -> FAILED (find_expert didn't return the same slot just uploaded)\n"; all_ok = false; }

        // Real dispatch against each projection, at its own real offset
        // WITHIN the slot (gate@0, up@gb, down@gb+ub -- exactly what
        // populate_packed_slab/dispatch_expert_k assume) and its own real
        // type -- the same shape of call the production hot path makes.
        int I_gate = 6144, O_gate = 2048, I_down = 2048, O_down = 6144;
        struct Proj { const char* name; size_t off; size_t bytes; GGUFType type; int I, O; };
        Proj projs[3] = {
            {"gate", 0,                              experts[e].gb, experts[e].gtype, I_gate, O_gate},
            {"up",   experts[e].gb,                  experts[e].ub, experts[e].utype, I_gate, O_gate},
            {"down", experts[e].gb + experts[e].ub,   experts[e].db, experts[e].dtype, I_down, O_down},
        };
        std::mt19937 rng(9999 + e);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& p : projs) {
            std::vector<float> x(p.I);
            for (auto& v : x) v = dist(rng);
            std::vector<float> y_gpu(p.O, -999.0f);
            bool ok = gpu.dense_matmul_host(p.type, loc->device_ptr + p.off, x.data(), y_gpu.data(), 1, p.I, p.O);
            if (!ok) { std::cout << "  -> FAILED (expert " << e << " " << p.name << ": dense_matmul_host failed)\n"; all_ok = false; continue; }

            // uint8_t, not float -- a real bug this test hit on its first
            // real run: declaring this as std::vector<float> still compiles
            // (p.bytes silently becomes an element count, not a byte count)
            // but makes host_bytes.data() + o*rb below stride by o*rb
            // FLOATS (4x too far) instead of o*rb bytes. Row 0 (offset 0
            // either way) came out byte-correct; every later row read from
            // the wrong place -- confirmed via a one-off debug dump showing
            // y_gpu[0]==y_cpu[0] exactly but every later element diverging,
            // which is what pointed at a stride bug specifically rather
            // than a real GPU-side one (a real raw D2H readback, also
            // added and removed during that same debugging pass, had
            // already confirmed the uploaded VRAM bytes were byte-for-byte
            // identical to what was uploaded).
            std::vector<uint8_t> host_bytes(p.bytes);
            std::memcpy(host_bytes.data(), experts[e].slab.data() + p.off, p.bytes);
            int rb = (int)(p.bytes / p.O);
            std::vector<float> wrow(p.I), y_cpu(p.O, 0.0f);
            for (int o = 0; o < p.O; ++o) {
                ggml_dequant_row((uint32_t)p.type, host_bytes.data() + (size_t)o * rb, wrow.data(), p.I);
                double acc = 0.0;
                for (int i = 0; i < p.I; ++i) acc += (double)x[i] * (double)wrow[i];
                y_cpu[o] = (float)acc;
            }
            float max_abs = 0, max_rel = 0;
            for (int o = 0; o < p.O; ++o) {
                float d = std::abs(y_cpu[o] - y_gpu[o]);
                max_abs = std::max(max_abs, d);
                max_rel = std::max(max_rel, d / std::max(1e-6f, std::abs(y_cpu[o])));
            }
            bool proj_ok = (max_abs < 1e-2f) || (max_rel < 0.02f);
            std::cout << "  expert=" << e << " " << p.name << " (in-slot offset=" << p.off << " bytes=" << p.bytes
                      << "): " << (proj_ok ? "OK" : "SUSPECT") << " max_abs=" << max_abs << " max_rel=" << max_rel << "\n";
            if (!proj_ok) all_ok = false;
        }
    }

    std::cout << "  -> " << (all_ok ? "PASSED" : "FAILED")
              << " (real upload_expert/find_expert cache path, two different-sized real experts, "
                 "one fixed-size slot pool, real per-projection dispatch all checked)\n";
    return all_ok;
}

// Test 23: fast, deterministic repro (milliseconds, no model/disk needed) of
// the real compute-sanitizer finding from a live GLM-5.3 generation run --
// 1375 flagged cudaErrorHostMemoryAlreadyRegistered "errors" inside
// upload_expert_async's speculative cudaHostRegister, once
// ram_pinned_budget_gb had persistently pinned most RAM-tier slabs (see
// upload_expert_async's own header comment). Reproduces the EXACT CUDA call
// sequence directly: manually cudaHostRegister a buffer (standing in for
// ensure_slab_capacity's persistent pin), then call the real
// upload_expert_async on it with the fixed, real production call shape
// (src_already_pinned=true) -- the permanent regression guard, always runs,
// expected to be sanitizer-clean.
//
// Set ZEROMOE_REPRO_SANITIZER_ISSUE=1 to ALSO run a second, deliberate
// call with the hint omitted (src_already_pinned=false, the pre-fix shape)
// -- this is what actually reproduces the flagged-but-harmless
// cudaErrorHostMemoryAlreadyRegistered under compute-sanitizer (confirmed:
// `ZEROMOE_REPRO_SANITIZER_ISSUE=1 compute-sanitizer --tool memcheck
// ./zeromoe_tests` -> exactly 2 errors, both attributed to this call, 0
// anywhere else). Off by default so a normal `compute-sanitizer` pass over
// this whole suite stays a trustworthy "0 errors" signal -- this
// demonstration call is not itself a bug, it's the OLD, still-correct
// fallback shape that any future caller forgetting to pass the hint would
// still safely (if noisily, under sanitizer) fall back to.
bool test_upload_async_already_pinned_src() {
    std::cout << "[Test 23] upload_expert_async + already-cudaHostRegister'd source "
                 "(fast repro of the real compute-sanitizer finding)...\n";

    CudaManager gpu;
    if (!gpu.initialize()) {
        std::cout << "  -> SKIPPED (no CUDA device)\n";
        return true;
    }
    gpu.topology.vault_gpus = {gpu.device_id};
    const size_t buf_bytes = 128 * 1024;  // small and arbitrary -- only the
                                           // cudaHostRegister interaction matters here
    EngineConfig cache_config;
    cache_config.gpus.emplace_back();
    cache_config.gpus.back().device_id = gpu.device_id;
    // This pinning test needs two tiny experts, not the allocator's default
    // whole-card cache. Leave headroom for other native GPU fixtures.
    cache_config.gpus.back().moe_cache_gb = double(4 * buf_bytes) / 1e9;
    gpu.topology.allocate_expert_cache(buf_bytes, 2, &cache_config);

    auto make_registered_buf = [&](uint32_t seed) -> std::vector<uint8_t> {
        std::vector<uint8_t> buf(buf_bytes);
        std::mt19937 rng(seed);
        for (auto& b : buf) b = (uint8_t)(rng() & 0xFF);
        // Simulate ensure_slab_capacity's persistent pin under
        // ram_pinned_budget_gb -- the exact state that makes
        // upload_expert_async's own speculative register call doomed.
        cudaError_t err = cudaHostRegister(buf.data(), buf_bytes, cudaHostRegisterPortable);
        if (err != cudaSuccess) {
            std::cout << "  -> SKIPPED (cudaHostRegister setup failed: "
                      << cudaGetErrorString(err) << ")\n";
            buf.clear();
        }
        return buf;
    };

    auto verify_upload = [&](const char* label, ExpertLocation* loc,
                              const std::vector<uint8_t>& src) -> bool {
        if (!loc) { std::cout << "  -> FAILED (" << label << ": upload_expert_async returned null)\n"; return false; }
        cudaDeviceSynchronize();
        std::vector<uint8_t> readback(buf_bytes);
        cudaError_t err = cudaMemcpy(readback.data(), loc->device_ptr, buf_bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cout << "  -> FAILED (" << label << ": readback cudaMemcpy: " << cudaGetErrorString(err) << ")\n";
            return false;
        }
        bool ok = (readback == src);
        std::cout << "  " << label << ": " << (ok ? "OK (VRAM bytes match)" : "MISMATCH") << "\n";
        return ok;
    };

    bool all_ok = true;
    std::vector<uint8_t> src_a;

    // Deliberate, opt-in-only demonstration of the OLD (still-correct,
    // still-safe) no-hint shape -- this is the one that actually reproduces
    // the flagged sanitizer error, since host_data is already registered by
    // this test above and upload_expert_async will still try
    // cudaHostRegister on it speculatively when not told otherwise.
    if (std::getenv("ZEROMOE_REPRO_SANITIZER_ISSUE")) {
        src_a = make_registered_buf(1001);
        if (src_a.empty()) return true;  // setup skip already logged
        ExpertLocation* loc_a = gpu.topology.upload_expert_async(0, 0, src_a.data(), buf_bytes);
        all_ok &= verify_upload("no-hint call (expect a flagged-but-harmless sanitizer error above)", loc_a, src_a);
    } else {
        std::cout << "  no-hint demonstration: SKIPPED by default (set ZEROMOE_REPRO_SANITIZER_ISSUE=1 "
                     "to reproduce the flagged-but-harmless sanitizer error this test's fix removes)\n";
    }

    // The permanent regression guard: real production call shape after the
    // fix -- src_already_pinned=true. Always runs; expected sanitizer-clean.
    std::vector<uint8_t> src_b = make_registered_buf(2002);
    if (src_b.empty()) return true;
    ExpertLocation* loc_b = gpu.topology.upload_expert_async(0, 1, src_b.data(), buf_bytes, -1, /*src_already_pinned=*/true);
    all_ok &= verify_upload("with-hint call (expect zero sanitizer errors)", loc_b, src_b);

    if (!src_a.empty()) cudaHostUnregister(src_a.data());
    cudaHostUnregister(src_b.data());

    std::cout << "  -> " << (all_ok ? "PASSED" : "FAILED")
              << " (both calls uploaded correct bytes; compare a compute-sanitizer log around "
                 "this test's two calls to confirm the hint actually removes the flagged error)\n";
    return all_ok;
}






// 2026-09-04: GPU-side routed-expert accumulation (kernels/math/expert_accum.cu).
// An end-to-end output diff CANNOT validate this: the engine is already
// non-deterministic run-to-run (cache-state-dependent dispatch), AND the
// feature legitimately changes summation order (device-accumulated experts are
// summed first, then added onto CPU-dispatched ones). So this compares the
// kernel directly against the exact CPU loop it replaces, on fixed data.
//
// Shapes chosen so a stride/index bug cannot alias into a pass: n != D, tokens
// deliberately REPEATED across experts (so the accumulate really accumulates
// rather than just scattering), one token left untouched by any expert (must
// stay exactly zero), and non-uniform weights.
bool test_expert_gpu_accum() {
    std::cout << "[Test 26] MoE GPU-side expert accumulation (device scatter-add vs CPU loop)...\n";
    CudaManager gpu;
    if (!gpu.initialize()) {
        std::cout << "  -> Skipped (No CUDA Device Found on system)\n";
        return true;
    }
    const int S = 7, D = 20, n_experts = 3;
    // pairs: (expert, token, weight) -- token 5 is touched by NO expert
    const std::vector<std::vector<int>> ex_tok = {{0, 2, 3}, {1, 2}, {4, 6, 0}};
    std::mt19937 rng(777);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);

    std::vector<int> tok; std::vector<float> w;
    for (auto& e : ex_tok) for (int t : e) { tok.push_back(t); w.push_back(d(rng)); }
    const int total_pairs = (int)tok.size();

    // per-expert output rows
    std::vector<std::vector<float>> eout;
    for (auto& e : ex_tok) { std::vector<float> v((size_t)e.size() * D); for (auto& x : v) x = d(rng); eout.push_back(v); }

    // CPU reference: exactly the loop this replaces
    std::vector<float> ref((size_t)S * D, 0.0f);
    { int off = 0;
      for (size_t e = 0; e < ex_tok.size(); ++e) {
        for (size_t i = 0; i < ex_tok[e].size(); ++i) {
            const int s = tok[off + i]; const float wt = w[off + i];
            for (int c = 0; c < D; ++c) ref[(size_t)s * D + c] += wt * eout[e][i * D + c];
        }
        off += (int)ex_tok[e].size();
      } }

    std::vector<float> got((size_t)S * D, 0.0f);
    if (!gpu.expert_accum_begin(0, S, D, tok.data(), w.data(), total_pairs)) {
        std::cout << "  -> FAILED: expert_accum_begin returned false\n"; return false;
    }
    int off = 0;
    for (size_t e = 0; e < ex_tok.size(); ++e) {
        const int n = (int)ex_tok[e].size();
        float* d_src = nullptr;
        if (cudaMalloc((void**)&d_src, (size_t)n * D * sizeof(float)) != cudaSuccess) {
            std::cout << "  -> FAILED: cudaMalloc\n"; return false;
        }
        cudaMemcpy(d_src, eout[e].data(), (size_t)n * D * sizeof(float), cudaMemcpyHostToDevice);
        if (!gpu.expert_accum_add(0, d_src, off, n, D, S)) {
            std::cout << "  -> FAILED: expert_accum_add\n"; cudaFree(d_src); return false;
        }
        cudaDeviceSynchronize();
        cudaFree(d_src);
        off += n;
    }
    if (!gpu.expert_accum_end(0, got.data(), S, D)) {
        std::cout << "  -> FAILED: expert_accum_end returned false\n"; return false;
    }

    double max_abs = 0.0, refmax = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) { max_abs = std::max(max_abs, (double)std::fabs(got[i]-ref[i]));
                                              refmax = std::max(refmax, (double)std::fabs(ref[i])); }
    // token 5 must be exactly untouched
    double untouched = 0.0;
    for (int c = 0; c < D; ++c) untouched = std::max(untouched, (double)std::fabs(got[(size_t)5*D + c]));
    std::cout << "  -> max_abs=" << max_abs << " normalized=" << (max_abs/(refmax+1e-9))
              << " untouched_row_max=" << untouched << "\n";
    if (untouched != 0.0) { std::cout << "  -> FAILED: a row no expert routed to was written\n"; return false; }
    if (!(max_abs/(refmax+1e-9) < 1e-6)) { std::cout << "  -> FAILED: diverges from the CPU loop\n"; return false; }
    std::cout << "  -> PASSED\n";
    return true;
}



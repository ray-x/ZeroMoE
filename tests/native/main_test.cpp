#include "native_tests.hpp"

bool test_exl3_ggml_graph_capture_config() {
    std::cout << "[Test] EXL3 GGML graph-capture config...\n";
    char config_path[] = "/tmp/moestream_exl3_graph_capture_config_XXXXXX";
    const int config_fd = mkstemp(config_path);
    if (config_fd < 0)
        return false;
    close(config_fd);
    const auto write_config = [&](const std::string &extra) {
        std::ofstream output(config_path);
        output << "model_path = \"/test/model.gguf\"\n"
                  << extra
                  << "[[gpus]]\n"
                     "device_id = 0\n"
                     "name = \"test\"\n"
                     "role = \"dense+moe\"\n"
                     "vram_gb = 24.0\n"
                     "dense_gb = 4.0\n"
                     "kv_cache_gb = 2.0\n"
                     "moe_cache_gb = 8.0\n";
        return output.good();
    };
    bool ok = write_config("");
    EngineConfig defaults;
    ok = ok && defaults.load(config_path) &&
         !defaults.exl3_ggml_graph_capture;
    ok = ok && write_config("exl3_ggml_graph_capture = true\n");
    EngineConfig invalid;
    ok = ok && invalid.load(config_path) && !invalid.validate();
    ok = ok && write_config(
                  "exl3_ggml_bridge_model_path = \"/test/exl3\"\n"
                  "exl3_ggml_bridge_layer = 0\n"
                  "exl3_ggml_graph_capture = true\n"
                  "exl3_gpu_cache_gb = 1.0\n"
                  "exl3_cpu_expert_cache_gb = 0.0\n");
    EngineConfig enabled;
    ok = ok && enabled.load(config_path) && enabled.validate() &&
         enabled.exl3_ggml_graph_capture;
    std::remove(config_path);
    std::cout << "  -> " << (ok ? "PASSED" : "FAILED") << "\n";
    return ok;
}

bool test_exl3_cpu_dispatch() {
    std::cout << "[Test] EXL3 CPU dispatch and reconstruction...\n";
    constexpr int width = 128;
    std::vector<uint16_t> signs(width, fp32_to_fp16(1.0f));
    std::vector<float> input(width, 1.0f), output(width), weight(width * width);
    for (int bits : {4, 5, 6}) {
        std::vector<uint16_t> trellis(8 * 8 * 16 * bits, 0);
        Exl3Mul1MatrixView matrix{trellis.data(), signs.data(), signs.data(),
                                  nullptr, width, width, bits};
        const auto packed = exl3_mul1_pack_matrix(matrix);
        if (!cpu_quant_matmul(GGUFType::EXL3, packed.data(), input.data(),
                               output.data(), 1, width, width) ||
            !exl3_mul1_reconstruct(matrix, weight.data())) return false;
        const float expected = width * fp16_to_fp32(fp32_to_fp16(
            std::fma(1024.0f, fp16_to_fp32(0x1eee), fp16_to_fp32(0xc931))));
        if (std::abs(output[0] - expected) > 1e-3f) return false;
        for (int column = 1; column < width; ++column)
            if (std::abs(output[column]) > 1e-3f) return false;
        for (int column = 0; column < width; ++column) {
            float total = 0.0f;
            for (int row = 0; row < width; ++row)
                total += weight[row * width + column];
            if (std::abs(output[column] - total) > 1e-3f) return false;
        }
    }
    std::cout << "  -> PASSED\n";
    return true;
}

bool test_exl3_cpu_matrix_cache() {
    std::cout << "[Test] EXL3 DDR matrix cache LRU and held entries...\n";
    const auto matrix = [](uint8_t fill) {
        Exl3Mul1MatrixData value;
        value.trellis.assign(16, fill);
        value.suh.assign(8, fill);
        value.svh.assign(8, fill);
        value.input_features = 128;
        value.output_features = 128;
        value.bits = 4;
        return value;
    };
    Exl3CpuMatrixCache cache(64);
    auto held = cache.insert("expert-a.gate_proj", matrix(1));
    cache.insert("expert-b.gate_proj", matrix(2));
    auto hit = cache.get("expert-a.gate_proj");
    if (!hit || hit->trellis.front() != 1 ||
        cache.stats().resident_bytes != 64)
        return false;
    cache.insert("expert-c.gate_proj", matrix(3));
    if (cache.get("expert-b.gate_proj") ||
        held->trellis.front() != 1 || hit->trellis.front() != 1)
      return false;
    const auto stats = cache.stats();
    const bool passed = stats.evictions == 1 && stats.resident_bytes == 64 &&
                        stats.hits == 1 && stats.misses == 1;
    std::cout << "  " << (passed ? "PASSED" : "FAILED") << "\n";
    return passed;
}

bool test_exl3_gpu_hc(int device) {
    constexpr int width = 37, streams = 3, rank = 17, rows = 3;
    constexpr int expanded = width * streams;
    std::mt19937 random(8001);
    std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
    std::vector<float> down(expanded * rank), up(rank * expanded), inject(expanded * streams);
    std::vector<float> normed(rows * expanded);
    for (auto *values : {&down, &up, &inject, &normed})
        for (float &value : *values) value = distribution(random);
    const size_t bytes = (down.size() + up.size() + inject.size()) * sizeof(float);
    Exl3GpuExecutor reference(device, 2 * bytes);
    std::vector<float> low(rows * rank), gate(rows * expanded), expected_injection(rows * streams);
    reference.matmul("hc.down", static_cast<uint32_t>(GGUFType::F32),
                     reinterpret_cast<const uint8_t *>(down.data()), down.size() * sizeof(float),
                     normed.data(), low.data(), rows, expanded, rank);
    for (float &value : low) {
        value /= static_cast<float>(streams);
        value /= 1.0f + std::exp(-value);
    }
    reference.matmul("hc.up", static_cast<uint32_t>(GGUFType::F32),
                     reinterpret_cast<const uint8_t *>(up.data()), up.size() * sizeof(float),
                     low.data(), gate.data(), rows, rank, expanded);
    for (float &value : gate) value = 1.0f / (1.0f + std::exp(-value));
    reference.matmul("hc.inject", static_cast<uint32_t>(GGUFType::F32),
                     reinterpret_cast<const uint8_t *>(inject.data()), inject.size() * sizeof(float),
                     normed.data(), expected_injection.data(), rows, expanded, streams);
    const auto expected = [&](bool divide_each_stream) {
        std::vector<float> output(rows * width, 0.0f);
        for (int row = 0; row < rows; ++row) {
            for (int column = 0; column < width; ++column) {
                for (int stream_index = 0; stream_index < streams; ++stream_index) {
                    const size_t offset = (static_cast<size_t>(row) * streams + stream_index) * width + column;
                    float value = normed[offset] * gate[offset];
                    if (divide_each_stream) value /= static_cast<float>(streams);
                    output[row * width + column] += value;
                }
                if (!divide_each_stream) output[row * width + column] /= static_cast<float>(streams);
            }
        }
        return output;
    };
    const auto close = [](const std::vector<float> &actual, const std::vector<float> &reference_values) {
        double error = 0.0, energy = 0.0;
        for (size_t index = 0; index < actual.size(); ++index) {
            if (!std::isfinite(actual[index])) return false;
            const double difference = actual[index] - reference_values[index];
            error += difference * difference;
            energy += static_cast<double>(reference_values[index]) * reference_values[index];
        }
        return std::sqrt(error / std::max(energy, 1e-20)) < 1e-5;
    };
    const Exl3GpuF32Weight down_weight{"hc.down", down.data(), expanded, rank};
    const Exl3GpuF32Weight up_weight{"hc.up", up.data(), rank, expanded};
    const Exl3GpuF32Weight inject_weight{"hc.inject", inject.data(), expanded, streams};
    Exl3GpuExecutor executor(device, 2 * bytes);
    std::vector<float> actual(rows * width), actual_injection(rows * streams);
    executor.hc_mix(down_weight, up_weight, &inject_weight, normed.data(), actual.data(),
                    actual_injection.data(), rows, width, streams, true);
    if (!close(actual, expected(true)) || !close(actual_injection, expected_injection)) return false;
    executor.hc_mix(down_weight, up_weight, nullptr, normed.data(), actual.data(),
                    nullptr, rows, width, streams, false);
    if (!close(actual, expected(false))) return false;
    const auto stats = executor.stats();
    if (stats.hc_mixes != 2 || stats.uploads != 3 || stats.cache_hits != 2) return false;
    bool rejected = false;
    try {
        executor.hc_mix(down_weight, up_weight, nullptr, normed.data(), actual.data(),
                        nullptr, 0, width, streams, false);
    } catch (const std::runtime_error &) { rejected = true; }
    std::cout << "  device=" << device << " HC device chain "
              << (rejected ? "PASSED" : "FAILED") << "\n";
    return rejected;
}

bool test_exl3_gpu_routed_device_async(int device) {
    std::cout << "[Test] EXL3 asynchronous routed cache and shared scratch...\n";
    constexpr int hidden = 128;
    constexpr int intermediate = 128;
    constexpr int bits = 4;
    std::mt19937 random(90210);
    std::uniform_real_distribution<float> distribution(-0.4f, 0.4f);
    std::unordered_map<std::string, std::vector<uint8_t>> packed_matrices;
    for (int expert = 0; expert < 3; ++expert) {
        for (int projection = 0; projection < 3; ++projection) {
            std::vector<uint16_t> trellis(8 * 8 * 16 * bits);
            std::vector<uint16_t> suh(hidden), svh(intermediate);
            for (uint16_t &word : trellis)
                word = static_cast<uint16_t>(random());
            for (uint16_t &value : suh)
                value = fp32_to_fp16(distribution(random));
            for (uint16_t &value : svh)
                value = fp32_to_fp16(distribution(random));
            Exl3Mul1MatrixView matrix{trellis.data(), suh.data(), svh.data(),
                                      nullptr, hidden, intermediate, bits};
            packed_matrices.emplace(
                "model.language_model.layers.0.mlp.experts." +
                    std::to_string(expert) + "." +
                    (projection == 0 ? "gate_proj" :
                     projection == 1 ? "up_proj" : "down_proj"),
                exl3_mul1_pack_matrix(matrix));
        }
    }
    const size_t matrix_bytes = packed_matrices.begin()->second.size();
    for (const auto &[key, matrix] : packed_matrices)
        if (matrix.size() != matrix_bytes)
            return false;

    const auto load = [&](const std::string &key) {
        const auto found = packed_matrices.find(key);
        if (found == packed_matrices.end())
            throw std::runtime_error("missing async EXL3 matrix: " + key);
        return found->second;
    };
    const auto reference = [&](const std::vector<int32_t> &route_ids,
                               const std::vector<float> &route_weights,
                               const std::vector<float> &input,
                               std::vector<float> &output) {
        output.assign(hidden, 0.0f);
        for (size_t route = 0; route < route_ids.size(); ++route) {
            const std::string prefix =
                "model.language_model.layers.0.mlp.experts." +
                std::to_string(route_ids[route]) + ".";
            const auto gate = load(prefix + "gate_proj");
            const auto up = load(prefix + "up_proj");
            const auto down = load(prefix + "down_proj");
            std::vector<float> gate_value(intermediate), up_value(intermediate);
            std::vector<float> activation(intermediate), expert_output(hidden);
            if (!exl3_mul1_matmul_packed(gate.data(), input.data(),
                                        gate_value.data(), 1, hidden,
                                        intermediate) ||
                !exl3_mul1_matmul_packed(up.data(), input.data(),
                                         up_value.data(), 1, hidden,
                                         intermediate))
                return false;
            for (int index = 0; index < intermediate; ++index)
                activation[index] = gate_value[index] /
                    (1.0f + std::exp(-gate_value[index])) * up_value[index];
            if (!exl3_mul1_matmul_packed(down.data(), activation.data(),
                                        expert_output.data(), 1,
                                        intermediate, hidden))
                return false;
            for (int index = 0; index < hidden; ++index)
                output[index] += route_weights[route] * expert_output[index];
        }
        return true;
    };
    const auto close = [](const std::vector<float> &actual,
                          const std::vector<float> &expected) {
        double error = 0.0;
        double energy = 0.0;
        for (size_t index = 0; index < actual.size(); ++index) {
            if (!std::isfinite(actual[index]))
                return false;
            const double difference = actual[index] - expected[index];
            error += difference * difference;
            energy += static_cast<double>(expected[index]) * expected[index];
        }
        // fp16-activation upstream kernels over a gate/up/down chain land near
        // 1e-3 relative RMS; scratch corruption between streams is O(1).
        return std::sqrt(error / std::max(energy, 1e-20)) < 1e-2;
    };

    struct DeviceResources {
        cudaStream_t streams[2] = {};
        float *inputs[3] = {};
        float *outputs[3] = {};
        float *partials[2] = {};
        std::vector<uint8_t *> packed_weights;
        ~DeviceResources() {
            for (cudaStream_t stream : streams)
                if (stream)
                    cudaStreamSynchronize(stream);
            for (float *pointer : inputs)
                if (pointer)
                    cudaFree(pointer);
            for (float *pointer : outputs)
                if (pointer)
                    cudaFree(pointer);
            for (float *pointer : partials)
                if (pointer)
                    cudaFree(pointer);
            for (uint8_t *pointer : packed_weights)
                if (pointer)
                    cudaFree(pointer);
            for (cudaStream_t stream : streams)
                if (stream)
                    cudaStreamDestroy(stream);
        }
    } device_resources;

    try {
        if (cudaSetDevice(device) != cudaSuccess ||
            cudaStreamCreateWithFlags(&device_resources.streams[0],
                                      cudaStreamNonBlocking) != cudaSuccess ||
            cudaStreamCreateWithFlags(&device_resources.streams[1],
                                      cudaStreamNonBlocking) != cudaSuccess)
            return false;
        std::vector<std::vector<float>> host_inputs(3,
                                                    std::vector<float>(hidden));
        for (auto &input : host_inputs)
            for (float &value : input)
                value = distribution(random);
        for (int index = 0; index < 3; ++index)
            if (cudaMalloc(&device_resources.inputs[index],
                           hidden * sizeof(float)) != cudaSuccess ||
                cudaMalloc(&device_resources.outputs[index],
                           hidden * sizeof(float)) != cudaSuccess)
                return false;
        if (cudaMemcpyAsync(device_resources.inputs[0], host_inputs[0].data(),
                            hidden * sizeof(float), cudaMemcpyHostToDevice,
                            device_resources.streams[0]) != cudaSuccess ||
            cudaMemcpyAsync(device_resources.inputs[1], host_inputs[1].data(),
                            hidden * sizeof(float), cudaMemcpyHostToDevice,
                            device_resources.streams[1]) != cudaSuccess ||
            cudaMemcpyAsync(device_resources.inputs[2], host_inputs[2].data(),
                            hidden * sizeof(float), cudaMemcpyHostToDevice,
                            device_resources.streams[0]) != cudaSuccess)
            return false;

        const std::vector<std::string> prefixes = {
            "model.language_model.layers.0.mlp.experts.0.",
            "model.language_model.layers.0.mlp.experts.1.",
            "model.language_model.layers.0.mlp.experts.2."};
        const std::vector<int32_t> routes_a = {0, 1};
        const std::vector<float> weights_a = {0.3f, 0.7f};
        const std::vector<int32_t> routes_b = {1, 0};
        const std::vector<float> weights_b = {0.6f, 0.4f};
        const std::vector<int32_t> routes_c = {2};
        const std::vector<float> weights_c = {1.0f};
        std::vector<float> expected_a, expected_b, expected_c;
        if (!reference(routes_a, weights_a, host_inputs[0], expected_a) ||
            !reference(routes_b, weights_b, host_inputs[1], expected_b) ||
            !reference(routes_c, weights_c, host_inputs[2], expected_c))
            return false;

        Exl3GpuExecutor executor(device, 6 * matrix_bytes);
        executor.routed_ffn_device(
            prefixes, routes_a.data(), weights_a.data(),
            device_resources.inputs[0], device_resources.outputs[0], 1, 2,
            hidden, intermediate, device_resources.streams[0], false, load);
        executor.routed_ffn_device(
            prefixes, routes_b.data(), weights_b.data(),
            device_resources.inputs[1], device_resources.outputs[1], 1, 2,
            hidden, intermediate, device_resources.streams[1], false, load);
        executor.routed_ffn_device(
            prefixes, routes_c.data(), weights_c.data(),
            device_resources.inputs[2], device_resources.outputs[2], 1, 1,
            hidden, intermediate, device_resources.streams[0], false, load);
        if (cudaStreamSynchronize(device_resources.streams[0]) != cudaSuccess ||
            cudaStreamSynchronize(device_resources.streams[1]) != cudaSuccess)
            return false;
        std::vector<float> actual_a(hidden), actual_b(hidden), actual_c(hidden);
        if (cudaMemcpy(actual_a.data(), device_resources.outputs[0],
                       hidden * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(actual_b.data(), device_resources.outputs[1],
                       hidden * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(actual_c.data(), device_resources.outputs[2],
                       hidden * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
            return false;
        const Exl3GpuStats stats = executor.stats();
        const bool passed = close(actual_a, expected_a) &&
                            close(actual_b, expected_b) &&
                            close(actual_c, expected_c) &&
                            stats.routed_ffns == 3 && stats.matmuls == 15 &&
                            stats.uploads == 9 && stats.evictions == 3 &&
                            stats.resident_bytes <= 6 * matrix_bytes;
        std::cout << "  device=" << device << " async routed cache and "
                  << "cross-stream scratch "
                  << (passed ? "PASSED" : "FAILED") << "\n";
        if (!passed)
            return false;

        const auto device_view = [&](const std::string &key, int input_width,
                                     int output_width) {
            const auto &packed = load(key);
            Exl3PackedMatrixHeader header{};
            if (packed.size() < sizeof(header))
                throw std::runtime_error("truncated external test matrix");
            std::memcpy(&header, packed.data(), sizeof(header));
            if (!exl3_valid_packed_header(header, packed.size()) ||
                header.input_features != input_width ||
                header.output_features != output_width)
                throw std::runtime_error("invalid external test matrix header");
            uint8_t *device_packed = nullptr;
            if (cudaMalloc(&device_packed, packed.size()) != cudaSuccess)
                throw std::runtime_error("external test matrix allocation failed");
            device_resources.packed_weights.push_back(device_packed);
            if (cudaMemcpy(device_packed, packed.data(), packed.size(),
                           cudaMemcpyHostToDevice) != cudaSuccess)
                throw std::runtime_error("external test matrix upload failed");
            return Exl3CudaMatrixView{
                reinterpret_cast<const uint16_t *>(device_packed +
                                                   header.trellis_offset),
                reinterpret_cast<const uint16_t *>(device_packed +
                                                   header.suh_offset),
                reinterpret_cast<const uint16_t *>(device_packed +
                                                   header.svh_offset),
                (header.flags & kExl3PackedHasBias)
                    ? reinterpret_cast<const uint16_t *>(device_packed +
                                                         header.bias_offset)
                    : nullptr,
                input_width, output_width, static_cast<int>(header.bits)};
        };
        std::vector<Exl3GpuExternalExpert> external_experts(2);
        std::vector<std::weak_ptr<int>> weak_holds(2);
        for (int route = 0; route < 2; ++route) {
            const std::string prefix = prefixes[static_cast<size_t>(route)];
            external_experts[static_cast<size_t>(route)].gate = device_view(
                prefix + "gate_proj", hidden, intermediate);
            external_experts[static_cast<size_t>(route)].up = device_view(
                prefix + "up_proj", hidden, intermediate);
            external_experts[static_cast<size_t>(route)].down = device_view(
                prefix + "down_proj", intermediate, hidden);
            auto hold = std::make_shared<int>(route);
            weak_holds[static_cast<size_t>(route)] = hold;
            external_experts[static_cast<size_t>(route)].hold = hold;
        }
        Exl3GpuExecutor external_executor(device, matrix_bytes);
        external_executor.routed_ffn_device_external(
            external_experts.data(), weights_a.data(), nullptr,
            device_resources.inputs[0], device_resources.outputs[0], 1, 2,
            hidden, intermediate, device_resources.streams[0]);
        external_executor.routed_ffn_device_external(
            external_experts.data(), weights_a.data(), nullptr,
            device_resources.inputs[0], device_resources.outputs[0], 1, 2,
            hidden, intermediate, device_resources.streams[0]);
        external_experts.clear();
        if (cudaStreamSynchronize(device_resources.streams[0]) != cudaSuccess)
            return false;
        std::vector<float> external_output(hidden);
        if (cudaMemcpy(external_output.data(), device_resources.outputs[0],
                       hidden * sizeof(float), cudaMemcpyDeviceToHost) !=
            cudaSuccess)
            return false;
        const Exl3GpuStats external_stats = external_executor.stats();
        const bool external_passed =
            close(external_output, expected_a) &&
            weak_holds[0].expired() && weak_holds[1].expired() &&
            external_stats.routed_ffns == 2 &&
            external_stats.matmuls == 12 && external_stats.uploads == 0;
        std::cout << "  device=" << device << " externally-owned routed weights "
                  << (external_passed ? "PASSED" : "FAILED") << "\n";
        if (!external_passed)
            return false;

        const auto make_packed = [&](int bits) {
            std::vector<uint16_t> trellis(8 * 8 * 16 * bits);
            std::vector<uint16_t> suh(hidden), svh(intermediate);
            for (uint16_t &word : trellis)
                word = static_cast<uint16_t>(random());
            for (uint16_t &value : suh)
                value = fp32_to_fp16(distribution(random));
            for (uint16_t &value : svh)
                value = fp32_to_fp16(distribution(random));
            Exl3Mul1MatrixView matrix{trellis.data(), suh.data(), svh.data(),
                                      nullptr, hidden, intermediate, bits};
            return exl3_mul1_pack_matrix(matrix);
        };
        const auto mixed_gate_packed = make_packed(3);
        const auto mixed_up_packed = make_packed(5);
        const auto upload_view = [&](const std::vector<uint8_t> &packed) {
            Exl3PackedMatrixHeader header{};
            if (packed.size() < sizeof(header))
                throw std::runtime_error("truncated mixed-bit test matrix");
            std::memcpy(&header, packed.data(), sizeof(header));
            if (!exl3_valid_packed_header(header, packed.size()))
                throw std::runtime_error("invalid mixed-bit test matrix");
            uint8_t *device_packed = nullptr;
            if (cudaMalloc(&device_packed, packed.size()) != cudaSuccess)
                throw std::runtime_error("mixed-bit matrix allocation failed");
            device_resources.packed_weights.push_back(device_packed);
            if (cudaMemcpy(device_packed, packed.data(), packed.size(),
                           cudaMemcpyHostToDevice) != cudaSuccess)
                throw std::runtime_error("mixed-bit matrix upload failed");
            return Exl3CudaMatrixView{
                reinterpret_cast<const uint16_t *>(device_packed +
                                                   header.trellis_offset),
                reinterpret_cast<const uint16_t *>(device_packed +
                                                   header.suh_offset),
                reinterpret_cast<const uint16_t *>(device_packed +
                                                   header.svh_offset),
                nullptr, hidden, intermediate, static_cast<int>(header.bits)};
        };
        const auto mixed_gate = upload_view(mixed_gate_packed);
        const auto mixed_up = upload_view(mixed_up_packed);
        if (cudaMalloc(&device_resources.partials[0],
                       intermediate * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&device_resources.partials[1],
                       intermediate * sizeof(float)) != cudaSuccess)
            return false;
        std::vector<float> mixed_gate_output(intermediate);
        std::vector<float> mixed_up_output(intermediate);
        std::vector<float> mixed_activation(intermediate);
        if (!exl3_mul1_matmul_packed(
                mixed_gate_packed.data(), host_inputs[0].data(),
                mixed_gate_output.data(), 1, hidden, intermediate) ||
            !exl3_mul1_matmul_packed(
                mixed_up_packed.data(), host_inputs[0].data(),
                mixed_up_output.data(), 1, hidden, intermediate))
            return false;
        for (int index = 0; index < intermediate; ++index)
            mixed_activation[static_cast<size_t>(index)] =
                mixed_gate_output[static_cast<size_t>(index)] /
                (1.0f + std::exp(-mixed_gate_output[static_cast<size_t>(index)])) *
                mixed_up_output[static_cast<size_t>(index)];
        if (launch_exl3_mul1_gate_up(
                mixed_gate, mixed_up, device_resources.inputs[0],
                device_resources.outputs[1], 1, device_resources.partials[0],
                device_resources.partials[1], device_resources.streams[0]) !=
                cudaSuccess ||
            cudaStreamSynchronize(device_resources.streams[0]) != cudaSuccess)
            return false;
        std::vector<float> mixed_actual(intermediate);
        if (cudaMemcpy(mixed_actual.data(), device_resources.outputs[1],
                       intermediate * sizeof(float), cudaMemcpyDeviceToHost) !=
            cudaSuccess)
            return false;
        const bool mixed_passed = close(mixed_actual, mixed_activation);
        std::cout << "  device=" << device << " mixed-bit gate/up (3/5) "
                  << (mixed_passed ? "PASSED" : "FAILED") << "\n";
        if (!mixed_passed)
            return false;

        Exl3GgmlCachedRoutedFfnState route_state;
        if (!route_state.initialize(&external_executor, prefixes, 3, 2, hidden,
                                    intermediate))
            return false;
        cudaPointerAttributes route_attributes{};
        if (cudaPointerGetAttributes(&route_attributes,
                                     route_state.route_ids_host) !=
                cudaSuccess ||
            route_attributes.type != cudaMemoryTypeHost)
            return false;
        const std::vector<int32_t> host_route_ids = {0, 1, 1, 2, 2, 0};
        const std::vector<float> host_route_weights =
            {0.2f, 0.8f, 0.3f, 0.7f, 0.4f, 0.6f};
        if (!route_state.set_host_routes(host_route_ids.data(),
                                         host_route_weights.data()))
            return false;
        int32_t *device_route_ids = nullptr;
        cudaStream_t route_stream = nullptr;
        if (cudaMalloc(&device_route_ids,
                       host_route_ids.size() * sizeof(int32_t)) != cudaSuccess ||
            cudaStreamCreateWithFlags(&route_stream, cudaStreamNonBlocking) !=
                cudaSuccess) {
            if (device_route_ids)
                cudaFree(device_route_ids);
            return false;
        }
        const bool route_copy_ok =
            cudaMemcpy(device_route_ids, host_route_ids.data(),
                       host_route_ids.size() * sizeof(int32_t),
                       cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpyAsync(route_state.route_ids_host, device_route_ids,
                            host_route_ids.size() * sizeof(int32_t),
                            cudaMemcpyDeviceToHost, route_stream) ==
                cudaSuccess &&
            cudaStreamSynchronize(route_stream) == cudaSuccess &&
            std::equal(host_route_ids.begin(), host_route_ids.end(),
                       route_state.route_ids_host);
        cudaStreamDestroy(route_stream);
        cudaFree(device_route_ids);
        std::cout << "  device=" << device << " pinned route D2H staging "
                  << (route_copy_ok ? "PASSED" : "FAILED") << "\n";
        return route_copy_ok;
    } catch (const std::exception &error) {
        std::cerr << "  async routed cache error: " << error.what() << "\n";
        return false;
    }
}

bool test_exl3_cuda_kernels() {
    std::cout << "[Test] EXL3 CUDA matmul/reconstruction/cache...\n";
    int device_count = 0;
    int original_device = 0;
    if (cudaGetDevice(&original_device) != cudaSuccess) return false;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        return false;
    const int width = 128;
    std::mt19937 random(4242);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (int device = 0; device < device_count; ++device) {
        if (cudaSetDevice(device) != cudaSuccess) return false;
        for (int bits = 1; bits <= 8; ++bits) {
            const int input_width = bits <= 3 ? width : 640;
            const int output_width = bits <= 3 ? 256 : 384;
            const int rows = bits == 5 ? 17 : bits == 6 ? 3 : 1;
            std::vector<uint16_t> trellis(input_width / 16 * (output_width / 16) * 16 * bits);
            std::vector<uint16_t> suh(input_width), svh(output_width), bias(output_width);
            for (auto &word : trellis) word = static_cast<uint16_t>(random());
            for (auto &value : suh) value = fp32_to_fp16(distribution(random));
            for (auto &value : svh) value = fp32_to_fp16(distribution(random));
            for (auto &value : bias) value = fp32_to_fp16(distribution(random) * 0.01f);
            Exl3Mul1MatrixView host{trellis.data(), suh.data(), svh.data(), bits % 2 ? bias.data() : nullptr,
                                    input_width, output_width, bits};
            auto packed = exl3_mul1_pack_matrix(host);
            Exl3PackedMatrixHeader header{};
            std::memcpy(&header, packed.data(), sizeof(header));
            std::vector<float> input(rows * input_width), expected(rows * output_width);
            for (auto &value : input) value = distribution(random);
            if (!exl3_mul1_matmul(host, input.data(), expected.data(), rows)) return false;
            void *device_packed = nullptr;
            float *device_input = nullptr, *device_output = nullptr;
            float *partials = nullptr, *device_weight = nullptr;
            float *device_accumulator = nullptr;
            const auto release = [&] {
                cudaFree(device_packed); cudaFree(device_input); cudaFree(device_output);
                cudaFree(partials); cudaFree(device_weight); cudaFree(device_accumulator);
            };
            if (cudaMalloc(&device_packed, packed.size()) != cudaSuccess ||
                cudaMalloc(&device_input, input.size() * sizeof(float)) != cudaSuccess ||
                cudaMalloc(&device_output, expected.size() * sizeof(float)) != cudaSuccess ||
                cudaMalloc(&partials, 2 * expected.size() * exl3_cuda_key_splits(input_width) * sizeof(float)) != cudaSuccess ||
                cudaMalloc(&device_weight, static_cast<size_t>(input_width) * output_width * sizeof(float)) != cudaSuccess ||
                cudaMalloc(&device_accumulator, expected.size() * sizeof(float)) != cudaSuccess) {
                release(); return false;
            }
            cudaMemcpy(device_packed, packed.data(), packed.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(device_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
            const auto *base = static_cast<const uint8_t *>(device_packed);
            Exl3CudaMatrixView gpu{
                reinterpret_cast<const uint16_t *>(base + header.trellis_offset),
                reinterpret_cast<const uint16_t *>(base + header.suh_offset),
                reinterpret_cast<const uint16_t *>(base + header.svh_offset),
                host.bias ? reinterpret_cast<const uint16_t *>(base + header.bias_offset) : nullptr,
                input_width, output_width, bits};
            if (launch_exl3_mul1_matmul(gpu, device_input, device_output, rows,
                                          partials) != cudaSuccess) {
                release(); return false;
            }
            std::vector<float> actual(expected.size());
            if (cudaMemcpy(actual.data(), device_output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
                release(); return false;
            }
            const auto close = [](const std::vector<float> &left, const std::vector<float> &right) {
                double error = 0.0, energy = 0.0;
                for (size_t index = 0; index < left.size(); ++index) {
                    if (!std::isfinite(left[index])) return false;
                    const double difference = left[index] - right[index];
                    error += difference * difference;
                    energy += static_cast<double>(right[index]) * right[index];
                }
                // Upstream ExLlamaV3 kernels take fp16 activations (~5e-4 relative RMS)
                // and, for one row at <= 5 bits on Ampere, int8 activations (~7e-3).
                return std::sqrt(error / std::max(energy, 1e-20)) < 2e-2;
            };
            bool ok = close(actual, expected);
            std::vector<float> expected_activation = expected;
            for (float &value : expected_activation)
                value = value / (1.0f + std::exp(-value)) * value;
            if (launch_exl3_mul1_gate_up(
                    gpu, gpu, device_input, device_output, rows, partials,
                    partials + expected.size() * exl3_cuda_key_splits(input_width)) != cudaSuccess ||
                cudaMemcpy(actual.data(), device_output,
                           actual.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
                ok = false;
            ok = ok && close(actual, expected_activation);
            if (!CudaKernelDispatcher::matmul(GGUFType::EXL3, device_output, device_input,
                                                device_packed, rows, input_width, output_width)) ok = false;
            if (cudaMemcpy(actual.data(), device_output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
                ok = false;
            ok = ok && close(actual, expected);
            std::vector<float> expected_accumulation = expected;
            for (float &value : expected_accumulation) value *= 0.25f;
            cudaMemset(device_accumulator, 0, expected.size() * sizeof(float));
            if (launch_exl3_mul1_matmul_accumulate(
                    gpu, device_input, device_accumulator, 0.75f, rows,
                    partials) != cudaSuccess ||
                launch_exl3_mul1_matmul_accumulate(
                    gpu, device_input, device_accumulator, -0.5f, rows,
                    partials) != cudaSuccess ||
                cudaMemcpy(actual.data(), device_accumulator,
                           actual.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
                ok = false;
            ok = ok && close(actual, expected_accumulation);
            auto malformed = header;
            malformed.trellis_offset += 2;
            cudaMemcpy(device_packed, &malformed, sizeof(malformed), cudaMemcpyHostToDevice);
            if (exl3_valid_packed_header(malformed, packed.size()) ||
                CudaKernelDispatcher::matmul(GGUFType::EXL3, device_output, device_input,
                                              device_packed, rows, input_width, output_width))
                ok = false;
            if (launch_exl3_mul1_matmul(gpu, device_input, device_output, 0,
                                          partials) != cudaErrorInvalidValue)
                ok = false;
            release();
            Exl3GpuExecutor executor(device, 2 * packed.size());
            for (int attempt = 0; attempt < 4; ++attempt) {
                const std::string key = "expert" + std::to_string(attempt);
                executor.matmul(key, static_cast<uint32_t>(GGUFType::EXL3), packed.data(),
                                  packed.size(), input.data(), actual.data(), rows,
                                  input_width, output_width, false);
                ok = ok && close(actual, expected);
            }
            ok = ok && executor.matmul_cached("expert3", input.data(), actual.data(), rows, input_width, output_width);
            ok = ok && close(actual, expected);
            const auto stats = executor.stats();
            ok = ok && stats.matmuls == 5 && stats.cache_hits == 1 && stats.evictions == 2 &&
                 stats.buffer_reuses == 2 &&
                 stats.resident_bytes <= 2 * packed.size();
            Exl3Mul1MatrixView down_matrix{host.trellis, host.svh, host.suh, nullptr,
                                            output_width, input_width, bits};
            const auto down_packed = exl3_mul1_pack_matrix(down_matrix);
            std::vector<float> activated(output_width), expert_result(input_width);
            for (int column = 0; column < output_width; ++column)
                activated[column] = expected[column] / (1.0f + std::exp(-expected[column])) * expected[column];
            if (!exl3_mul1_matmul(down_matrix, activated.data(), expert_result.data(), 1)) return false;
            std::vector<float> combined(input_width), combined_expected(input_width);
            for (int column = 0; column < input_width; ++column)
                combined_expected[column] = 0.25f * expert_result[column] + 0.75f * expert_result[column];
            const float routing_weights[] = {0.25f, 0.75f};
            size_t load_count = 0;
            const auto load = [&](const std::string &key) {
                ++load_count;
                return key.ends_with("down_proj") ? down_packed : packed;
            };
            const size_t ffn_budget = 4 * packed.size() + 2 * down_packed.size();
            Exl3GpuExecutor ffn_executor(device, ffn_budget);
            for (int attempt = 0; attempt < 3; ++attempt) {
                const std::vector<std::string> prefixes = attempt < 2
                    ? std::vector<std::string>{"expert0.", "expert1."}
                    : std::vector<std::string>{"expert1.", "expert2."};
                ffn_executor.routed_ffn(prefixes, routing_weights, input.data(), combined.data(),
                                         input_width, output_width, load);
                ok = ok && close(combined, combined_expected);
            }
            const auto ffn_stats = ffn_executor.stats();
            ok = ok && load_count == 9 && ffn_stats.routed_ffns == 3 && ffn_stats.matmuls == 18 &&
                 ffn_stats.evictions == 3 && ffn_stats.resident_bytes <= ffn_budget;
            Exl3GpuExecutor held_executor(device, 2 * packed.size());
            bool held_rejected = false;
            try {
                held_executor.routed_ffn({"held."}, routing_weights, input.data(), combined.data(),
                                          input_width, output_width, load);
            } catch (const std::runtime_error &) { held_rejected = true; }
            held_executor.matmul("after-failure", static_cast<uint32_t>(GGUFType::EXL3),
                                   packed.data(), packed.size(), input.data(), actual.data(), rows,
                                   input_width, output_width, false);
            ok = ok && held_rejected && close(actual, expected);
            std::cout << "  device=" << device << " bits=" << bits << " rows=" << rows
                      << " " << (ok ? "PASSED" : "FAILED") << "\n";
            if (!ok) return false;
        }
        const int input_width = 7, output_width = 11, rows = 3;
        std::vector<float> weights(input_width * output_width), input(rows * input_width);
        std::vector<float> expected(rows * output_width, 0.0f), actual(expected.size());
        for (auto &value : weights) value = distribution(random);
        for (auto &value : input) value = distribution(random);
        for (int row = 0; row < rows; ++row)
            for (int column = 0; column < output_width; ++column)
                for (int feature = 0; feature < input_width; ++feature)
                    expected[row * output_width + column] +=
                        input[row * input_width + feature] * weights[column * input_width + feature];
        Exl3GpuExecutor dense_executor(device, weights.size() * sizeof(float));
        dense_executor.matmul("dense", static_cast<uint32_t>(GGUFType::F32),
                              reinterpret_cast<const uint8_t *>(weights.data()), weights.size() * sizeof(float),
                              input.data(), actual.data(), rows, input_width, output_width);
        for (size_t index = 0; index < expected.size(); ++index)
            if (!std::isfinite(actual[index]) || std::abs(actual[index] - expected[index]) > 1e-5f)
                return false;
        if (!dense_executor.matmul_cached("dense", input.data(), actual.data(), rows, input_width, output_width) ||
            dense_executor.matmul_cached("missing", input.data(), actual.data(), rows, input_width, output_width))
            return false;
        bool rejected = false;
        try {
            dense_executor.matmul("over-budget", static_cast<uint32_t>(GGUFType::F32),
                                  reinterpret_cast<const uint8_t *>(weights.data()), weights.size() * sizeof(float),
                                  input.data(), actual.data(), rows, input_width, output_width, false);
        } catch (const std::runtime_error &) { rejected = true; }
        if (!rejected || dense_executor.stats().evictions != 0) return false;
        std::cout << "  device=" << device << " F32 dispatch and pinned-cache budget PASSED\n";
        if (!test_exl3_gpu_hc(device)) return false;
        if (!test_exl3_gpu_routed_device_async(device)) return false;
    }
    return cudaSetDevice(original_device) == cudaSuccess;
}

int main() {
    if (std::getenv("MOESTREAM_TEST_ONLY_RAM_PROMOTION"))
        return test_ram_async_source_hold() ? 0 : 1;
    if (std::getenv("MOESTREAM_TEST_ONLY_EXL3_GGML_CONFIG"))
        return test_exl3_ggml_graph_capture_config() ? 0 : 1;
    if (!test_exl3_cpu_dispatch()) return 1;
    if (!test_exl3_cpu_matrix_cache()) return 1;
    if (std::getenv("MOESTREAM_TEST_EXL3_CPU_ONLY")) return 0;
    if (!test_graph_config()) return 1;
    if (!test_exl3_cuda_kernels()) return 1;
    if (std::getenv("MOESTREAM_TEST_EXL3_ONLY")) return 0;
    std::cout << "========================================\n";
    std::cout << "  MoeStream Native Test Harness\n";
    std::cout << "========================================\n";

    if (std::getenv("MOESTREAM_TEST_ONLY_Q2_0_VNNI")) {
        return test_qwen_q2_0_vnni_kernel() ? 0 : 1;
    }
    if (std::getenv("MOESTREAM_TEST_ONLY_QWEN_STATE_HANDOFF_CONFIG")) {
        return test_qwen_prefill_state_handoff_config() ? 0 : 1;
    }

    // Diagnostic-only escape hatch (2026-09-10): run just Test 35 in
    // isolation so compute-sanitizer's overhead is tractable -- the full
    // suite's CPU-only tests dominate wall time and add nothing to a
    // race-condition investigation scoped to GgmlQwenSharedRunner.
    if (std::getenv("MOESTREAM_TEST_ONLY_35")) {
        bool ok = test_qwen_shared_expert_runner();
        return ok ? 0 : 1;
    }
    if (std::getenv("MOESTREAM_TEST_ONLY_36")) {
        bool ok = test_qwen_hc_runner();
        return ok ? 0 : 1;
    }
    if (std::getenv("MOESTREAM_TEST_ONLY_37")) {
        bool ok = test_qwen_hc_runner_q8_exact();
        return ok ? 0 : 1;
    }
    if (std::getenv("MOESTREAM_TEST_ONLY_QWEN_CPU_BENCH")) {
        bool ok = bench_qwen_cpu_moe();
        return ok ? 0 : 1;
    }
    if (std::getenv("MOESTREAM_TEST_ONLY_Q3K_CPU_BENCH")) {
        bench_q3k_cpu_only();
        return 0;
    }
#ifdef __linux__
    if (std::getenv("MOESTREAM_TEST_ONLY_Q3K_THP_BENCH")) {
        bench_q3k_thp_cpu_only();
        return 0;
    }
#endif
    if (std::getenv("MOESTREAM_TEST_ONLY_CPU_IQ")) {
        // Keep both diagnostics independent.  In particular, an IQ1_M
        // failure must not hide the Qwen graph test, and a graph failure must
        // not hide the lower-level quantized-kernel result.
        const bool iq_ok = test_cpu_iq_matmul_paths();
        const bool graph_ok = test_qwen_cpu_moe_graph();
        const bool iq1m_graph_ok = test_qwen_cpu_moe_iq1m_graph();
        const bool mixed_graph_ok = test_qwen_cpu_moe_mixed_strided_graph();
        const bool iq2_graph_ok = test_qwen_cpu_moe_iq2_strided_graph();
        return (iq_ok && graph_ok && iq1m_graph_ok && mixed_graph_ok &&
                iq2_graph_ok)
                   ? 0
                   : 1;
    }
    if (std::getenv("MOESTREAM_TEST_ONLY_CUDA_FUSED")) {
        return test_cuda_fused_kernels() ? 0 : 1;
    }
    if (std::getenv("MOESTREAM_TEST_ONLY_COMPACT_CACHE")) {
        return test_compact_vram_slot_planner() ? 0 : 1;
    }

    test_cpu_matrix_ops();
    test_graph_initializer();
    test_attention_extraction();
    test_cuda_execution();
    bool fused_kernels_ok = test_cuda_fused_kernels();
    bool avx512_dequant_ok = test_avx512_dequant();
    bool avx512_dequant_iq_ok = test_avx512_dequant_iq();
    bool cpu_iq_matmul_ok = test_cpu_iq_matmul_paths();
    bool qwen_cpu_moe_graph_ok = test_qwen_cpu_moe_graph();
    bool qwen_cpu_moe_iq1m_graph_ok = test_qwen_cpu_moe_iq1m_graph();
    bool qwen_cpu_moe_mixed_strided_graph_ok =
        test_qwen_cpu_moe_mixed_strided_graph();
    bool qwen_cpu_moe_iq2_strided_graph_ok =
        test_qwen_cpu_moe_iq2_strided_graph();
    // Test 8 is throughput-only (no pass/fail verdict, see its own header
    // comment) and runs many timed dequant/matmul passes -- by far the
    // slowest thing in this binary, and irrelevant to every other test's
    // pass/fail verdict. Skipped by default so a normal `make test`/targeted
    // re-run (e.g. Test 11's real-expert comparison via
    // MOESTREAM_REAL_EXPERT_BIN/MOESTREAM_REAL_ACTIVATION_BIN) stays fast;
    // set MOESTREAM_RUN_BENCH=1 to opt back into the throughput numbers.
    if (std::getenv("MOESTREAM_RUN_BENCH")) {
        bench_cpu_dequant_matmul();
        bench_qabsorb_cpu_vs_gpu();
        bench_iq2xxs_vnni_prototype();
        bench_iq3xxs_avx2_prototype();
    } else {
        std::cout << "\n[Test 8] SKIPPED by default (set MOESTREAM_RUN_BENCH=1 to run the throughput bench)\n";
    }
    bool bf16_dense_ok = test_bf16_cublas_dense_matmul();
    bool bf16_matched_dense_ok = test_bf16_matched_cublas_dense_matmul();
    bool bf16_batched_qabsorb_ok = test_bf16_batched_qabsorb();
    bool score_gpu_ok = test_score_gpu();
    bool score_gpu_nope_only_ok = test_score_gpu_nope_only();
    bool wuv_batched_gpu_ok = test_wuv_batched_gpu();
    bool wuv_int8_batched_gpu_ok = test_wuv_int8_batched_gpu();
    bool wuv_int8_batched_prefill_ok = test_wuv_int8_batched_gpu_prefill();
    bool expert_gpu_accum_ok = test_expert_gpu_accum();
    bool dense_bf16_kquant_ok = test_dense_bf16_kquant();
    bool gdn_conv1d_gpu_ok = test_gdn_conv1d_gpu();
    bool gdn_conv_fused_ok = test_gdn_conv_fused();
    bool qwen_shared_expert_ok = test_qwen_shared_expert_runner();
    bool qwen_hc_runner_ok = test_qwen_hc_runner();
    bool qwen_hc_q8_exact_ok = test_qwen_hc_runner_q8_exact();
    bool score_gpu_prefill_ok = test_score_gpu_prefill();
    bool wuv_batched_prefill_ok = test_wuv_batched_gpu_prefill();
    bool real_expert_iq2xxs_ok = test_real_expert_iq2xxs_dequant();
    bool mixed_precision_layer_ok = test_mixed_precision_layer();
    bool unpacked_relaxed_check_ok = test_unpacked_format_relaxed_check();
    bool slab_bytes_max_ok = test_compute_expert_slab_bytes_max();
    bool compact_vram_planner_ok = test_compact_vram_slot_planner();
    bool ram_async_source_hold_ok = test_ram_async_source_hold();
    bool vram_mixed_sizes_ok = test_vram_cache_mixed_sizes();
    bool upload_async_pinned_src_ok = test_upload_async_already_pinned_src();
    bool glm5next_kda_gpu_ok = test_glm5next_kda_gpu();
    bool glm5next_router_gpu_ok = test_glm5next_router_gpu();
    bool glm5next_dense_projection_fusion_ok =
        test_glm5next_dense_projection_fusion();
    bool gpu_device_allowlist_ok = test_gpu_device_allowlist();
    bool qwen_state_handoff_config_ok = test_qwen_prefill_state_handoff_config();
    bool exl3_graph_capture_config_ok =
        test_exl3_ggml_graph_capture_config();

    std::cout << "========================================\n";
    if (!fused_kernels_ok) {
        std::cout << "❌ CUDA fused kernel correctness test FAILED -- see [Test 5] output above.\n";
        return 1;
    }
    if (!avx512_dequant_ok) {
        std::cout << "❌ AVX-512 dequant correctness test FAILED -- see [Test 6] output above.\n";
        return 1;
    }
    if (!avx512_dequant_iq_ok) {
        std::cout << "❌ AVX-512 IQ dequant correctness test FAILED -- see [Test 7] output above.\n";
        return 1;
    }
    if (!cpu_iq_matmul_ok) {
        std::cout << "❌ CPU IQ1_M/IQ2_XXS/IQ3_XXS matmul test FAILED -- see [Test 7b] output above.\n";
        return 1;
    }
    if (!qwen_cpu_moe_graph_ok) {
        std::cout << "❌ Qwen persistent CPU GGML MoE graph test FAILED -- see [Test 7c] output above.\n";
        return 1;
    }
    if (!qwen_cpu_moe_iq1m_graph_ok) {
        std::cout << "❌ Qwen IQ1_M CPU GGML MoE graph test FAILED -- see [Test 7d] output above.\n";
        return 1;
    }
    if (!qwen_cpu_moe_mixed_strided_graph_ok) {
        std::cout << "❌ Qwen mixed-type strided CPU GGML MoE graph test FAILED -- see [Test 7e] output above.\n";
        return 1;
    }
    if (!qwen_cpu_moe_iq2_strided_graph_ok) {
        std::cout << "❌ Qwen IQ2_XXS strided CPU GGML MoE graph test FAILED -- see [Test 7f] output above.\n";
        return 1;
    }
    if (!bf16_dense_ok) {
        std::cout << "❌ Test 9: cuBLAS unexpectedly ACCEPTED the mismatched-type call AND produced a "
                     "wrong result -- see [Test 9] output above (this is not the normal 'rejected as "
                     "expected' outcome).\n";
        return 1;
    }
    if (!bf16_matched_dense_ok) {
        std::cout << "❌ BF16 matched-precision cuBLAS dense matmul correctness test FAILED -- see [Test 10] output above.\n";
        return 1;
    }
    if (!bf16_batched_qabsorb_ok) {
        std::cout << "❌ BF16 strided-batched qabsorb correctness test FAILED -- see [Test 12] output above.\n";
        return 1;
    }
    if (!score_gpu_prefill_ok) {
        std::cout << "❌ Score GPU prefill (S>1, causal mask) correctness test FAILED -- see [Test 16] output above.\n";
        return 1;
    }
    if (!wuv_batched_prefill_ok) {
        std::cout << "❌ WUV batched prefill (S>1 scatter) correctness test FAILED -- see [Test 17] output above.\n";
        return 1;
    }
    if (!score_gpu_ok) {
        std::cout << "❌ Score GPU path correctness test FAILED -- see [Test 13] output above.\n";
        return 1;
    }
    if (!score_gpu_nope_only_ok) {
        std::cout << "❌ Score GPU nope-only/DSA path FAILED -- see [Test 33] output above.\n";
        return 1;
    }
    if (!wuv_batched_gpu_ok) {
        std::cout << "❌ WUV batched GPU path correctness test FAILED -- see [Test 14] output above.\n";
        return 1;
    }
    if (!wuv_int8_batched_gpu_ok) {
        std::cout << "❌ WUV int8 batched GPU path correctness test FAILED -- see [Test 15] output above.\n";
        return 1;
    }
    if (!dense_bf16_kquant_ok) {
        std::cout << "❌ Dense BF16 Q5_K/Q6_K path FAILED -- see [Test 27] output above.\n";
        return 1;
    }
    if (!gdn_conv1d_gpu_ok) {
        std::cout << "❌ GDN causal conv1d+SiLU GPU path FAILED -- see [Test 28] output above.\n";
        return 1;
    }
    if (!gdn_conv_fused_ok) {
        std::cout << "❌ GDN proj+conv1d fused GPU path FAILED -- see [Test 29] output above.\n";
        return 1;
    }
    if (!qwen_shared_expert_ok) {
        std::cout << "❌ Qwen4Exp shared-expert GGML runner FAILED -- see [Test 35] output above.\n";
        return 1;
    }
    if (!qwen_hc_runner_ok) {
        std::cout << "❌ Qwen4Exp HC GGML runner FAILED -- see [Test 36] output above.\n";
        return 1;
    }
    if (!qwen_hc_q8_exact_ok) {
        std::cout << "❌ Qwen4Exp HC exact Q8_0 path FAILED -- see [Test 37] output above.\n";
        return 1;
    }
    if (!expert_gpu_accum_ok) {
        std::cout << "❌ MoE GPU-side expert accumulation FAILED -- see [Test 26] output above.\n";
        return 1;
    }
    if (!wuv_int8_batched_prefill_ok) {
        std::cout << "❌ WUV int8 batched GPU path FAILED at S>1 -- see [Test 22] output above.\n";
        return 1;
    }
    if (!real_expert_iq2xxs_ok) {
        std::cout << "❌ Real-Data IQ2_XXS dequant consistency test FAILED -- see [Test 11] output above.\n";
        return 1;
    }
    if (!mixed_precision_layer_ok) {
        std::cout << "❌ Mixed-precision-within-one-layer test FAILED -- see [Test 18] output above.\n";
        return 1;
    }
    if (!unpacked_relaxed_check_ok) {
        std::cout << "❌ sanity_check_unpacked_format() relaxation test FAILED -- see [Test 19] output above.\n";
        return 1;
    }
    if (!slab_bytes_max_ok) {
        std::cout << "❌ compute_expert_slab_bytes() max-size test FAILED -- see [Test 20] output above.\n";
        return 1;
    }
    if (!compact_vram_planner_ok) {
        std::cout << "❌ compact VRAM slot planner test FAILED -- see [Test 20b] output above.\n";
        return 1;
    }
    if (!ram_async_source_hold_ok) {
        std::cout << "❌ RAM async source hold test FAILED -- see [Test 20c] output above.\n";
        return 1;
    }
    if (!vram_mixed_sizes_ok) {
        std::cout << "❌ VRAM cache mixed-sizes test FAILED -- see [Test 21] output above.\n";
        return 1;
    }
    if (!upload_async_pinned_src_ok) {
        std::cout << "❌ upload_expert_async already-pinned-source test FAILED -- see [Test 23] output above.\n";
        return 1;
    }
    if (!glm5next_kda_gpu_ok) {
        std::cout << "❌ GLM5NEXT KDA GPU path FAILED -- see [Test 30] output above.\n";
        return 1;
    }
    if (!glm5next_router_gpu_ok) {
        std::cout << "❌ GLM5NEXT native router GPU path FAILED -- see [Test 31] output above.\n";
        return 1;
    }
    if (!glm5next_dense_projection_fusion_ok) {
        std::cout << "❌ GLM5NEXT KDA dense projection fusion FAILED -- see [Test 34] output above.\n";
        return 1;
    }
    if (!gpu_device_allowlist_ok) {
        std::cout << "❌ GPU config allow-list test FAILED -- see [Test 32] output above.\n";
        return 1;
    }
    if (!qwen_state_handoff_config_ok) {
        std::cout << "❌ Qwen prefill state handoff config test FAILED.\n";
        return 1;
    }
    if (!exl3_graph_capture_config_ok) {
        std::cout << "❌ EXL3 GGML graph-capture config test FAILED.\n";
        return 1;
    }
    std::cout << "✅ All core engine tests passed successfully!\n";
    return 0;
}

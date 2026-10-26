#include "fixture.hpp"
#include "qwen4exp_mtp_graph.hpp"

#include <cstdio>
#include <filesystem>
#include <stdexcept>

int main(int argc, char **argv) {
  if (argc != 7) {
    std::fprintf(stderr, "usage: %s TRUNK.gguf MTP.gguf fixture.txt dump.txt "
                         "DEVICE DEVICE_BUDGET_MIB\n", argv[0]);
    return 2;
  }
  try {
    const mtp_test::Fixture fixture(argv[3]);
    if (fixture.rows.size() < 32)
      throw std::runtime_error("parity gate requires at least 32 replay rows");
    if (std::filesystem::exists(argv[4]))
      throw std::runtime_error("refusing to overwrite an existing parity dump");
    const int device = std::stoi(argv[5]);
    const auto budget_mib = std::stoull(argv[6]);
    if (!budget_mib || budget_mib > SIZE_MAX / (1 << 20))
      throw std::runtime_error("invalid device budget");
    auto engine = std::make_shared<StreamingEngine>();
    if (std::filesystem::is_directory(argv[1]) ||
        std::filesystem::is_directory(argv[2]) ||
        !engine->model_file.load_header(argv[1]) ||
        !engine->mtp_model_file.load_header(argv[2]))
      throw std::runtime_error("must load explicit trunk and MTP GGUF files");
    engine->has_mtp_model = true;
    engine->initialize_resident_memory({"output.weight"});
    std::vector<std::string> names;
    const auto prefix = "blk." +
        std::to_string(engine->model_file.get_config_int("block_count", 0)) + ".";
    for (const auto &[name, meta] : engine->mtp_model_file.tensors)
      if (name.starts_with(prefix)) names.push_back(name);
    engine->initialize_resident_memory(names, &engine->mtp_model_file);

    engine->hc_stream_buf = {1.0f, 2.0f, 3.0f};
    engine->qwen4exp_recur_state = {{4.0f, 5.0f}};
    engine->qwen4exp_conv_history = {{6.0f, 7.0f}};
    engine->qwen4exp_ple_conv_history = {{8.0f, 9.0f}};
    engine->qwen4exp_ple_tok_hist = {3, 2, 1};
    engine->qwen4exp_ple_next_pos = 17;
    std::string error;
    auto runtime = qwen4exp_mtp::create(engine,
        {device, static_cast<int>(fixture.rows.size()),
         budget_mib * (1 << 20), true}, error);
    if (!runtime) throw std::runtime_error(error);
    const auto builds = runtime->build_count();
    const int experts = engine->model_file.get_config_int("expert_count", 0);
    const int topk = engine->model_file.get_config_int("expert_used_count", 0);
    std::ofstream output(argv[4]);
    if (!output) throw std::runtime_error("cannot open parity dump");
    fixture.dump_header(output);
    std::vector<float> hidden(fixture.W), logits(fixture.V);
    for (int phase = 0; phase < 3; ++phase) {
      if (!runtime->reset(error)) throw std::runtime_error(error);
      const int chunk = phase == 1 ? 7 : static_cast<int>(fixture.rows.size());
      for (size_t begin = 0; begin < fixture.rows.size(); begin += chunk)
        for (size_t i = begin; i < std::min(begin + chunk, fixture.rows.size()); ++i) {
          const auto &row = fixture.rows[i];
          if (!runtime->replay(
                  {row.position, row.embedding, row.hidden, hidden, logits}, error))
            throw std::runtime_error(error);
          mtp_test::dump(output, phase, i, "hidden", hidden);
          mtp_test::dump(output, phase, i, "logits", logits);
          for (const std::string stage : {"bridge", "post_attention", "post_ffn",
                                          "router_probabilities", "router_weights"}) {
            const int size = stage == "router_probabilities" ? experts :
                             stage == "router_weights" ? topk : fixture.W;
            std::vector<float> values(size);
            if (!runtime->read_diagnostic(stage, values, error))
              throw std::runtime_error(error);
            mtp_test::dump(output, phase, i, stage, values);
          }
          std::vector<int32_t> ids(topk);
          if (!runtime->read_ids("router_ids", ids, error))
            throw std::runtime_error(error);
          mtp_test::dump(output, phase, i, "router_ids", ids);
        }
    }
    output.flush();
    if (!output || runtime->build_count() != builds ||
        runtime->replay_count() != 3 * fixture.rows.size() ||
        engine->hc_stream_buf != std::vector<float>({1, 2, 3}) ||
        engine->qwen4exp_recur_state != std::vector<std::vector<float>>{{4, 5}} ||
        engine->qwen4exp_conv_history != std::vector<std::vector<float>>{{6, 7}} ||
        engine->qwen4exp_ple_conv_history != std::vector<std::vector<float>>{{8, 9}} ||
        engine->qwen4exp_ple_tok_hist != std::vector<int32_t>({3, 2, 1}) ||
        engine->qwen4exp_ple_next_pos != 17)
      throw std::runtime_error("draft reuse/output/target-state invariant failed");
    const auto &m = runtime->memory();
    std::fprintf(stderr, "draft graph: builds=%zu replays=%zu weights=%zu "
                         "KV=%zu device_workspace=%zu host_workspace=%zu\n",
                 builds, runtime->replay_count(), m.weights, m.state,
                 m.device_workspace, m.host_workspace);
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "MTP parity: %s\n", e.what());
    return 1;
  }
}

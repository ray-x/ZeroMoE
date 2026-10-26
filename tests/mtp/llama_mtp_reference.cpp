#include "fixture.hpp"
#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"

#include <cstdio>
#include <filesystem>
#include <map>
#include <memory>

namespace {
struct Capture {
  int combines = 0;
  std::map<std::string, std::vector<float>> values;
  std::vector<int32_t> ids;
  std::string error;
};

bool capture(ggml_tensor *tensor, bool ask, void *opaque) {
  auto &state = *static_cast<Capture *>(opaque);
  const std::string name = ggml_get_name(tensor);
  std::string stage;
  if (name.starts_with("mtp_eh_proj-")) stage = "bridge";
  else if (name.starts_with("hc_combine-"))
    stage = state.combines == 0 ? "post_attention" : "post_ffn";
  else if (name.starts_with("ffn_moe_probs-")) stage = "router_probabilities";
  else if (name.starts_with("ffn_moe_weights_norm-") ||
           name.starts_with("ffn_moe_weights_scaled-")) stage = "router_weights";
  else if (name.starts_with("ffn_moe_topk-")) stage = "router_ids";
  if (ask) return !stage.empty();
  if (stage.empty()) return true;
  if (!ggml_is_contiguous(tensor)) {
    state.error = "reference diagnostic is noncontiguous: " + name;
    return false;
  }
  if (stage == "router_ids" && tensor->type == GGML_TYPE_I32) {
    state.ids.resize(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, state.ids.data(), 0, ggml_nbytes(tensor));
  } else if (tensor->type == GGML_TYPE_F32) {
    auto &values = state.values[stage];
    values.resize(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
  } else {
    state.error = "unsupported reference diagnostic type: " + name;
    return false;
  }
  if (name.starts_with("hc_combine-")) ++state.combines;
  return true;
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 6) {
    std::fprintf(stderr, "usage: %s REFERENCE_WITH_MTP.gguf fixture.txt dump.txt "
                         "NEXTN_LAYER cpu|cuda\n", argv[0]);
    return 2;
  }
  try {
    const mtp_test::Fixture fixture(argv[2]);
    if (fixture.rows.size() < 32)
      throw std::runtime_error("reference gate requires at least 32 rows");
    if (std::filesystem::exists(argv[3]))
      throw std::runtime_error("refusing to overwrite existing reference dump");
    const int layer = std::stoi(argv[4]);
    const std::string placement = argv[5];
    if (layer < 0 || (placement != "cpu" && placement != "cuda"))
      throw std::runtime_error("invalid reference layer or placement");
    llama_backend_init();
    ggml_backend_load_all();
    auto mp = llama_model_default_params();
    mp.load_mtp = true;
    mp.n_gpu_layers = 0;
    const std::string pattern = "blk\\." + std::to_string(layer) + "\\..*";
    llama_model_tensor_buft_override overrides[2] = {};
    if (placement == "cuda") {
      auto device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
      if (!device || std::string(ggml_backend_dev_name(device)).find("CUDA") ==
                         std::string::npos)
        throw std::runtime_error("no CUDA backend for matched-placement reference");
      overrides[0] = {pattern.c_str(), ggml_backend_dev_buffer_type(device)};
      mp.tensor_buft_overrides = overrides;
    }
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(argv[1], mp), llama_model_free);
    if (!model || llama_model_n_embd_inp(model.get()) != fixture.D ||
        llama_model_n_embd_out(model.get()) != fixture.W ||
        llama_model_n_layer(model.get()) != layer ||
        llama_model_n_layer_nextn(model.get()) != 1 ||
        llama_vocab_n_tokens(llama_model_get_vocab(model.get())) != fixture.V)
      throw std::runtime_error("reference model/fixture dimensions or NextN mismatch");
    Capture diagnostics;
    auto cp = llama_context_default_params();
    cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    cp.n_ctx = fixture.rows.size();
    cp.n_batch = cp.n_ubatch = 1;
    cp.n_seq_max = 1;
    cp.type_k = cp.type_v = GGML_TYPE_F16;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.cb_eval = capture;
    cp.cb_eval_user_data = &diagnostics;
    std::unique_ptr<llama_context, decltype(&llama_free)> context(
        llama_init_from_model(model.get(), cp), llama_free);
    if (!context) throw std::runtime_error("cannot create reference MTP context");
    llama_set_embeddings_nextn(context.get(), true, false);
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(
        llama_batch_ext_init(context.get()), llama_batch_ext_free);
    if (!batch) throw std::runtime_error("cannot allocate reference batch");
    std::ofstream output(argv[3]);
    if (!output) throw std::runtime_error("cannot open reference dump");
    fixture.dump_header(output);
    for (int phase = 0; phase < 3; ++phase) {
      llama_memory_clear(llama_get_memory(context.get()), true);
      const int chunk = phase == 1 ? 7 : static_cast<int>(fixture.rows.size());
      for (size_t begin = 0; begin < fixture.rows.size(); begin += chunk)
        for (size_t i = begin; i < std::min(begin + chunk, fixture.rows.size()); ++i) {
          diagnostics = {};
          const auto &row = fixture.rows[i];
          llama_batch_ext_clear(batch.get());
          const int idx = llama_batch_ext_add_embd(batch.get(), 0,
              {row.embedding.data(), 1, row.embedding.size()});
          const llama_pos positions[4] = {row.position, row.position,
                                          row.position, row.position};
          if (idx < 0 || !llama_batch_ext_set_pos(batch.get(), idx, positions) ||
              !llama_batch_ext_set_embd_state(batch.get(), idx,
                  {row.hidden.data(), 1, row.hidden.size()}) ||
              !llama_batch_ext_set_output_logits(batch.get(), idx, true) ||
              llama_process(context.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) ||
              !diagnostics.error.empty())
            throw std::runtime_error("reference replay failed: " + diagnostics.error);
          const auto *hidden = llama_get_embeddings_nextn_ith(context.get(), 0);
          const auto *logits = llama_get_logits_ith(context.get(), 0);
          if (!hidden || !logits || diagnostics.combines != 2 ||
              diagnostics.values.size() != 5 || diagnostics.ids.empty())
            throw std::runtime_error("missing reference output/intermediate diagnostics");
          mtp_test::dump(output, phase, i, "hidden",
                         std::vector<float>(hidden, hidden + fixture.W));
          mtp_test::dump(output, phase, i, "logits",
                         std::vector<float>(logits, logits + fixture.V));
          for (const auto &[stage, values] : diagnostics.values)
            mtp_test::dump(output, phase, i, stage, values);
          mtp_test::dump(output, phase, i, "router_ids", diagnostics.ids);
        }
    }
    output.flush();
    if (!output) throw std::runtime_error("reference dump flush failed");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "MTP reference: %s\n", e.what());
    return 1;
  }
}

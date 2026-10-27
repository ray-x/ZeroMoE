#include <algorithm>
#include "mtp_graph_runtime.hpp"

#if defined(ZEROMOE_USE_GGML_CUDA)
#include <climits>
#include <chrono>
#include <limits>

namespace zeromoe::graph {

MtpGraphState::~MtpGraphState() {
  if (sched) {
    ggml_backend_sched_synchronize(sched);
    ggml_backend_sched_free(sched);
    sched = nullptr;
  }
  for (auto buffer : host_buffers)
    if (buffer) ggml_backend_buffer_free(buffer);
  host_buffers.clear();
  dense.free_buffer();
  expert.free_buffer();
  state.free_buffer();
  if (weight_ctx) ggml_free(weight_ctx);
  if (state_ctx) ggml_free(state_ctx);
  auto cuda_backend = cuda;
  auto cpu_backend = cpu;
  release_core();
  if (cuda_backend) ggml_backend_free(cuda_backend);
  if (cpu_backend) ggml_backend_free(cpu_backend);
}

ggml_context *MtpGraphState::make_context(size_t bytes) {
  void *storage = alloc_host(bytes);
  return storage ? ggml_init({bytes, storage, true}) : nullptr;
}

std::unique_ptr<MtpGraphRuntime> MtpGraphRuntime::create(
    MtpGraphShape shape, size_t device_budget, const Builder &build,
    std::string &error) {
  error.clear();
  if (!shape.embedding || !shape.hidden || !shape.vocab ||
      shape.embedding > INT_MAX || shape.hidden > INT_MAX ||
      shape.vocab > INT_MAX || shape.capacity <= 0 || !device_budget) {
    error = "invalid draft graph shape or memory budget";
    return nullptr;
  }
  const auto setup_started = std::chrono::steady_clock::now();
  auto result = std::unique_ptr<MtpGraphRuntime>(new MtpGraphRuntime(shape));
  const auto measure_build_started = std::chrono::steady_clock::now();
  auto measured = build(error);
  const double measure_build_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - measure_build_started).count();
  if (!measured) {
    if (error.empty()) error = "draft measurement graph construction failed";
    return nullptr;
  }
  ++result->build_count_;
  const auto complete = [](const MtpGraphState &g) {
    return g.graph && g.sched && g.cuda && g.cpu && g.embedding && g.hidden &&
           g.final_streams && g.logits && g.prepare_position && g.bindings_valid;
  };
  if (!complete(*measured)) {
    error = "draft graph builder returned incomplete bindings";
    return nullptr;
  }
  size_t workspace[2] = {};
  ggml_backend_sched_reserve_size(measured->sched, measured->graph, workspace);
  result->memory_ = {measured->device_weight_bytes, measured->device_state_bytes,
                     workspace[0], workspace[1]};
  const auto &m = result->memory_;
  if (m.weights > device_budget || m.state > device_budget - m.weights ||
      m.device_workspace > device_budget - m.weights - m.state) {
    error = "draft weights, KV and measured workspace exceed device budget";
    return nullptr;
  }
  // reserve_size mutates scheduler graph splitting. Neither that scheduler
  // nor its graph is reused for execution (see the Qwen target incident).
  measured.reset();
  const auto execution_build_started = std::chrono::steady_clock::now();
  result->state_ = build(error);
  const double execution_build_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - execution_build_started).count();
  if (!result->state_) {
    if (error.empty()) error = "draft execution graph construction failed";
    return nullptr;
  }
  ++result->build_count_;
  auto &g = *result->state_;
  if (!complete(g) || g.device_weight_bytes != m.weights ||
      g.device_state_bytes != m.state || g.kv_capacity < shape.capacity ||
      g.embedding->type != GGML_TYPE_F32 || g.hidden->type != GGML_TYPE_F32 ||
      g.final_streams->type != GGML_TYPE_F32 || g.logits->type != GGML_TYPE_F32 ||
      !ggml_is_contiguous(g.final_streams) || !ggml_is_contiguous(g.logits) ||
      ggml_nelements(g.embedding) != static_cast<int64_t>(shape.embedding) ||
      ggml_nelements(g.hidden) != static_cast<int64_t>(shape.hidden) ||
      ggml_nelements(g.final_streams) != static_cast<int64_t>(shape.hidden) ||
      ggml_nelements(g.logits) != static_cast<int64_t>(shape.vocab)) {
    error = "draft graph shape or placement changed between measure and build";
    return nullptr;
  }
  size_t free_bytes = 0, total_bytes = 0;
  if (cudaSetDevice(g.device) != cudaSuccess ||
      cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
      m.device_workspace > free_bytes) {
    error = "insufficient free device memory for draft workspace";
    return nullptr;
  }
  const size_t scratch_bytes = (shape.hidden + shape.vocab) * sizeof(float);
  if (m.host_workspace > std::numeric_limits<size_t>::max() - scratch_bytes) {
    error = "draft host workspace accounting overflow";
    return nullptr;
  }
  if (!MemoryGuard::try_reserve_host_bytes(m.host_workspace + scratch_bytes)) {
    error = "host budget refused draft scheduler workspace";
    return nullptr;
  }
  result->workspace_reservation_ = m.host_workspace + scratch_bytes;
  const auto scheduler_alloc_started = std::chrono::steady_clock::now();
  if (!ggml_backend_sched_alloc_graph(g.sched, g.graph)) {
    error = "draft scheduler graph allocation failed";
    return nullptr;
  }
  result->memory_.scheduler_alloc_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - scheduler_alloc_started).count();
  const size_t actual_device = ggml_backend_sched_get_buffer_size(g.sched, g.cuda);
  const size_t actual_host = ggml_backend_sched_get_buffer_size(g.sched, g.cpu);
  // reserve_size estimates cross-backend copies slightly low (e.g. when a node
  // is forced onto the other backend). The allocation already succeeded, so
  // reconcile the admitted figures with what is really resident and refuse only
  // if that exceeds the device budget or the host guard.
  if (actual_device > m.device_workspace || actual_host > m.host_workspace) {
    if (actual_device > device_budget - m.weights - m.state) {
      error = "draft scheduler workspace exceeds device budget after allocation";
      return nullptr;
    }
    const size_t extra_host = actual_host > m.host_workspace
        ? actual_host - m.host_workspace : 0;
    if (extra_host && !MemoryGuard::try_reserve_host_bytes(extra_host)) {
      error = "host budget refused draft scheduler workspace growth";
      return nullptr;
    }
    result->workspace_reservation_ += extra_host;
    result->memory_.device_workspace = std::max(actual_device, m.device_workspace);
    result->memory_.host_workspace = std::max(actual_host, m.host_workspace);
  }
  result->hidden_scratch_.resize(shape.hidden);
  result->logits_scratch_.resize(shape.vocab);
  g.ready = true;
  const auto reset_started = std::chrono::steady_clock::now();
  if (!result->reset(error)) return nullptr;
  result->memory_.measure_build_ms = measure_build_ms;
  result->memory_.execution_build_ms = execution_build_ms;
  result->memory_.reset_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - reset_started).count();
  result->memory_.total_setup_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - setup_started).count();
  return result;
}

MtpGraphRuntime::~MtpGraphRuntime() {
  state_.reset();
  std::vector<float>().swap(hidden_scratch_);
  std::vector<float>().swap(logits_scratch_);
  MemoryGuard::release_host_bytes(workspace_reservation_);
}

bool MtpGraphRuntime::replay(const MtpGraphRequest &r, std::string &error) {
  error = cursor_.validate(r);
  if (!error.empty()) return false;
  auto &g = *state_;
  if (!g.bindings_valid()) {
    error = "draft borrowed weight bindings changed; rebuild required";
    cursor_.poison();
    has_output_ = false;
    return false;
  }
  error = g.prepare_position(r.position);
  if (!error.empty()) return false;
  ggml_backend_tensor_set(g.embedding, r.embedding.data(), 0,
                          r.embedding.size_bytes());
  ggml_backend_tensor_set(g.hidden, r.hidden.data(), 0, r.hidden.size_bytes());
  has_output_ = false;
  if (ggml_backend_sched_graph_compute(g.sched, g.graph) != GGML_STATUS_SUCCESS) {
    error = "draft graph execution failed; reset required";
    cursor_.poison();
    return false;
  }
  ggml_backend_sched_synchronize(g.sched);
  ggml_backend_tensor_get(g.final_streams, hidden_scratch_.data(), 0,
                          hidden_scratch_.size() * sizeof(float));
  ggml_backend_tensor_get(g.logits, logits_scratch_.data(), 0,
                          logits_scratch_.size() * sizeof(float));
  const auto finite = [](float v) { return std::isfinite(v); };
  if (!std::all_of(hidden_scratch_.begin(), hidden_scratch_.end(), finite) ||
      !std::all_of(logits_scratch_.begin(), logits_scratch_.end(), finite)) {
    error = "draft graph produced non-finite output; reset required";
    cursor_.poison();
    return false;
  }
  std::copy(hidden_scratch_.begin(), hidden_scratch_.end(), r.next_hidden.begin());
  std::copy(logits_scratch_.begin(), logits_scratch_.end(), r.logits.begin());
  cursor_.commit();
  ++replay_count_;
  has_output_ = true;
  return true;
}

bool MtpGraphRuntime::reset(std::string &error) {
  error.clear();
  auto &g = *state_;
  ggml_backend_sched_synchronize(g.sched);
  if (!g.bindings_valid()) {
    error = "draft bindings changed; reset cannot replace a graph rebuild";
    return false;
  }
  if (g.state.buffer) ggml_backend_buffer_clear(g.state.buffer, 0);
  cursor_.reset();
  has_output_ = false;
  return true;
}

bool MtpGraphRuntime::rewind(int position, std::string &error) {
  error = cursor_.rewind(position);
  if (!error.empty()) return false;
  ggml_backend_sched_synchronize(state_->sched);
  has_output_ = false;
  return true;
}

bool MtpGraphRuntime::read_diagnostic(
    const std::string &name, std::span<float> output, std::string &error) const {
  error.clear();
  auto it = state_->diagnostics.find(name);
  if (!has_output_ || cursor_.poisoned() || it == state_->diagnostics.end()) {
    error = "draft diagnostic unavailable before successful replay or unknown name";
    return false;
  }
  auto *t = it->second;
  if (t->type != GGML_TYPE_F32 ||
      ggml_nelements(t) != static_cast<int64_t>(output.size())) {
    error = "draft diagnostic shape/type mismatch";
    return false;
  }
  ggml_backend_tensor_get(t, output.data(), 0, output.size_bytes());
  return true;
}

bool MtpGraphRuntime::read_ids(
    const std::string &name, std::span<int32_t> output, std::string &error) const {
  error.clear();
  auto it = state_->diagnostics.find(name);
  if (!has_output_ || cursor_.poisoned() || it == state_->diagnostics.end() ||
      it->second->type != GGML_TYPE_I32 ||
      ggml_nelements(it->second) != static_cast<int64_t>(output.size())) {
    error = "draft ID diagnostic unavailable or shape/type mismatch";
    return false;
  }
  ggml_backend_tensor_get(it->second, output.data(), 0, output.size_bytes());
  return true;
}

} // namespace zeromoe::graph
#endif

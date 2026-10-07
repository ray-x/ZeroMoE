// CUDA graph topology contract for the partial-layer tier.
//
// The parent graph classifies route slots on-device, runs resident and staged
// pure-GPU children for mixed route sets, merges their outputs, and then runs
// an unconditional host node outside all conditional bodies.
#include <cuda_runtime.h>

#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstring>

extern "C" void *moestream_qwen4exp_master_route_mode_kernel_ptr();
extern "C" void *moestream_qwen4exp_unpack_staged_payload_kernel_ptr();
extern "C" void *moestream_qwen4exp_zero_moe_outputs_kernel_ptr();
extern "C" void *moestream_qwen4exp_merge_moe_outputs_kernel_ptr();

namespace {

__global__ void write_branch(float *output, float value) { *output = value; }

struct HostContext {
  int *mode = nullptr;
  std::atomic<int> *calls = nullptr;
};

void CUDART_CB parent_host_node(void *opaque) {
  auto *ctx = static_cast<HostContext *>(opaque);
  if (ctx && ctx->mode && ctx->calls)
    ctx->calls->fetch_add(1, std::memory_order_relaxed);
}

void check(cudaError_t status, const char *what) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
    std::abort();
  }
}

}  // namespace

int main() {
  int *ids = nullptr, *mode = nullptr, *staged_dispatch_ids = nullptr;
  float *resident_output = nullptr, *staged_output = nullptr, *output = nullptr;
  float *staged_route_mask = nullptr;
  unsigned char *hot = nullptr, *staged = nullptr;
  unsigned char *payload = nullptr, *gate_dst = nullptr, *up_dst = nullptr,
                *down_dst = nullptr;
  int *host_mode = nullptr;
  check(cudaMalloc(&ids, 2 * sizeof(int)), "cudaMalloc ids");
  check(cudaMalloc(&mode, sizeof(int)), "cudaMalloc mode");
  check(cudaMalloc(&resident_output, sizeof(int)), "cudaMalloc resident output");
  check(cudaMalloc(&staged_output, sizeof(int)), "cudaMalloc staged output");
  check(cudaMalloc(&output, sizeof(int)), "cudaMalloc output");
  check(cudaMalloc(&staged_dispatch_ids, 2 * sizeof(int)),
        "cudaMalloc staged dispatch IDs");
  check(cudaMalloc(&staged_route_mask, 2 * sizeof(float)),
        "cudaMalloc staged route mask");
  check(cudaMalloc(&hot, 8), "cudaMalloc hot");
  check(cudaMalloc(&staged, 8), "cudaMalloc staged");
  check(cudaMalloc(&payload, 18), "cudaMalloc payload");
  check(cudaMalloc(&gate_dst, 6), "cudaMalloc gate destination");
  check(cudaMalloc(&up_dst, 4), "cudaMalloc up destination");
  check(cudaMalloc(&down_dst, 8), "cudaMalloc down destination");
  check(cudaMallocHost(&host_mode, sizeof(int)), "cudaMallocHost mode");
  unsigned char hot_host[8] = {}, staged_host[8] = {};
  hot_host[1] = hot_host[2] = 1;
  staged_host[3] = staged_host[4] = 1;
  check(cudaMemcpy(hot, hot_host, 8, cudaMemcpyHostToDevice), "copy hot");
  check(cudaMemcpy(staged, staged_host, 8, cudaMemcpyHostToDevice),
        "copy staged");
  unsigned char payload_host[18];
  for (int i = 0; i < 18; ++i) payload_host[i] = static_cast<unsigned char>(i + 1);
  check(cudaMemcpy(payload, payload_host, sizeof(payload_host),
                   cudaMemcpyHostToDevice),
        "copy payload");
  check(cudaMemset(gate_dst, 0, 6), "clear gate destination");
  check(cudaMemset(up_dst, 0, 4), "clear up destination");
  check(cudaMemset(down_dst, 0, 8), "clear down destination");

  cudaGraph_t resident_graph = nullptr, staged_graph = nullptr,
              parent_graph = nullptr;
  cudaGraphExec_t parent_exec = nullptr;
  check(cudaGraphCreate(&resident_graph, 0), "resident graph");
  check(cudaGraphCreate(&staged_graph, 0), "staged graph");
  check(cudaGraphCreate(&parent_graph, 0), "parent graph");

  cudaKernelNodeParams resident_params{};
  resident_params.func = reinterpret_cast<void *>(&write_branch);
  float resident_value = 1.0f;
  void *resident_args[] = {&resident_output, &resident_value};
  resident_params.gridDim = dim3(1);
  resident_params.blockDim = dim3(1);
  resident_params.kernelParams = resident_args;
  cudaGraphNode_t resident_kernel = nullptr;
  check(cudaGraphAddKernelNode(&resident_kernel, resident_graph, nullptr, 0,
                               &resident_params),
        "resident child kernel");

  cudaKernelNodeParams staged_params = resident_params;
  float staged_value = 2.0f;
  void *staged_args[] = {&staged_output, &staged_value};
  staged_params.kernelParams = staged_args;
  void *payload_base = payload;
  void *gate_base = gate_dst;
  void *up_base = up_dst;
  void *down_base = down_dst;
  size_t gate_bytes = 3, up_bytes = 2, down_bytes = 4;
  size_t gate_stride = 3, up_stride = 2, down_stride = 4;
  int payload_topk = 2;
  void *unpack_args[] = {&payload_base, &gate_base, &up_base, &down_base,
                         &gate_bytes, &up_bytes, &down_bytes, &gate_stride,
                         &up_stride, &down_stride, &payload_topk};
  cudaKernelNodeParams unpack_params{};
  unpack_params.func = moestream_qwen4exp_unpack_staged_payload_kernel_ptr();
  unpack_params.gridDim = dim3(2);
  unpack_params.blockDim = dim3(32);
  unpack_params.kernelParams = unpack_args;
  cudaGraphNode_t unpack_node = nullptr;
  check(cudaGraphAddKernelNode(&unpack_node, staged_graph, nullptr, 0,
                               &unpack_params),
        "staged payload unpack");
  cudaGraphNode_t staged_kernel = nullptr;
  check(cudaGraphAddKernelNode(&staged_kernel, staged_graph, &unpack_node, 1,
                               &staged_params),
        "staged child kernel");

  cudaGraphNode_t prefix = nullptr;
  check(cudaGraphAddEmptyNode(&prefix, parent_graph, nullptr, 0),
        "parent prefix");
  int branch_count = 1;
  cudaKernelNodeParams zero_params{};
  zero_params.func = moestream_qwen4exp_zero_moe_outputs_kernel_ptr();
  void *zero_args[] = {&resident_output, &staged_output, &branch_count};
  zero_params.gridDim = dim3(1);
  zero_params.blockDim = dim3(32);
  zero_params.kernelParams = zero_args;
  cudaGraphNode_t zero_node = nullptr;
  check(cudaGraphAddKernelNode(&zero_node, parent_graph, &prefix, 1,
                               &zero_params),
        "branch output reset");
  cudaGraphConditionalHandle resident_handle{}, staged_handle{};
  check(cudaGraphConditionalHandleCreate(&resident_handle, parent_graph, 0,
                                         cudaGraphCondAssignDefault),
        "resident handle");
  check(cudaGraphConditionalHandleCreate(&staged_handle, parent_graph, 0,
                                         cudaGraphCondAssignDefault),
        "staged handle");
  cudaKernelNodeParams predicate{};
  predicate.func = moestream_qwen4exp_master_route_mode_kernel_ptr();
  int topk = 2, n_experts = 8;
  void *predicate_args[] = {&resident_handle, &staged_handle, &ids, &hot,
                            &staged, &topk, &n_experts, &mode,
                            &staged_route_mask, &staged_dispatch_ids};
  predicate.gridDim = dim3(1);
  predicate.blockDim = dim3(1);
  predicate.kernelParams = predicate_args;
  cudaGraphNode_t predicate_node = nullptr;
  check(cudaGraphAddKernelNode(&predicate_node, parent_graph, &zero_node, 1,
                               &predicate),
        "route predicate");

  cudaGraphNodeParams resident_cond{};
  resident_cond.type = cudaGraphNodeTypeConditional;
  resident_cond.conditional.handle = resident_handle;
  resident_cond.conditional.type = cudaGraphCondTypeIf;
  resident_cond.conditional.size = 1;
  cudaGraphNode_t resident_if = nullptr;
  check(cudaGraphAddNode(&resident_if, parent_graph, &predicate_node, nullptr,
                         1, &resident_cond),
        "resident conditional");
  cudaGraphNode_t resident_child = nullptr;
  check(cudaGraphAddChildGraphNode(&resident_child,
                                   resident_cond.conditional.phGraph_out[0],
                                   nullptr, 0, resident_graph),
        "resident child");

  cudaGraphNodeParams staged_cond = resident_cond;
  staged_cond.conditional.handle = staged_handle;
  cudaGraphNode_t staged_if = nullptr;
  check(cudaGraphAddNode(&staged_if, parent_graph, &predicate_node, nullptr, 1,
                         &staged_cond),
        "staged conditional");
  cudaGraphNode_t staged_child = nullptr;
  check(cudaGraphAddChildGraphNode(&staged_child,
                                   staged_cond.conditional.phGraph_out[0],
                                   nullptr, 0, staged_graph),
        "staged child");

  cudaGraphNode_t merge_deps[] = {resident_if, staged_if};
  cudaKernelNodeParams merge_params{};
  merge_params.func = moestream_qwen4exp_merge_moe_outputs_kernel_ptr();
  int merge_count = 1;
  void *merge_args[] = {&resident_output, &staged_output, &output,
                        &merge_count};
  merge_params.gridDim = dim3(1);
  merge_params.blockDim = dim3(32);
  merge_params.kernelParams = merge_args;
  cudaGraphNode_t merge_node = nullptr;
  check(cudaGraphAddKernelNode(&merge_node, parent_graph, merge_deps, 2,
                               &merge_params),
        "merge routed outputs");

  std::atomic<int> callback_calls{0};
  HostContext host_context{host_mode, &callback_calls};
  cudaGraphNode_t mode_copy = nullptr;
  check(cudaGraphAddMemcpyNode1D(&mode_copy, parent_graph, &merge_node, 1,
                                 host_mode, mode, sizeof(int),
                                 cudaMemcpyDeviceToHost),
        "mode readback");
  cudaHostNodeParams host_params{};
  host_params.fn = parent_host_node;
  host_params.userData = &host_context;
  cudaGraphNode_t host_node = nullptr;
  check(cudaGraphAddHostNode(&host_node, parent_graph, &mode_copy, 1,
                             &host_params),
        "parent host node");
  check(cudaGraphInstantiate(&parent_exec, parent_graph, 0),
        "parent instantiate");

  cudaStream_t stream = nullptr;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "stream create");
  const int cases[][2] = {{1, 2}, {3, 4}, {1, 3}, {1, 7}};
  const int expected_mode[] = {1, 2, 3, 0};
  const float expected_output[] = {1.0f, 2.0f, 3.0f, 0.0f};
  const float expected_mask[][2] = {{0, 0}, {1, 1}, {0, 1}, {0, 0}};
  const int expected_dispatch[][2] = {{-1, -1}, {0, 1}, {-1, 1}, {-1, -1}};
  for (int i = 0; i < 4; ++i) {
    check(cudaMemcpyAsync(ids, cases[i], 2 * sizeof(int),
                          cudaMemcpyHostToDevice, stream),
          "route upload");
    check(cudaMemsetAsync(output, 0, sizeof(int), stream), "output clear");
    check(cudaGraphLaunch(parent_exec, stream), "parent launch");
    check(cudaStreamSynchronize(stream), "parent sync");
    float output_host = -1.0f;
    check(cudaMemcpy(&output_host, output, sizeof(float),
                     cudaMemcpyDeviceToHost),
          "output readback");
    assert(*host_mode == expected_mode[i]);
    assert(output_host == expected_output[i]);
    float mask_host[2] = {};
    int dispatch_host[2] = {};
    check(cudaMemcpy(mask_host, staged_route_mask, sizeof(mask_host),
                     cudaMemcpyDeviceToHost),
          "read staged mask");
    check(cudaMemcpy(dispatch_host, staged_dispatch_ids,
                     sizeof(dispatch_host), cudaMemcpyDeviceToHost),
          "read staged dispatch IDs");
    assert(mask_host[0] == expected_mask[i][0] &&
           mask_host[1] == expected_mask[i][1]);
    assert(dispatch_host[0] == expected_dispatch[i][0] &&
           dispatch_host[1] == expected_dispatch[i][1]);
    if (expected_mode[i] == 2 || expected_mode[i] == 3) {
      unsigned char gate_host[6] = {}, up_host[4] = {}, down_host[8] = {};
      check(cudaMemcpy(gate_host, gate_dst, sizeof(gate_host),
                       cudaMemcpyDeviceToHost),
            "read gate payload");
      check(cudaMemcpy(up_host, up_dst, sizeof(up_host),
                       cudaMemcpyDeviceToHost),
            "read up payload");
      check(cudaMemcpy(down_host, down_dst, sizeof(down_host),
                       cudaMemcpyDeviceToHost),
            "read down payload");
      const unsigned char expected_gate[] = {1, 2, 3, 10, 11, 12};
      const unsigned char expected_up[] = {4, 5, 13, 14};
      const unsigned char expected_down[] = {6, 7, 8, 9, 15, 16, 17, 18};
      assert(std::memcmp(gate_host, expected_gate, sizeof(gate_host)) == 0);
      assert(std::memcmp(up_host, expected_up, sizeof(up_host)) == 0);
      assert(std::memcmp(down_host, expected_down, sizeof(down_host)) == 0);
    }
  }
  assert(callback_calls.load(std::memory_order_relaxed) == 4);

  cudaStreamDestroy(stream);
  cudaGraphExecDestroy(parent_exec);
  cudaGraphDestroy(parent_graph);
  cudaGraphDestroy(resident_graph);
  cudaGraphDestroy(staged_graph);
  cudaFreeHost(host_mode);
  cudaFree(staged_route_mask);
  cudaFree(staged_dispatch_ids);
  cudaFree(down_dst);
  cudaFree(up_dst);
  cudaFree(gate_dst);
  cudaFree(payload);
  cudaFree(staged);
  cudaFree(hot);
  cudaFree(output);
  cudaFree(mode);
  cudaFree(staged_output);
  cudaFree(resident_output);
  cudaFree(ids);
  std::puts("PASS: resident/staged conditionals with parent host node");
}

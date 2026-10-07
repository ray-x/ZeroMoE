#include "qwen4exp_adaptive_policy.hpp"
#include "qwen4exp_storage_service.hpp"

#include <cassert>
#include <cstdio>

using namespace moestream::qwen4exp;

int main() {
  constexpr size_t kTopK = 2;
  constexpr size_t kSlotBytes = 6;
  StorageService service(1, 1, [](const StorageRequest &request,
                                  StorageCompletion &completion) {
    auto payload = request.payload;
    payload->bytes.resize(kTopK * kSlotBytes, 0xa5);
    payload->offsets = {0, kSlotBytes, kTopK * kSlotBytes};
    completion.experts = request.experts;
    completion.payload = payload;
    return true;
  });

  auto handle = service.submit(StorageRequest{
      7, 11, 13, {4, 4}, std::make_shared<StorageRequest::Payload>()});
  assert(handle);
  auto completion = service.wait_consume(*handle);
  assert(completion && completion->state == StorageState::Ready);
  assert(storage_payload_uniform_slots(*completion, kTopK));

  AdaptiveTiming timing;
  timing.cpu_queue_ms = 1.2;
  timing.cpu_compute_ms = 2.1;
  timing.transfer_queue_ms = 0.1;
  timing.h2d_ms = 0.2;
  timing.gpu_compute_ms = 0.4;
  timing.storage_ms = completion->queue_ms + completion->service_ms;
  timing.cpu_ready = true;
  timing.staged_ready = true;
  timing.storage_required = true;
  const AdaptiveDecision decision = choose_adaptive_target(timing);
  assert(decision.target == AdaptiveTarget::StagedGpu);
  std::puts("PASS: staged storage payload and adaptive boundary");
}

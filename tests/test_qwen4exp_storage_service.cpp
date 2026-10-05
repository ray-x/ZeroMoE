#include "qwen4exp_storage_service.hpp"

#include <cassert>
#include <cstdio>
#include <stdexcept>

using namespace moestream::qwen4exp;

int main() {
  int loads = 0;
  auto payload = std::make_shared<StorageRequest::Payload>();
  payload->bytes.resize(4);
  StorageService service(2, 2, [&](const StorageRequest &request,
                                  StorageCompletion &completion) {
    ++loads;
    if (request.layer == 7) {
      completion.error_code = 404;
      return false;
    }
    completion.experts = request.experts;
    if (request.payload) {
      request.payload->bytes[0] = 0x5a;
      request.payload->offsets = {0, request.payload->bytes.size()};
    }
    return true;
  });

  auto ready = service.submit(StorageRequest{11, 1, 3, {4, 9}, payload});
  auto error = service.submit(StorageRequest{11, 2, 7, {2}});
  assert(ready && error);
  auto ready_result = service.wait(*ready);
  auto error_result = service.wait(*error);
  assert(ready_result && error_result);
  assert(ready_result->state == StorageState::Ready);
  assert(error_result->state == StorageState::Error);
  assert(error_result->error_code == 404);
  assert(ready_result->queue_ms >= 0.0 && ready_result->service_ms >= 0.0);
  assert(error_result->queue_ms >= 0.0 && error_result->service_ms >= 0.0);
  assert(ready_result->model_generation == 11 &&
         ready_result->request_generation == 1);
  assert(ready_result->payload == payload && payload->bytes[0] == 0x5a);
  assert(service.release(*ready));
  assert(service.release(*error));

  auto stale = service.submit(StorageRequest{12, 3, 4, {1}});
  assert(stale);
  auto wrong = *stale;
  wrong.model_generation = 11;
  assert(!service.poll(wrong));
  auto stale_result = service.wait(*stale);
  assert(stale_result && stale_result->model_generation == 12);
  assert(service.release(*stale));
  assert(loads == 3);
  const auto stats = service.stats();
  assert(stats.submitted == 3 && stats.rejected == 0);
  assert(stats.ready == 2 && stats.errors == 1);
  assert(stats.queue_ms_total >= 0.0 && stats.service_ms_total >= 0.0);

  // Loader exceptions are converted into an explicit Error completion rather
  // than leaving a pending slot that a later completion graph could mistake
  // for a ready payload.
  StorageService throwing(1, 1, [](const StorageRequest &, StorageCompletion &) {
    throw std::runtime_error("simulated I/O failure");
    return true;
  });
  auto thrown = throwing.submit(StorageRequest{20, 1, 2, {3}});
  assert(thrown);
  auto thrown_result = throwing.wait(*thrown);
  assert(thrown_result && thrown_result->state == StorageState::Error &&
         thrown_result->error_code == -2);
  assert(throwing.release(*thrown));
  assert(!throwing.submit(StorageRequest{0, 1, 2, {3}}));
  std::puts("PASS: storage service ready/error generations and stale handles");
}

#include "qwen4exp_storage_service.hpp"

#include <cassert>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <thread>

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
  // The staged unpack boundary requires fixed router-order slots. Duplicate
  // route IDs remain valid because exact-once applies to route slots, not
  // unique expert ownership.
  StorageCompletion shaped = *ready_result;
  shaped.experts = {4, 4};
  shaped.payload = std::make_shared<StorageRequest::Payload>();
  shaped.payload->bytes.resize(6);
  shaped.payload->offsets = {0, 3, 6};
  assert(storage_payload_uniform_slots(shaped, 2));
  shaped.payload->offsets = {0, 2, 6};
  assert(!storage_payload_uniform_slots(shaped, 2));
  shaped.payload->offsets = {0, 3};
  assert(!storage_payload_uniform_slots(shaped, 2));
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

  // A full bounded queue rejects demand instead of allocating an unbounded
  // request. Once the first generation is consumed, the same slot can be
  // safely reused by a later generation.
  StorageService bounded(1, 1, [](const StorageRequest &request,
                                  StorageCompletion &completion) {
    completion.experts = request.experts;
    return true;
  });
  auto first = bounded.submit(StorageRequest{50, 1, 1, {8}});
  assert(first);
  auto rejected = bounded.submit(StorageRequest{50, 2, 1, {9}});
  assert(!rejected);
  auto first_completion = bounded.wait_consume(*first);
  assert(first_completion && first_completion->state == StorageState::Ready &&
         first_completion->request_generation == 1);
  auto reused = bounded.submit(StorageRequest{50, 2, 1, {9}});
  assert(reused && reused->slot == first->slot &&
         reused->request_generation == 2);
  auto reused_completion = bounded.wait_consume(*reused);
  assert(reused_completion && reused_completion->state == StorageState::Ready &&
         reused_completion->request_generation == 2);
  const auto bounded_stats = bounded.stats();
  assert(bounded_stats.submitted == 2 && bounded_stats.rejected == 1 &&
         bounded_stats.ready == 2);
  bounded.shutdown();

  // A slow storage generation can be abandoned after a deadline while its
  // worker safely finishes the read. The completed payload is discarded and
  // the bounded slot becomes reusable for the next request generation.
  std::atomic<bool> slow_started{false};
  std::atomic<bool> release_slow{false};
  StorageService timed(1, 1, [&](const StorageRequest &request,
                                StorageCompletion &completion) {
    slow_started.store(true, std::memory_order_release);
    while (!release_slow.load(std::memory_order_acquire))
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    completion.experts = request.experts;
    return true;
  });
  auto slow = timed.submit(StorageRequest{60, 1, 1, {12}});
  assert(slow);
  const auto start_deadline = std::chrono::steady_clock::now() +
                              std::chrono::seconds(2);
  while (!slow_started.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < start_deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  assert(slow_started.load(std::memory_order_acquire));
  bool timed_out = false;
  auto late = timed.wait_consume_for(*slow, std::chrono::milliseconds(2),
                                     &timed_out);
  assert(!late && timed_out);
  release_slow.store(true, std::memory_order_release);
  const auto completion_deadline = std::chrono::steady_clock::now() +
                                   std::chrono::seconds(2);
  while (timed.stats().ready == 0 &&
         std::chrono::steady_clock::now() < completion_deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  assert(timed.stats().ready == 1);
  assert(!timed.wait(*slow)); // retired generations cannot hang a waiter
  auto after_timeout = timed.submit(StorageRequest{60, 2, 1, {13}});
  assert(after_timeout && after_timeout->slot == slow->slot &&
         after_timeout->request_generation == 2);
  auto after_timeout_result = timed.wait_consume_for(
      *after_timeout, std::chrono::seconds(2));
  assert(after_timeout_result &&
         after_timeout_result->state == StorageState::Ready &&
         after_timeout_result->request_generation == 2);
  timed.shutdown();

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

  StorageService atomic_consume(1, 1, [](const StorageRequest &,
                                         StorageCompletion &) { return true; });
  auto consumable = atomic_consume.submit(StorageRequest{30, 1, 0, {0}});
  assert(consumable);
  auto wrong_consumer = *consumable;
  wrong_consumer.request_generation = 99;
  assert(!atomic_consume.wait_consume(wrong_consumer));
  auto consumed = atomic_consume.wait_consume(*consumable);
  assert(consumed && consumed->state == StorageState::Ready &&
         consumed->model_generation == 30 && consumed->request_generation == 1);
  assert(!atomic_consume.poll(*consumable));

  StorageService dispatched(1, 1, [](const StorageRequest &,
                                     StorageCompletion &) { return true; });
  auto dispatch_handle =
      dispatched.submit(StorageRequest{40, 1, 2, {5}});
  assert(dispatch_handle);
  StorageCompletion dispatched_completion;
  bool submit_called = false;
  StorageCompletionDispatcher dispatcher(dispatched);
  assert(dispatcher.wait_and_submit(
      *dispatch_handle,
      [&](const StorageCompletion &completion) {
        submit_called = true;
        return completion.state == StorageState::Ready &&
               completion.model_generation == 40 &&
               completion.request_generation == 1;
      },
      &dispatched_completion));
  assert(submit_called && dispatched_completion.state == StorageState::Ready);
  std::puts("PASS: storage service ready/error generations and stale handles");
}

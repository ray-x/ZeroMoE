#pragma once

// Model-independent ExpertStore -> ColdMoEAdapter bridge. It owns the
// per-lane in-flight demand request and counter accounting; graph builders
// only provide the store and layer identity.

#include "cold_moe_strategy.hpp"
#include "expert_store.hpp"

#include <chrono>
#include <future>
#include <memory>
#include <string>

namespace zeromoe::graph {

struct OffloadResidencyFlight {
  zeromoe::offload::ExpertResidency::Plan plan;
  std::future<zeromoe::offload::ExpertResidency::Ensure> result;
  std::chrono::steady_clock::time_point started;
  int routes = 0;
};

inline void account_offload_ensure(
    zeromoe::offload::ExpertStore &store, int routes,
    const zeromoe::offload::ExpertResidency::Ensure &result,
    std::chrono::steady_clock::time_point started) {
  auto &ct = store.counters();
  ct.calls.fetch_add(1, std::memory_order_relaxed);
  ct.routes.fetch_add((uint64_t)routes, std::memory_order_relaxed);
  ct.cold_routes.fetch_add((uint64_t)routes, std::memory_order_relaxed);
  ct.misses.fetch_add((uint64_t)result.misses, std::memory_order_relaxed);
  ct.loads.fetch_add((uint64_t)result.loads, std::memory_order_relaxed);
  ct.evictions.fetch_add((uint64_t)result.evictions,
                         std::memory_order_relaxed);
  ct.load_us.fetch_add((uint64_t)(result.load_ms * 1000.0),
                       std::memory_order_relaxed);
  ct.cb_us.fetch_add(
      (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - started).count(),
      std::memory_order_relaxed);
  if (result.errors > 0)
    ct.errors.fetch_add((uint64_t)result.errors,
                        std::memory_order_relaxed);
}

inline ColdMoEAdapter make_offload_residency_adapter(
    zeromoe::offload::ExpertStore &store, int layer, bool split_load,
    std::string log_tag = "[MOE-OFFLOAD]") {
  ColdMoEAdapter adapter;
  if (split_load && store.async_loads()) {
    auto flight = std::make_shared<OffloadResidencyFlight>();
    adapter.begin_load = [&store, layer, flight](
        const int32_t *experts, int n, std::vector<int32_t> &pending) {
      flight->started = std::chrono::steady_clock::now();
      flight->routes = n;
      flight->plan = store.begin_ensure(
          layer, experts, n,
          zeromoe::offload::ExpertCache::Kind::Demand);
      pending = flight->plan.to_load;
      pending.insert(pending.end(), flight->plan.to_wait.begin(),
                     flight->plan.to_wait.end());
      const auto policy = pending.empty() && flight->plan.r.errors == 0
                              ? std::launch::deferred
                              : std::launch::async;
      flight->result = std::async(policy, [&store, flight] {
        return store.finish_ensure(flight->plan);
      });
      return true;
    };
    adapter.finish_load = [&store, layer, flight, log_tag]() {
      const auto result = flight->result.get();
      account_offload_ensure(store, flight->routes, result,
                             flight->started);
      if (result.errors > 0) {
        LOG_ERROR("%s layer %d: %d expert load error(s); refusing compute\n",
                  log_tag.c_str(), layer, result.errors);
        store.report("fatal");
        return false;
      }
      return true;
    };
  } else {
    adapter.before_compute = [&store, layer, log_tag](
        const int32_t *experts, int n) {
      const auto started = std::chrono::steady_clock::now();
      const auto result = store.ensure(
          layer, experts, n,
          zeromoe::offload::ExpertCache::Kind::Demand);
      account_offload_ensure(store, n, result, started);
      if (result.errors > 0) {
        LOG_ERROR("%s layer %d: %d expert load error(s); refusing compute\n",
                  log_tag.c_str(), layer, result.errors);
        store.report("fatal");
        return false;
      }
      return true;
    };
  }
  return adapter;
}

} // namespace zeromoe::graph

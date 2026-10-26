#pragma once
#include "server/server.hpp"
#include <memory>

struct CPUEngine;

namespace zeromoe::server {
class CPUBackend final : public Backend {
public:
    CPUBackend(CPUEngine &, size_t cache_bytes, std::function<bool()> cancelled = [] { return false; });
    ~CPUBackend();
    Prepared prepare(const Request &) override;
    Result generate(const Request &, const Prepared &, const std::atomic_bool &,
                    const std::function<bool(const std::string &)> &) override;
    void clear_cache();
    size_t cache_bytes() const;
    size_t cache_entries() const;
    // Also used for teacher-forced numerical comparisons without sampling.
    std::vector<float> evaluate(const std::vector<int> &, bool reuse, int &cached_tokens);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}

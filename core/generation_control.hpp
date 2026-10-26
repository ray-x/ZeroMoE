#pragma once
#include <functional>
#include <string>

struct GenerationControl {
    std::function<bool()> cancelled = [] { return false; };
    std::function<bool(const std::string &)> emit;
    int generated_tokens = 0;
    int cached_tokens = 0;
    bool completed = false;
    bool stopped = false;
    bool interrupted = false;
};

#pragma once
#include "server/types.hpp"

namespace zeromoe::server {
// Keeps partial tags, UTF-8 code points, and tool JSON off the wire.
class OutputParser {
public:
    using Emit = std::function<void(const Block &)>;
    OutputParser(bool thinking, const std::vector<Tool> &tools, Emit emit)
        : thinking_(thinking), tools_(tools), emit_(std::move(emit)) {}
    void feed(const std::string &);
    void finish();
    bool has_tools() const { return has_tools_; }
private:
    void drain(bool final);
    void content(std::string);
    bool thinking_;
    bool in_tool_ = false;
    bool has_tools_ = false;
    std::vector<Tool> tools_;
    Emit emit_;
    std::string buffer_;
};
}

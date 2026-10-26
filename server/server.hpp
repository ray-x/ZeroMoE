#pragma once
#include "server/types.hpp"
#include <memory>

namespace zeromoe::server {
struct Options {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string api_key;
    size_t queue_capacity = 16;
    size_t response_buffer_bytes = 256 * 1024;
    size_t body_limit_bytes = 8 * 1024 * 1024;
    int timeout_seconds = 600;
    int http_threads = 24;
};

class Server {
public:
    Server(Backend &, ModelInfo, Options);
    ~Server();
    Server(const Server &) = delete;
    Server &operator=(const Server &) = delete;
    // Runs inference on this thread; HTTP workers never touch Backend.
    void run(const std::function<bool()> &should_stop = [] { return false; });
    void stop();
    int port() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}

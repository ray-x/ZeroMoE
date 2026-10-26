#include "server/server.hpp"
#include "server/output_parser.hpp"

#include <httplib.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <thread>

namespace zeromoe::server {
namespace {
using Clock = std::chrono::steady_clock;
struct Event {
    enum Kind { Ready, Text, Done, Error } kind = Ready;
    std::string text;
    Result result;
    int status = 500;
};
struct Job {
    Request request;
    bool count_only = false;
    std::atomic_bool cancel{false};
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<Event> events;
    size_t bytes = 0;
    size_t limit;
    Clock::time_point deadline;
    explicit Job(size_t limit, int timeout) : limit(limit), deadline(Clock::now() + std::chrono::seconds(timeout)) {}

    bool push(Event e) {
        std::unique_lock lock(mutex);
        if (Clock::now() >= deadline) cancel = true;
        const size_t size = e.text.size() + 128;
        if (size > limit) throw ApiError(500, "inference event exceeds response buffer");
        while (bytes + size > limit && !cancel) {
            if (cv.wait_until(lock, deadline) == std::cv_status::timeout) cancel = true;
        }
        if (cancel) return false;
        bytes += size;
        events.push_back(std::move(e));
        cv.notify_all();
        return true;
    }
    Event next(const std::function<bool()> &closed) {
        std::unique_lock lock(mutex);
        if (closed() || Clock::now() >= deadline) cancel = true;
        while (events.empty() && !cancel) {
            if (closed() || Clock::now() >= deadline) cancel = true;
            else cv.wait_for(lock, std::chrono::milliseconds(50));
        }
        if (cancel) {
            cv.notify_all();
            throw ApiError(408, "request cancelled or timed out");
        }
        Event e = std::move(events.front());
        events.pop_front();
        bytes -= e.text.size() + 128;
        cv.notify_all();
        return e;
    }
    void stop() {
        std::lock_guard lock(mutex);
        cancel = true;
        cv.notify_all();
    }
};
struct CancelOnExit {
    std::shared_ptr<Job> job;
    ~CancelOnExit() { if (job) job->stop(); }
};
Json usage_json(const Usage &u, Protocol p) {
    if (p == Protocol::Anthropic)
        return {{"input_tokens", u.prompt_tokens - u.cached_tokens}, {"output_tokens", u.output_tokens},
                {"cache_read_input_tokens", u.cached_tokens}, {"cache_creation_input_tokens", 0}};
    return {{"prompt_tokens", u.prompt_tokens}, {"completion_tokens", u.output_tokens},
            {"total_tokens", u.prompt_tokens + u.output_tokens},
            {"prompt_tokens_details", {{"cached_tokens", u.cached_tokens}}}};
}

class Response {
public:
    Response(const Request &r, const ModelInfo &model)
        : request(r), id(new_id(r.protocol == Protocol::Anthropic ? "msg_" : "chatcmpl_")),
          parser(model.starts_thinking, r.tools, [this](const Block &b) { block(b); }) {}
    void start(int prompt_tokens, std::function<void(const std::string &)> write) {
        write_ = std::move(write);
        if (request.protocol == Protocol::Anthropic) {
            Json msg = {{"id", id}, {"type", "message"}, {"role", "assistant"}, {"model", request.model},
                        {"content", Json::array()}, {"stop_reason", nullptr}, {"stop_sequence", nullptr},
                        {"usage", usage_json({prompt_tokens, 0, 0}, request.protocol)}};
            send("message_start", {{"type", "message_start"}, {"message", msg}});
        } else chunk({{"role", "assistant"}, {"content", ""}});
    }
    void feed(const std::string &s) { parser.feed(s); }
    void writer(std::function<void(const std::string &)> write) { write_ = std::move(write); }
    void finish(const Result &r) {
        parser.finish();
        close_block();
        const std::string reason = parser.has_tools() ? (anthropic() ? "tool_use" : "tool_calls") :
                                   r.finish == Finish::Length ? (anthropic() ? "max_tokens" : "length") :
                                   !r.stop_sequence.empty() ? (anthropic() ? "stop_sequence" : "stop") :
                                   (anthropic() ? "end_turn" : "stop");
        const Json stop_sequence = reason == "stop_sequence" ? Json(r.stop_sequence) : Json(nullptr);
        if (write_) {
            if (anthropic()) {
                send("message_delta", {{"type", "message_delta"},
                     {"delta", {{"stop_reason", reason}, {"stop_sequence", stop_sequence}}},
                     {"usage", usage_json(r.usage, request.protocol)}});
                send("message_stop", {{"type", "message_stop"}});
            } else {
                chunk(Json::object(), reason);
                if (request.include_usage) {
                    Json j = envelope("chat.completion.chunk");
                    j["choices"] = Json::array();
                    j["usage"] = usage_json(r.usage, request.protocol);
                    send("", j);
                }
                write_("data: [DONE]\n\n");
            }
        }
        final_ = envelope(anthropic() ? "message" : "chat.completion");
        final_["usage"] = usage_json(r.usage, request.protocol);
        if (anthropic()) {
            final_.erase("object");
            final_.erase("created");
            final_["type"] = "message";
            final_["role"] = "assistant";
            final_["content"] = blocks;
            final_["stop_reason"] = reason;
            final_["stop_sequence"] = stop_sequence;
        } else {
            Json message = {{"role", "assistant"}, {"content", text.empty() && !calls.empty() ? Json(nullptr) : Json(text)}};
            if (!thinking.empty()) message["reasoning_content"] = thinking;
            if (!calls.empty()) message["tool_calls"] = calls;
            final_["choices"] = Json::array({{{"index", 0}, {"message", message}, {"finish_reason", reason}}});
        }
    }
    Json result() const { return final_; }
private:
    Request request;
    std::string id;
    const int64_t created = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    OutputParser parser;
    std::function<void(const std::string &)> write_;
    std::string text, thinking;
    Json blocks = Json::array(), calls = Json::array(), final_;
    int open_index = -1;
    BlockType open_type = BlockType::ToolResult;
    bool anthropic() const { return request.protocol == Protocol::Anthropic; }
    Json envelope(const std::string &object) const {
        return {{"id", id}, {"object", object}, {"model", request.model}, {"created", created}};
    }
    void send(const std::string &event, const Json &j) {
        if (write_) write_((event.empty() ? "" : "event: " + event + "\n") + "data: " + j.dump() + "\n\n");
    }
    void chunk(const Json &delta, const std::string &reason = "") {
        Json j = envelope("chat.completion.chunk");
        j["choices"] = Json::array({{{"index", 0}, {"delta", delta},
                                    {"finish_reason", reason.empty() ? Json(nullptr) : Json(reason)}}});
        send("", j);
    }
    void close_block() {
        if (open_index >= 0) {
            send("content_block_stop", {{"type", "content_block_stop"}, {"index", open_index}});
            open_index = -1;
        }
    }
    void block(const Block &b) {
        if (b.type == BlockType::ToolUse) {
            Json call = {{"id", b.id}, {"type", "function"},
                         {"function", {{"name", b.name}, {"arguments", b.arguments.dump()}}}};
            if (!anthropic()) {
                Json delta = call;
                delta["index"] = calls.size();
                chunk({{"tool_calls", Json::array({delta})}});
            } else {
                close_block();
                const int index = static_cast<int>(blocks.size());
                send("content_block_start", {{"type", "content_block_start"}, {"index", index},
                     {"content_block", {{"type", "tool_use"}, {"id", b.id}, {"name", b.name}, {"input", Json::object()}}}});
                send("content_block_delta", {{"type", "content_block_delta"}, {"index", index},
                     {"delta", {{"type", "input_json_delta"}, {"partial_json", b.arguments.dump()}}}});
                send("content_block_stop", {{"type", "content_block_stop"}, {"index", index}});
                blocks.push_back({{"type", "tool_use"}, {"id", b.id}, {"name", b.name}, {"input", b.arguments}});
            }
            calls.push_back(std::move(call));
            return;
        }
        const bool think = b.type == BlockType::Thinking;
        (think ? thinking : text) += b.text;
        if (!anthropic()) {
            chunk({{think ? "reasoning_content" : "content", b.text}});
            return;
        }
        const char *key = think ? "thinking" : "text";
        if (open_index < 0 || open_type != b.type) {
            close_block();
            open_index = static_cast<int>(blocks.size());
            open_type = b.type;
            blocks.push_back({{"type", think ? "thinking" : "text"}, {key, ""}});
            send("content_block_start", {{"type", "content_block_start"}, {"index", open_index},
                                        {"content_block", blocks.back()}});
        }
        blocks.back()[key] = blocks.back()[key].get<std::string>() + b.text;
        send("content_block_delta", {{"type", "content_block_delta"}, {"index", open_index},
             {"delta", {{"type", think ? "thinking_delta" : "text_delta"}, {key, b.text}}}});
    }
};

// Stop matching works across arbitrary token boundaries.
class StopFilter {
public:
    StopFilter(const std::vector<std::string> &stops, std::function<bool(const std::string &)> emit)
        : stops(stops), emit(std::move(emit)) {}
    bool feed(const std::string &s) {
        pending += s;
        size_t end = std::string::npos;
        for (const auto &stop : stops) end = std::min(end, pending.find(stop));
        if (end != std::string::npos) {
            stopped = true;
            for (const auto &stop : stops)
                if (pending.compare(end, stop.size(), stop) == 0) { matched = stop; break; }
            if (end) emit(pending.substr(0, end));
            pending.clear();
            return false;
        }
        size_t keep = 0;
        for (const auto &stop : stops)
            for (size_t n = 1; n < stop.size() && n <= pending.size(); ++n)
                if (pending.compare(pending.size() - n, n, stop, 0, n) == 0) keep = std::max(keep, n);
        std::string ready = pending.substr(0, pending.size() - keep);
        pending.erase(0, ready.size());
        return ready.empty() || emit(ready);
    }
    void finish() { if (!stopped && !pending.empty()) emit(pending); }
    bool stopped = false;
    std::string matched;
private:
    std::vector<std::string> stops;
    std::function<bool(const std::string &)> emit;
    std::string pending;
};
}

struct Server::Impl {
    Backend &backend;
    ModelInfo model;
    Options options;
    httplib::Server http;
    std::atomic_bool stopping{false}, ready{false};
    std::atomic_int bound_port{0};
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::shared_ptr<Job>> queue;
    std::shared_ptr<Job> active;

    Impl(Backend &backend, ModelInfo model, Options options)
        : backend(backend), model(std::move(model)), options(std::move(options)) {
        if (this->options.host != "127.0.0.1" && this->options.host != "::1" &&
            this->options.host != "localhost" && this->options.api_key.empty())
            throw ApiError(400, "non-loopback binding requires ZEROMOE_API_KEY");
        if (!this->options.queue_capacity || this->options.response_buffer_bytes < 4096 ||
            !this->options.body_limit_bytes || this->options.host.empty() ||
            this->options.timeout_seconds <= 0 || this->options.http_threads < 2 ||
            this->options.port < 0 || this->options.port > 65535)
            throw ApiError(400, "invalid server limits");
        if (this->model.id.empty() || this->model.context_size <= 0 ||
            this->model.default_max_tokens <= 0 ||
            this->model.max_output_tokens < this->model.default_max_tokens)
            throw ApiError(400, "invalid model limits");
        http.new_task_queue = [n = this->options.http_threads, q = this->options.queue_capacity] {
            return new httplib::ThreadPool(n, n, q + n);
        };
        http.set_socket_options([](socket_t socket) {
            httplib::set_socket_opt(socket, SOL_SOCKET, SO_REUSEADDR, 1);
        });
        http.set_payload_max_length(this->options.body_limit_bytes);
        http.set_read_timeout(10, 0);
        http.set_write_timeout(10, 0);
        http.set_keep_alive_max_count(100);
        http.Get("/health", [this](const auto &, auto &res) {
            res.status = ready ? 200 : 503;
            res.set_content(Json({{"status", ready ? "ok" : "unavailable"}}).dump(), "application/json");
        });
        http.Get("/v1/models", [this](const auto &req, auto &res) {
            guarded(req, res, Protocol::OpenAI, [&] {
                res.set_content(Json({{"object", "list"}, {"data", Json::array({
                    {{"id", this->model.id}, {"object", "model"}, {"created", 0}, {"owned_by", "zeromoe"}}
                })}}).dump(), "application/json");
            });
        });
        auto route = [this](const std::string &path, Protocol p, bool count) {
            http.Post(path, [this, p, count](const auto &req, auto &res) {
                guarded(req, res, p, [&] { handle(req, res, p, count); });
            });
        };
        route("/v1/chat/completions", Protocol::OpenAI, false);
        route("/v1/messages", Protocol::Anthropic, false);
        route("/v1/messages/count_tokens", Protocol::Anthropic, true);
        http.set_error_handler([](const auto &req, auto &res) {
            if (!res.body.empty()) return;
            const auto p = req.path.starts_with("/v1/messages") ? Protocol::Anthropic : Protocol::OpenAI;
            res.set_content(error_body(p, res.status, res.status == 413 ? "request body too large" : "HTTP request failed").dump(),
                            "application/json");
        });
    }
    void authenticate(const httplib::Request &req) const {
        if (options.api_key.empty()) return;
        const std::string bearer = req.get_header_value("Authorization");
        const std::string candidate = bearer.starts_with("Bearer ") ? bearer.substr(7) : req.get_header_value("x-api-key");
        size_t difference = candidate.size() ^ options.api_key.size();
        for (size_t i = 0; i < options.api_key.size(); ++i)
            difference |= static_cast<unsigned char>(options.api_key[i]) ^
                          (i < candidate.size() ? static_cast<unsigned char>(candidate[i]) : 0);
        if (difference) throw ApiError(401, "invalid API key");
    }
    void guarded(const httplib::Request &req, httplib::Response &res, Protocol p, const std::function<void()> &fn) {
        res.set_header("X-Request-Id", new_id("req_"));
        try {
            authenticate(req);
            if (!ready) throw ApiError(503, "inference unavailable");
            fn();
        } catch (const ApiError &e) {
            res.status = e.status;
            res.set_content(error_body(p, e.status, e.what()).dump(), "application/json");
        } catch (const Json::exception &e) {
            (void)e;
            std::fprintf(stderr, "[API] invalid JSON request or field type\n");
            res.status = 400;
            res.set_content(error_body(p, 400, "invalid JSON or field type").dump(), "application/json");
        } catch (const std::exception &e) {
            std::fprintf(stderr, "[API] request failed: %s\n", e.what());
            res.status = 500;
            res.set_content(error_body(p, 500, "internal server error").dump(), "application/json");
        }
    }
    void handle(const httplib::Request &req, httplib::Response &res, Protocol p, bool count) {
        auto job = std::make_shared<Job>(options.response_buffer_bytes, options.timeout_seconds);
        job->request = parse_request(Json::parse(req.body), p, model, count);
        job->count_only = count;
        {
            std::lock_guard lock(mutex);
            if (stopping) throw ApiError(503, "server shutting down");
            std::erase_if(queue, [](const auto &j) { return j->cancel.load(); });
            if (queue.size() >= options.queue_capacity) throw ApiError(429, "inference queue is full");
            queue.push_back(job);
            cv.notify_one();
        }
        const auto closed = [&req] { return req.is_connection_closed && req.is_connection_closed(); };
        CancelOnExit guard{job};
        Event first = job->next(closed);
        if (first.kind == Event::Error) throw ApiError(first.status, first.text);
        if (count) {
            res.set_content(Json({{"input_tokens", first.result.usage.prompt_tokens}}).dump(), "application/json");
            return;
        }
        auto response = std::make_shared<Response>(job->request, model);
        if (!job->request.stream) {
            for (;;) {
                Event e = job->next(closed);
                if (e.kind == Event::Error) throw ApiError(e.status, e.text);
                if (e.kind == Event::Done) { response->finish(e.result); break; }
                if (e.kind == Event::Text) response->feed(e.text);
            }
            res.set_content(response->result().dump(), "application/json");
            return;
        }
        res.set_header("Cache-Control", "no-cache");
        res.set_header("X-Accel-Buffering", "no");
        // The provider owns cancellation after this HTTP handler returns.
        res.set_chunked_content_provider("text/event-stream",
            [job, response, p, first, started = false](size_t, httplib::DataSink &sink) mutable {
                auto write = [&](const std::string &s) {
                    if (!sink.write(s.data(), s.size())) {
                        job->stop();
                        throw ApiError(408, "client disconnected");
                    }
                };
                try {
                    response->writer(write);
                    if (!started) {
                        response->start(first.result.usage.prompt_tokens, write);
                        started = true;
                    }
                    Event e = job->next([&] { return !sink.is_writable(); });
                    if (e.kind == Event::Error) throw ApiError(e.status, e.text);
                    if (e.kind == Event::Text) response->feed(e.text);
                    if (e.kind == Event::Done) {
                        response->finish(e.result);
                        sink.done();
                        return false;
                    }
                    return true;
                } catch (const std::exception &e) {
                    job->stop();
                    std::fprintf(stderr, "[API] stream failed: %s\n", e.what());
                    if (sink.is_writable()) {
                        const std::string s = (p == Protocol::Anthropic ? "event: error\n" : "") +
                                             std::string("data: ") + error_body(p, 500, e.what()).dump() + "\n\n";
                        sink.write(s.data(), s.size());
                        sink.done();
                    }
                    return false;
                }
            }, [job](bool) { job->stop(); });
        guard.job.reset();
    }
    void stop() {
        ready = false;
        stopping = true;
        {
            std::lock_guard lock(mutex);
            if (active) active->stop();
            for (const auto &j : queue) j->stop();
        }
        cv.notify_all();
        http.stop();
    }
    void run(const std::function<bool()> &should_stop) {
        errno = 0;
        const int port = options.port == 0 ? http.bind_to_any_port(options.host) :
                         http.bind_to_port(options.host, options.port) ? options.port : -1;
        const int bind_errno = errno;
        if (port < 0)
            throw ApiError(500, "could not bind HTTP listener at " + options.host + ":" +
                                  std::to_string(options.port) +
                                  " (socket error " + std::to_string(bind_errno) + ")");
        bound_port = port;
        ready = true;
        std::atomic_bool listener_failed{false};
        std::thread listener([&] {
            if (!http.listen_after_bind() && !stopping) {
                listener_failed = true;
                std::fprintf(stderr, "[API] HTTP listener failed\n");
            }
            ready = false;
            stopping = true;
            cv.notify_all();
        });
        struct ListenerGuard {
            Impl &owner;
            std::thread &thread;
            ~ListenerGuard() { owner.stop(); if (thread.joinable()) thread.join(); }
        } listener_guard{*this, listener};
        std::string fatal_error;
        std::fprintf(stderr, "[API] listening on %s:%d\n", options.host.c_str(), port);
        while (!stopping && !should_stop()) {
            std::shared_ptr<Job> job;
            {
                std::unique_lock lock(mutex);
                cv.wait_for(lock, std::chrono::milliseconds(50), [&] { return stopping || !queue.empty(); });
                if (queue.empty()) continue;
                job = queue.front();
                queue.pop_front();
                active = job;
            }
            if (!job->cancel) {
                try {
                    Prepared prepared = backend.prepare(job->request);
                    if (prepared.tokens.empty()) throw ApiError(400, "empty tokenized prompt");
                    if (prepared.tokens.size() > static_cast<size_t>(model.context_size) ||
                        (!job->count_only && static_cast<size_t>(job->request.max_tokens) >
                         static_cast<size_t>(model.context_size) - prepared.tokens.size()))
                        throw ApiError(400, "prompt and output exceed model context");
                    Event first;
                    first.result.usage.prompt_tokens = static_cast<int>(prepared.tokens.size());
                    if (job->push(first) && !job->count_only) {
                        StopFilter filter(job->request.stops, [&](const auto &s) {
                            return job->push({Event::Text, s});
                        });
                        Result result = backend.generate(job->request, prepared, job->cancel,
                                                         [&](const auto &s) { return filter.feed(s); });
                        filter.finish();
                        if (filter.stopped) {
                            result.finish = Finish::Stop;
                            result.stop_sequence = filter.matched;
                        }
                        else if (result.finish == Finish::Cancelled)
                            throw ApiError(408, "generation cancelled");
                        Event done{Event::Done};
                        done.result = result;
                        job->push(std::move(done));
                    }
                } catch (const ApiError &e) {
                    std::fprintf(stderr, "[API] inference request failed: %s\n", e.what());
                    Event error{Event::Error, e.what()};
                    error.status = e.status;
                    job->push(std::move(error));
                } catch (const std::exception &e) {
                    std::fprintf(stderr, "[API] fatal inference error: %s\n", e.what());
                    job->push({Event::Error, "inference runtime failed"});
                    fatal_error = e.what();
                    stopping = true;
                }
            }
            {
                std::lock_guard lock(mutex);
                active.reset();
            }
        }
        stop();
        listener.join();
        if (!fatal_error.empty()) throw std::runtime_error(fatal_error);
        if (listener_failed) throw std::runtime_error("HTTP listener failed");
    }
};

Server::Server(Backend &backend, ModelInfo model, Options options)
    : impl_(std::make_unique<Impl>(backend, std::move(model), std::move(options))) {}
Server::~Server() { stop(); }
void Server::run(const std::function<bool()> &stop) { impl_->run(stop); }
void Server::stop() { if (impl_) impl_->stop(); }
int Server::port() const { return impl_->bound_port; }
}

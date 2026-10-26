#include "server/server.hpp"
#include "server/chat_template.hpp"
#include "server/output_parser.hpp"
#include <httplib.h>
#include <chrono>
#include <cstdio>
#include <future>
#include <thread>

using namespace zeromoe::server;
using namespace std::chrono_literals;

namespace {
void check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
template<class Fn> void bad(Fn fn) {
    bool rejected = false;
    try { fn(); } catch (const ApiError &) { rejected = true; }
    catch (const Json::exception &) { rejected = true; }
    check(rejected, "invalid input was accepted");
}
Json request(bool anth = false) {
    Json r = {{"model", "test"}, {"messages", Json::array({{{"role", "user"}, {"content", "hello"}}})},
              {"max_tokens", 20}};
    (void)anth;
    return r;
}
Tool tool() { return {"read_file", "Read a file", {{"type", "object"}}}; }
Json tool_definition(bool anth) {
    if (anth) return {{"name", "read_file"}, {"description", "Read a file"}, {"input_schema", {{"type", "object"}}}};
    return {{"type", "function"}, {"function",
        {{"name", "read_file"}, {"description", "Read a file"}, {"parameters", {{"type", "object"}}}}}};
}
class FakeBackend : public Backend {
public:
    std::atomic_int calls{0};
    std::atomic_bool hold{false};
    std::atomic_bool entered{false};
    std::atomic_bool repeat{false};
    std::atomic_bool saw_cancel{false};
    std::atomic_bool fail{false};
    std::atomic_bool reasoning{false};
    std::thread::id owner;
    Prepared prepare(const Request &r) override {
        if (owner == std::thread::id{}) owner = std::this_thread::get_id();
        check(owner == std::this_thread::get_id(), "backend ownership changed");
        if (r.messages.back().blocks.front().text == "bad") throw ApiError(400, "bad prompt");
        return {std::vector<int>(10, 42)};
    }
    Result generate(const Request &r, const Prepared &, const std::atomic_bool &cancel,
                    const std::function<bool(const std::string &)> &emit) override {
        check(owner == std::this_thread::get_id(), "generation owner changed");
        calls++;
        entered = true;
        if (fail) throw std::runtime_error("synthetic backend failure");
        while (hold && !cancel) std::this_thread::sleep_for(2ms);
        if (cancel) return {{10, 0, 0}, Finish::Cancelled};
        while (repeat) {
            if (cancel || !emit("x")) {
                saw_cancel = true;
                return {{10, 0, 0}, Finish::Cancelled};
            }
            std::this_thread::sleep_for(1ms);
        }
        std::string output = r.tools.empty() ? "Hello \xe4\xb8\x96\xe7\x95\x8c!" :
            "<tool_call>\n{\"name\":\"read_file\",\"arguments\":{\"path\":\"a.cpp\"}}\n</tool_call>";
        if (reasoning) output = "<think>Check carefully.</think>The answer.";
        int tokens = 0;
        for (char c : output) {
            ++tokens;
            if (cancel || !emit(std::string(1, c))) break;
        }
        return {{10, 0, tokens}, Finish::Stop};
    }
};
void parser_tests() {
    ModelInfo model{"test"};
    model.tools = true;
    auto r = parse_request(request(), Protocol::OpenAI, model);
    check(r.messages.size() == 1 && r.max_tokens == 20, "request parsing");
    for (const auto &field : {"logprobs", "response_format", "thinking"}) {
        auto j = request(); j[field] = true;
        bad([&] { parse_request(j, Protocol::OpenAI, model); });
    }
    for (const auto &value : Json::array({-1, 0, 1.2, "20", true})) {
        auto j = request(); j["max_tokens"] = value;
        bad([&] { parse_request(j, Protocol::OpenAI, model); });
    }
    auto j = request();
    j["model"] = "unknown";
    bad([&] { parse_request(j, Protocol::OpenAI, model); });
    j = request(); j["messages"][0]["content"] = Json::array({{{"type", "image_url"}}});
    bad([&] { parse_request(j, Protocol::OpenAI, model); });
    j = request(); j["messages"][0] = {{"role", "tool"}, {"content", "orphan"}, {"tool_call_id", "x"}};
    bad([&] { parse_request(j, Protocol::OpenAI, model); });
    j = request(); j["messages"][0]["content"] = Json::array({
        {{"type", "text"}, {"text", "hello"}, {"cache_control", {{"type", "ephemeral"}}}}});
    check(parse_request(j, Protocol::Anthropic, model).messages.size() == 1, "cache hints");
    j["messages"][0]["content"][0]["cache_control"]["type"] = "persistent";
    bad([&] { parse_request(j, Protocol::Anthropic, model); });
    j = request(); j.erase("max_tokens");
    bad([&] { parse_request(j, Protocol::Anthropic, model); });
    parse_request(j, Protocol::Anthropic, model, true);
    j = request(); j["tools"] = Json::array({tool_definition(false)});
    check(parse_request(j, Protocol::OpenAI, model).tools.size() == 1, "tools parse");
    model.tools = false;
    bad([&] { parse_request(j, Protocol::OpenAI, model); });

    std::vector<Block> blocks;
    OutputParser parser(false, {tool()}, [&](const Block &b) { blocks.push_back(b); });
    const std::string output = "Hi \xe4\xb8\x96\xe7\x95\x8c<think>reason</think><tool_call>"
                               "{\"name\":\"read_file\",\"arguments\":{\"path\":\"x\"}}</tool_call>tail";
    for (char c : output) parser.feed(std::string(1, c));
    parser.finish();
    std::string text, thinking;
    int calls = 0;
    for (const auto &b : blocks) {
        if (b.type == BlockType::Text) text += b.text;
        if (b.type == BlockType::Thinking) thinking += b.text;
        if (b.type == BlockType::ToolUse) {
            ++calls;
            check(b.arguments["path"] == "x" && !b.id.empty(), "tool parser");
        }
    }
    check(text == "Hi \xe4\xb8\x96\xe7\x95\x8ctail" && thinking == "reason" && calls == 1, "fragmented parsing");
    bad([&] { OutputParser p(false, {}, [](auto &) {}); p.feed("<tool_call>{"); p.finish(); });
    bad([&] { OutputParser p(false, {}, [](auto &) {}); p.feed(std::string("\xf0", 1)); p.finish(); });
    bad([&] { OutputParser p(false, {}, [](auto &) {}); p.feed("<tool_call>{\"name\":\"x\",\"arguments\":{}}</tool_call>"); });

    r.tools = {tool()};
    auto segments = render_chat(r, TemplateKind::Qwen, false, "", "max");
    std::string rendered;
    for (const auto &s : segments) rendered += s.text;
    check(rendered.find("<tools>") != std::string::npos && rendered.find("read_file") != std::string::npos,
          "tool template missing definitions");
    bad([&] { render_chat(r, TemplateKind::MiMo, false, "", "max"); });
}

struct Fixture {
    FakeBackend backend;
    Options options;
    ModelInfo model;
    std::unique_ptr<Server> server;
    std::future<void> running;
    Fixture(size_t queue = 16, int timeout = 10) {
        options.port = 0;
        options.api_key = "secret";
        options.queue_capacity = queue;
        options.timeout_seconds = timeout;
        model.id = "test";
        model.tools = true;
        server = std::make_unique<Server>(backend, model, options);
        running = std::async(std::launch::async, [&] { server->run(); });
        const auto deadline = std::chrono::steady_clock::now() + 5s;
        while (!server->port()) {
            if (running.wait_for(0ms) == std::future_status::ready)
                running.get();
            check(std::chrono::steady_clock::now() < deadline, "server startup timed out");
            std::this_thread::sleep_for(2ms);
        }
        for (;;) {
            httplib::Client cli("127.0.0.1", server->port());
            auto health = cli.Get("/health");
            if (health && health->status == 200) break;
            check(std::chrono::steady_clock::now() < deadline, "server not ready");
            std::this_thread::sleep_for(2ms);
        }
    }
    ~Fixture() {
        backend.hold = false;
        server->stop();
        if (running.valid()) running.get();
    }
    httplib::Result post(const std::string &path, Json j, bool auth = true) {
        httplib::Client cli("127.0.0.1", server->port());
        cli.set_read_timeout(5, 0);
        return cli.Post(path, auth ? httplib::Headers{{"Authorization", "Bearer secret"}} : httplib::Headers{},
                        j.dump(), "application/json");
    }
};
void http_tests() {
    Fixture f;
    auto res = f.post("/v1/chat/completions", request(), false);
    check(res && res->status == 401, "authentication");
    res = f.post("/v1/chat/completions", request());
    check(res && res->status == 200, "OpenAI HTTP");
    auto j = Json::parse(res->body);
    check(j["choices"][0]["message"]["content"] == "Hello \xe4\xb8\x96\xe7\x95\x8c!", "OpenAI text");
    check(j["usage"]["prompt_tokens_details"]["cached_tokens"] == 0, "false cached usage");
    res = f.post("/v1/messages", request(true));
    check(res && res->status == 200, "Anthropic HTTP");
    j = Json::parse(res->body);
    check(j["type"] == "message" && j["content"][0]["text"] == "Hello \xe4\xb8\x96\xe7\x95\x8c!", "Anthropic text");
    const auto before = f.backend.calls.load();
    res = f.post("/v1/messages/count_tokens", request(true));
    check(res && res->status == 200 && Json::parse(res->body)["input_tokens"] == 10, "count tokens");
    check(f.backend.calls == before, "count tokens ran generation");
    auto r = request(); r["messages"][0]["content"] = "bad";
    res = f.post("/v1/chat/completions", r);
    check(res && res->status == 400, "prepare error");
    r = request(); r["max_tokens"] = 4096;
    res = f.post("/v1/chat/completions", r);
    check(res && res->status == 400, "context overflow");
    for (bool anth : {false, true}) {
        r = request(anth); r["tools"] = Json::array({tool_definition(anth)});
        const std::string path = anth ? "/v1/messages" : "/v1/chat/completions";
        res = f.post(path, r);
        check(res && res->status == 200, "tool generation HTTP");
        j = Json::parse(res->body);
        check(anth ? j["stop_reason"] == "tool_use" : j["choices"][0]["finish_reason"] == "tool_calls", "tool finish");
        auto &msgs = r["messages"];
        if (anth) {
            msgs.push_back({{"role", "assistant"}, {"content", j["content"]}});
            msgs.push_back({{"role", "user"}, {"content", Json::array({
                {{"type", "tool_result"}, {"tool_use_id", j["content"][0]["id"]}, {"content", "file contents"}}
            })}});
        } else {
            msgs.push_back(j["choices"][0]["message"]);
            msgs.push_back({{"role", "tool"}, {"tool_call_id", j["choices"][0]["message"]["tool_calls"][0]["id"]},
                            {"content", "file contents"}});
        }
        res = f.post(path, r);
        check(res && res->status == 200, "tool result round trip");
        r = request(anth); r["tools"] = Json::array({tool_definition(anth)});
        r["stream"] = true;
        if (!anth) r["stream_options"] = {{"include_usage", true}};
        res = f.post(path, r);
        check(res && res->status == 200 && res->get_header_value("Content-Type") == "text/event-stream",
              "SSE headers");
        check(res->body.find(anth ? "event: message_start" : "\"role\":\"assistant\"") != std::string::npos,
              "SSE start");
        check(res->body.find(anth ? "input_json_delta" : "\"tool_calls\"") != std::string::npos, "SSE tool");
        check(res->body.find(anth ? "event: message_stop" : "data: [DONE]") != std::string::npos, "SSE terminal");
    }
    r = request(); r["stop"] = Json::array({"lo "});
    res = f.post("/v1/chat/completions", r);
    check(res && res->status == 200 &&
          Json::parse(res->body)["choices"][0]["message"]["content"] == "Hel", "fragmented stop");
    r = request(true); r["stop_sequences"] = Json::array({"lo "});
    res = f.post("/v1/messages", r);
    check(res && res->status == 200 && Json::parse(res->body)["stop_reason"] == "stop_sequence" &&
          Json::parse(res->body)["stop_sequence"] == "lo ", "Anthropic stop sequence");
    r = request(); r["stream"] = true;
    res = f.post("/v1/chat/completions", r);
    check(res && res->body.find("data: [DONE]") != std::string::npos, "fragmented UTF-8 SSE");
    f.backend.reasoning = true;
    for (bool anth : {false, true}) {
        r = request(anth);
        const std::string path = anth ? "/v1/messages" : "/v1/chat/completions";
        res = f.post(path, r);
        check(res && res->status == 200, "reasoning response");
        j = Json::parse(res->body);
        check(anth ? j["content"][0]["thinking"] == "Check carefully." :
                     j["choices"][0]["message"]["reasoning_content"] == "Check carefully.", "reasoning content");
        r["messages"].push_back(anth ? Json({{"role", "assistant"}, {"content", j["content"]}}) :
                                      j["choices"][0]["message"]);
        r["messages"].push_back({{"role", "user"}, {"content", "Continue."}});
        res = f.post(path, r);
        check(res && res->status == 200, "reasoning history round trip");
        r["stream"] = true;
        res = f.post(path, r);
        check(res && res->body.find(anth ? "thinking_delta" : "reasoning_content") != std::string::npos,
              "reasoning stream");
    }
    f.backend.reasoning = false;
    r = request(); r["unknown"] = true;
    res = f.post("/v1/messages", r);
    check(res && res->status == 400 && Json::parse(res->body)["type"] == "error", "Anthropic error shape");
    httplib::Client cli("127.0.0.1", f.server->port());
    res = cli.Post("/v1/messages", {{"x-api-key", "secret"}}, "{", "application/json");
    check(res && res->status == 400, "malformed JSON");
    res = cli.Get("/v1/models", httplib::Headers{{"Authorization", "Bearer secret"}});
    check(res && res->status == 200 && Json::parse(res->body)["data"][0]["id"] == "test", "models");
    res = cli.Post("/v1/responses", "{}", "application/json");
    check(res && res->status == 404, "unsupported endpoint");
}
void scheduling_tests() {
    Fixture f(1, 1);
    f.backend.hold = true;
    auto first = std::async(std::launch::async, [&] { return f.post("/v1/chat/completions", request()); });
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (!f.backend.entered) {
        check(std::chrono::steady_clock::now() < deadline, "generation not entered");
        std::this_thread::sleep_for(2ms);
    }
    auto second = std::async(std::launch::async, [&] { return f.post("/v1/chat/completions", request()); });
    std::this_thread::sleep_for(50ms);
    auto third = f.post("/v1/chat/completions", request());
    check(third && third->status == 429, "queue admission");
    auto a = first.get();
    check(a && a->status == 408, "active timeout cancellation");
    auto b = second.get();
    check(b && b->status == 408, "queued timeout cancellation");
    f.backend.hold = false;
    auto next = f.post("/v1/chat/completions", request());
    check(next && next->status == 200, "recovery after cancellation");
    Options o; o.host = "0.0.0.0";
    bad([&] { Server server(f.backend, f.model, o); });
    o = f.options; o.port = f.server->port();
    bad([&] { Server server(f.backend, f.model, o); server.run(); });
}
template<class Predicate> void wait_for(Predicate done, const char *message) {
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (!done()) {
        check(std::chrono::steady_clock::now() < deadline, message);
        std::this_thread::sleep_for(2ms);
    }
}
void disconnect_test() {
    Fixture f;
    f.backend.repeat = true;
    httplib::Client cli("127.0.0.1", f.server->port());
    cli.set_read_timeout(3, 0);
    auto r = request(); r["stream"] = true;
    bool received = false;
    auto res = cli.Post("/v1/chat/completions", {{"x-api-key", "secret"}}, r.dump(), "application/json",
                        [&](const char *, size_t n) { received = n > 0; return false; });
    check(received && !res, "stream client did not disconnect");
    wait_for([&] { return f.backend.saw_cancel.load(); }, "disconnect did not cancel inference");
    f.backend.repeat = false;
    res = f.post("/v1/chat/completions", request());
    check(res && res->status == 200, "request after disconnect failed");
}
void streaming_timeout_test() {
    Fixture f(16, 1);
    f.backend.repeat = true;
    auto r = request(); r["stream"] = true;
    auto res = f.post("/v1/chat/completions", r);
    check(res && res->status == 200 && res->body.find("\"error\"") != std::string::npos &&
          res->body.find("[DONE]") == std::string::npos, "stream timeout returned successful completion");
    wait_for([&] { return f.backend.saw_cancel.load(); }, "stream timeout did not cancel backend");
    f.backend.repeat = false;
    res = f.post("/v1/chat/completions", request());
    check(res && res->status == 200, "request after stream timeout failed");
}
void lifecycle_tests() {
    {
        Fixture f;
        f.backend.hold = true;
        auto active = std::async(std::launch::async, [&] { return f.post("/v1/messages", request(true)); });
        wait_for([&] { return f.backend.entered.load(); }, "shutdown request not started");
        auto queued = std::async(std::launch::async, [&] { return f.post("/v1/messages", request(true)); });
        std::this_thread::sleep_for(30ms);
        f.server->stop();
        check(f.running.wait_for(2s) == std::future_status::ready, "shutdown did not finish");
        f.running.get();
        for (auto *pending : {&active, &queued}) {
            check(pending->wait_for(2s) == std::future_status::ready, "shutdown left a blocked request");
            const auto res = pending->get();
            check(!res || res->status >= 400, "shutdown returned a successful generation");
        }
    }
    {
        Fixture f;
        f.backend.fail = true;
        auto res = f.post("/v1/chat/completions", request());
        check(!res || res->status >= 400, "backend failure returned success");
        check(f.running.wait_for(2s) == std::future_status::ready, "backend failure did not stop server");
        bool failed = false;
        try { f.running.get(); }
        catch (const std::runtime_error &e) { failed = std::string(e.what()) == "synthetic backend failure"; }
        check(failed, "fatal backend error did not propagate");
    }
}
}
int main() {
    try {
        parser_tests();
        http_tests();
        scheduling_tests();
        disconnect_test();
        streaming_timeout_test();
        lifecycle_tests();
        std::puts("server tests passed");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}

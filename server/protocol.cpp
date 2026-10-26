#include "server/types.hpp"

#include <chrono>
#include <cmath>
#include <set>

namespace zeromoe::server {
namespace {
void require(bool ok, const std::string &message) {
    if (!ok) throw ApiError(400, message);
}
std::string str(const Json &j, const char *key) {
    require(j.contains(key) && j[key].is_string(), std::string(key) + " must be a string");
    return j[key].get<std::string>();
}
void keys(const Json &j, const std::set<std::string> &allowed) {
    require(j.is_object(), "expected an object");
    for (const auto &[key, value] : j.items()) {
        (void)value;
        require(allowed.contains(key), "unsupported field: " + key);
    }
}
int positive_int(const Json &j) {
    require(j.is_number_integer(), "token limit must be an integer");
    const auto n = j.get<int64_t>();
    require(n > 0 && n <= 1000000, "token limit must be in [1, 1000000]");
    return static_cast<int>(n);
}
float number(const Json &j, float lo, float hi, const char *name) {
    require(j.is_number(), std::string(name) + " must be a number");
    const double n = j.get<double>();
    require(std::isfinite(n) && n >= lo && n <= hi, std::string(name) + " out of range");
    return static_cast<float>(n);
}
void cache_hint(const Json &j) {
    if (!j.contains("cache_control")) return;
    const auto &c = j["cache_control"];
    keys(c, {"type", "ttl"});
    require(c.value("type", "") == "ephemeral", "unsupported cache_control type");
    if (c.contains("ttl"))
        require(c["ttl"] == "5m" || c["ttl"] == "1h", "unsupported cache_control ttl");
    // A hint is accepted, but never implies inference state was actually reused.
}
std::vector<Block> text_blocks(const Json &value, bool anthropic, bool allow_tools) {
    if (value.is_string()) return {{BlockType::Text, value.get<std::string>()}};
    require(value.is_array(), "content must be a string or array");
    std::vector<Block> blocks;
    for (const auto &v : value) {
        const std::string type = str(v, "type");
        cache_hint(v);
        if (type == "text") {
            keys(v, {"type", "text", "cache_control"});
            blocks.push_back({BlockType::Text, str(v, "text")});
        } else if (anthropic && allow_tools && type == "tool_use") {
            keys(v, {"type", "id", "name", "input", "cache_control"});
            require(v.contains("input") && v["input"].is_object(), "tool input must be an object");
            Block b{BlockType::ToolUse};
            b.id = str(v, "id");
            b.name = str(v, "name");
            b.arguments = v["input"];
            blocks.push_back(std::move(b));
        } else if (anthropic && allow_tools && type == "tool_result") {
            keys(v, {"type", "tool_use_id", "content", "is_error", "cache_control"});
            Block b{BlockType::ToolResult};
            b.id = str(v, "tool_use_id");
            require(!v.contains("is_error") || v["is_error"].is_boolean(), "is_error must be boolean");
            b.is_error = v.value("is_error", false);
            require(v.contains("content"), "tool_result content is required");
            for (const auto &part : text_blocks(v["content"], true, false)) b.text += part.text;
            blocks.push_back(std::move(b));
        } else if (anthropic && allow_tools && type == "thinking") {
            keys(v, {"type", "thinking", "signature"});
            if (v.contains("signature")) str(v, "signature");
            blocks.push_back({BlockType::Thinking, str(v, "thinking")});
        } else {
            throw ApiError(400, "unsupported content block: " + type);
        }
    }
    return blocks;
}
}

Request parse_request(const Json &j, Protocol protocol, const ModelInfo &model, bool count_only) {
    Request r;
    r.protocol = protocol;
    r.max_tokens = model.default_max_tokens;
    r.temperature = model.temperature;
    r.top_p = model.top_p;
    const bool anth = protocol == Protocol::Anthropic;
    if (anth)
        keys(j, {"model", "messages", "system", "max_tokens", "temperature", "top_p",
                 "stop_sequences", "stream", "tools", "tool_choice", "metadata"});
    else
        keys(j, {"model", "messages", "max_tokens", "max_completion_tokens", "temperature", "top_p",
                 "stop", "stream", "stream_options", "tools", "tool_choice", "n", "parallel_tool_calls"});
    r.model = str(j, "model");
    if (r.model != model.id) throw ApiError(404, "model not found: " + r.model);
    if (j.contains("stream")) {
        require(j["stream"].is_boolean(), "stream must be boolean");
        r.stream = j["stream"].get<bool>();
    }
    if (j.contains("stream_options")) {
        keys(j["stream_options"], {"include_usage"});
        require(!j["stream_options"].contains("include_usage") ||
                j["stream_options"]["include_usage"].is_boolean(), "include_usage must be boolean");
        r.include_usage = j["stream_options"].value("include_usage", false);
    }
    require(!j.contains("n") || j["n"] == 1, "only n=1 is supported");
    require(!j.contains("parallel_tool_calls") || j["parallel_tool_calls"] == true,
            "disabling parallel tool calls is not supported");
    require(!j.contains("max_tokens") || !j.contains("max_completion_tokens"),
            "specify only one token limit");
    if (anth && !count_only) require(j.contains("max_tokens"), "max_tokens is required");
    if (j.contains("max_tokens")) r.max_tokens = positive_int(j["max_tokens"]);
    if (j.contains("max_completion_tokens")) r.max_tokens = positive_int(j["max_completion_tokens"]);
    if (!count_only)
        require(r.max_tokens <= model.max_output_tokens, "output token limit exceeds server maximum");
    if (j.contains("temperature")) r.temperature = number(j["temperature"], 0, anth ? 1 : 2, "temperature");
    if (j.contains("top_p")) r.top_p = number(j["top_p"], 0.000001f, 1, "top_p");
    const char *stop_key = anth ? "stop_sequences" : "stop";
    if (j.contains(stop_key) && !j[stop_key].is_null()) {
        const auto &s = j[stop_key];
        if (!anth && s.is_string()) r.stops.push_back(s.get<std::string>());
        else {
            require(s.is_array() && s.size() <= 16, "stop sequences must be an array of at most 16 strings");
            for (const auto &v : s) {
                require(v.is_string(), "stop sequence must be a string");
                r.stops.push_back(v.get<std::string>());
            }
        }
        for (const auto &s : r.stops) require(!s.empty() && s.size() <= 1024, "invalid stop sequence");
    }
    if (j.contains("tool_choice")) {
        const auto &c = j["tool_choice"];
        require(anth ? (c.is_object() && c.size() == 1 && c.value("type", "") == "auto")
                     : c == "auto", "only automatic tool choice is supported");
    }
    if (j.contains("metadata")) {
        keys(j["metadata"], {"user_id"});
        if (j["metadata"].contains("user_id")) str(j["metadata"], "user_id");
    }
    if (j.contains("tools")) {
        require(j["tools"].is_array() && j["tools"].size() <= 128, "invalid tools array");
        std::set<std::string> names;
        for (const auto &t : j["tools"]) {
            if (anth) {
                keys(t, {"name", "description", "input_schema", "cache_control"});
                cache_hint(t);
            } else {
                keys(t, {"type", "function"});
                require(t.value("type", "") == "function", "only function tools are supported");
                require(t.contains("function"), "tool function is required");
                keys(t["function"], {"name", "description", "parameters", "strict"});
                require(!t["function"].contains("strict") || t["function"]["strict"] == false,
                        "strict tool schemas are not supported");
            }
            const auto &f = anth ? t : t["function"];
            Tool tool;
            tool.name = str(f, "name");
            require(!tool.name.empty() && names.insert(tool.name).second, "tool names must be nonempty and unique");
            if (f.contains("description")) tool.description = str(f, "description");
            const char *schema = anth ? "input_schema" : "parameters";
            require(f.contains(schema) && f[schema].is_object(), "tool schema must be an object");
            tool.schema = f[schema];
            require(tool.schema.value("type", "") == "object", "tool schema must have type object");
            r.tools.push_back(std::move(tool));
        }
    }
    if (anth && j.contains("system"))
        r.messages.push_back({"system", text_blocks(j["system"], true, false)});
    require(j.contains("messages") && j["messages"].is_array() && !j["messages"].empty(),
            "messages must be a nonempty array");
    std::set<std::string> pending;
    std::set<std::string> seen_calls;
    for (const auto &m : j["messages"]) {
        keys(m, anth ? std::set<std::string>{"role", "content"}
                     : std::set<std::string>{"role", "content", "tool_calls", "tool_call_id", "reasoning_content"});
        Message msg;
        msg.role = str(m, "role");
        require(anth ? (msg.role == "user" || msg.role == "assistant")
                     : (msg.role == "system" || msg.role == "user" || msg.role == "assistant" || msg.role == "tool"),
                "unsupported message role");
        if (m.contains("content") && !m["content"].is_null())
            msg.blocks = text_blocks(m["content"], anth, true);
        if (!anth && m.contains("reasoning_content")) {
            require(msg.role == "assistant", "reasoning_content is only valid for assistant messages");
            msg.blocks.insert(msg.blocks.begin(), {BlockType::Thinking, str(m, "reasoning_content")});
        }
        if (!anth && msg.role == "tool") {
            Block b{BlockType::ToolResult};
            b.id = str(m, "tool_call_id");
            for (const auto &part : msg.blocks) b.text += part.text;
            msg.blocks = {std::move(b)};
        }
        if (!anth && m.contains("tool_calls")) {
            require(msg.role == "assistant" && m["tool_calls"].is_array(), "invalid tool_calls");
            for (const auto &call : m["tool_calls"]) {
                keys(call, {"id", "type", "function"});
                require(call.value("type", "") == "function", "unsupported tool call type");
                const auto &f = call.at("function");
                keys(f, {"name", "arguments"});
                Block b{BlockType::ToolUse};
                b.id = str(call, "id");
                b.name = str(f, "name");
                b.arguments = Json::parse(str(f, "arguments"));
                require(b.arguments.is_object(), "tool arguments must be a JSON object");
                msg.blocks.push_back(std::move(b));
            }
        }
        for (const auto &b : msg.blocks) {
            require(b.type != BlockType::Thinking || msg.role == "assistant",
                    "thinking blocks are only valid for assistant messages");
            if (b.type == BlockType::ToolUse) {
                require(msg.role == "assistant" && !b.id.empty() && !b.name.empty() &&
                        seen_calls.insert(b.id).second, "invalid or duplicate tool call");
                pending.insert(b.id);
            }
            if (b.type == BlockType::ToolResult) {
                require((msg.role == "user" || msg.role == "tool") && pending.erase(b.id) == 1,
                        "tool result must match a pending tool call");
            }
        }
        require(!msg.blocks.empty(), "message content is required");
        r.messages.push_back(std::move(msg));
    }
    require(pending.empty(), "tool calls must have results before requesting generation");
    bool has_tool_history = false;
    for (const auto &m : r.messages)
        for (const auto &b : m.blocks)
            has_tool_history |= b.type == BlockType::ToolUse || b.type == BlockType::ToolResult;
    require(model.tools || (r.tools.empty() && !has_tool_history), "loaded model has no supported tool template");
    require(r.messages.back().role == "user" || r.messages.back().role == "tool",
            "the last message must be a user message or tool result");
    return r;
}

Json error_body(Protocol protocol, int status, const std::string &message) {
    std::string type = status == 401 ? "authentication_error" :
                       status == 404 ? "not_found_error" :
                       status == 429 ? "rate_limit_error" :
                       status == 503 ? "overloaded_error" :
                       status < 500 ? "invalid_request_error" : "api_error";
    if (protocol == Protocol::Anthropic)
        return {{"type", "error"}, {"error", {{"type", type}, {"message", message}}}};
    return {{"error", {{"message", message}, {"type", type}, {"param", nullptr}, {"code", nullptr}}}};
}

std::string new_id(const std::string &prefix) {
    static std::atomic<uint64_t> counter{0};
    const auto now = std::chrono::system_clock::now().time_since_epoch().count();
    return prefix + std::to_string(now) + "_" + std::to_string(counter.fetch_add(1));
}
}

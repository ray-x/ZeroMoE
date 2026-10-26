#include "server/output_parser.hpp"
#include <algorithm>

namespace zeromoe::server {
namespace {
size_t utf8_prefix(const std::string &s, size_t end) {
    size_t i = 0;
    while (i < end) {
        const auto c = static_cast<unsigned char>(s[i]);
        size_t n = c < 0x80 ? 1 : c >= 0xc2 && c <= 0xdf ? 2 :
                   c >= 0xe0 && c <= 0xef ? 3 : c >= 0xf0 && c <= 0xf4 ? 4 : 0;
        if (!n) throw ApiError(500, "model emitted invalid UTF-8");
        if (i + n > end) break;
        for (size_t j = 1; j < n; ++j)
            if ((static_cast<unsigned char>(s[i+j]) & 0xc0) != 0x80)
                throw ApiError(500, "model emitted invalid UTF-8");
        // JSON's UTF-8 validator also rejects overlong encodings, surrogates, and >U+10FFFF.
        try { Json(s.substr(i, n)).dump(); }
        catch (const Json::exception &) { throw ApiError(500, "model emitted invalid UTF-8"); }
        i += n;
    }
    return i;
}
}
void OutputParser::content(std::string s) {
    if (!s.empty()) emit_({thinking_ ? BlockType::Thinking : BlockType::Text, std::move(s)});
}
void OutputParser::feed(const std::string &s) {
    buffer_ += s;
    if (buffer_.size() > 1024 * 1024) throw ApiError(500, "model output block exceeds 1 MiB");
    drain(false);
}
void OutputParser::finish() { drain(true); }
void OutputParser::drain(bool final) {
    const std::vector<std::string> tags{"<think>", "</think>", "<tool_call>"};
    while (!buffer_.empty()) {
        if (in_tool_) {
            const auto end = buffer_.find("</tool_call>");
            if (end == std::string::npos) {
                if (final) throw ApiError(500, "model emitted an incomplete tool call");
                return;
            }
            Json j;
            try { j = Json::parse(buffer_.substr(0, end)); }
            catch (const Json::exception &) { throw ApiError(500, "model emitted invalid tool JSON"); }
            if (!j.is_object() || !j.contains("name") || !j["name"].is_string() ||
                !j.contains("arguments") || !j["arguments"].is_object())
                throw ApiError(500, "model emitted an invalid tool call");
            Block b{BlockType::ToolUse};
            b.name = j["name"].get<std::string>();
            if (std::none_of(tools_.begin(), tools_.end(), [&](const Tool &t) { return t.name == b.name; }))
                throw ApiError(500, "model called an undeclared tool");
            b.arguments = j["arguments"];
            b.id = new_id("call_");
            emit_(b);
            has_tools_ = true;
            in_tool_ = false;
            buffer_.erase(0, end + 12);
            continue;
        }
        size_t pos = std::string::npos;
        std::string tag;
        for (const auto &t : tags) {
            const auto p = buffer_.find(t);
            if (p < pos) { pos = p; tag = t; }
        }
        if (pos != std::string::npos) {
            if (utf8_prefix(buffer_, pos) != pos) throw ApiError(500, "invalid UTF-8 before model tag");
            content(buffer_.substr(0, pos));
            buffer_.erase(0, pos + tag.size());
            if (tag == "<tool_call>") {
                if (thinking_) throw ApiError(500, "tool call inside reasoning");
                in_tool_ = true;
            } else thinking_ = tag == "<think>";
            continue;
        }
        size_t keep = 0;
        if (!final)
            for (const auto &t : tags)
                for (size_t n = 1; n < t.size() && n <= buffer_.size(); ++n)
                    if (buffer_.compare(buffer_.size() - n, n, t, 0, n) == 0) keep = std::max(keep, n);
        const size_t n = utf8_prefix(buffer_, buffer_.size() - keep);
        if (final && n != buffer_.size()) throw ApiError(500, "model emitted incomplete UTF-8");
        content(buffer_.substr(0, n));
        buffer_.erase(0, n);
        return;
    }
}
}

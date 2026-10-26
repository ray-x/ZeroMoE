#include "server/chat_template.hpp"

namespace zeromoe::server {
std::vector<Segment> render_chat(const Request &request, TemplateKind kind, bool thinking,
                                 const std::string &default_system, const std::string &effort) {
    std::vector<Segment> out;
    auto text = [&](std::string s) { out.push_back({Marker::Text, std::move(s)}); };
    auto mark = [&](Marker m) { out.push_back({m, {}}); };
    std::vector<Message> messages = request.messages;
    if (kind == TemplateKind::Bailing && (messages.empty() || messages.front().role != "system"))
        messages.insert(messages.begin(), {"system", {}});
    if (!default_system.empty()) {
        if (!messages.empty() && messages.front().role == "system")
            messages.front().blocks.insert(messages.front().blocks.begin(), {BlockType::Text, default_system + "\n\n"});
        else messages.insert(messages.begin(), {"system", {{BlockType::Text, default_system}}});
    }
    if (!request.tools.empty() && kind != TemplateKind::Qwen)
        throw ApiError(400, "tool rendering is only supported for Qwen ChatML");
    if (kind == TemplateKind::Qwen) {
        std::string instructions;
        if (thinking)
            instructions = "Reasoning effort is set to xhigh. Please think carefully through "
                           "the task, validate key assumptions, consider plausible alternatives, "
                           "and prioritize correctness, consistency, and clarity in the final answer.";
        if (!request.tools.empty()) {
            if (!instructions.empty()) instructions += "\n\n";
            instructions += "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>";
            for (const auto &tool : request.tools)
                instructions += "\n" + Json({{"type", "function"}, {"function",
                    {{"name", tool.name}, {"description", tool.description}, {"parameters", tool.schema}}}}).dump();
            instructions += "\n</tools>\n\nFor each function call, return a json object with function name "
                            "and arguments within <tool_call></tool_call> XML tags:\n"
                            "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";
        }
        if (!instructions.empty()) {
            if (!messages.empty() && messages.front().role == "system")
                messages.front().blocks.insert(messages.front().blocks.begin(), {BlockType::Text, instructions + "\n\n"});
            else messages.insert(messages.begin(), {"system", {{BlockType::Text, instructions}}});
        }
    }
    if (kind == TemplateKind::Glm || kind == TemplateKind::GlmFlash) {
        mark(Marker::Gmask);
        mark(Marker::Sop);
        if (kind == TemplateKind::GlmFlash && thinking)
            text("<|system|>Reasoning Effort: " + std::string(effort == "low" ? "Low" : effort == "high" ? "High" : "Max"));
    }
    if (kind == TemplateKind::DeepSeek) mark(Marker::Bos);
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto &m = messages[i];
        bool tool_result = !m.blocks.empty() && m.blocks.front().type == BlockType::ToolResult;
        bool previous_tool = i > 0 && !messages[i - 1].blocks.empty() &&
                             messages[i - 1].blocks.front().type == BlockType::ToolResult;
        bool next_tool = i + 1 < messages.size() && !messages[i + 1].blocks.empty() &&
                         messages[i + 1].blocks.front().type == BlockType::ToolResult;
        if (kind == TemplateKind::Qwen || kind == TemplateKind::MiMo) {
            if (!(tool_result && previous_tool)) {
                mark(Marker::ImStart);
                text((tool_result ? "user" : m.role) + "\n");
            }
        } else if (kind == TemplateKind::Glm || kind == TemplateKind::GlmFlash) {
            if (m.role == "system") text("<|system|>");
            else mark(m.role == "user" ? Marker::User : Marker::Assistant);
            if (kind == TemplateKind::Glm) text("\n");
        } else if (kind == TemplateKind::Bailing) {
            text("<role>" + std::string(m.role == "system" ? "SYSTEM" : m.role == "user" ? "HUMAN" : "ASSISTANT") + "</role>");
        } else if (m.role != "system") {
            mark(m.role == "user" ? Marker::User : Marker::Assistant);
        }
        for (const auto &b : m.blocks) {
            if (b.type == BlockType::Text) text(b.text);
            else if (b.type == BlockType::Thinking) {
                // Prior reasoning is not part of Qwen's next-turn assistant content.
                if (kind != TemplateKind::Qwen) text("<think>" + b.text + "</think>");
            } else if (b.type == BlockType::ToolUse) {
                if (kind != TemplateKind::Qwen) throw ApiError(400, "unsupported tool history");
                text("\n<tool_call>\n" + Json({{"name", b.name}, {"arguments", b.arguments}}).dump() + "\n</tool_call>");
            } else {
                if (kind != TemplateKind::Qwen) throw ApiError(400, "unsupported tool result");
                text("\n<tool_response>\n" + b.text + "\n</tool_response>");
            }
            if (kind == TemplateKind::Bailing && m.role == "system")
                text(thinking ? "\ndetailed thinking on" : "\ndetailed thinking off");
        }
        if (kind == TemplateKind::Qwen || kind == TemplateKind::MiMo) {
            if (!(tool_result && next_tool)) {
                mark(Marker::ImEnd);
                if (kind == TemplateKind::Qwen) text("\n");
            }
        } else if (kind == TemplateKind::Bailing) mark(Marker::RoleEnd);
        else if (kind == TemplateKind::DeepSeek && m.role == "assistant") mark(Marker::Eos);
    }
    if (kind == TemplateKind::Qwen || kind == TemplateKind::MiMo) {
        mark(Marker::ImStart);
        text("assistant\n");
        if (kind == TemplateKind::Qwen) {
            mark(Marker::ThinkStart);
            text(thinking ? "\n" : "\n\n");
            if (!thinking) { mark(Marker::ThinkEnd); text("\n\n"); }
        } else if (!thinking) { mark(Marker::ThinkStart); mark(Marker::ThinkEnd); }
    } else if (kind == TemplateKind::Bailing) {
        text("<role>ASSISTANT</role>\n");
        mark(Marker::ThinkStart);
        if (!thinking) mark(Marker::ThinkEnd);
    } else {
        mark(Marker::Assistant);
        if (kind == TemplateKind::Glm) text("\n");
        else if (kind == TemplateKind::GlmFlash) {
            mark(Marker::ThinkStart);
            if (!thinking) mark(Marker::ThinkEnd);
        } else mark(Marker::ThinkEnd);
    }
    return out;
}
}

#pragma once
#include "server/types.hpp"

namespace zeromoe::server {
enum class TemplateKind { Qwen, MiMo, Glm, GlmFlash, DeepSeek, Bailing };
enum class Marker { Text, Bos, ImStart, ImEnd, Gmask, Sop, User, Assistant, RoleEnd, Eos, ThinkStart, ThinkEnd };
struct Segment {
    Marker marker;
    std::string text;
};
std::vector<Segment> render_chat(const Request &, TemplateKind, bool thinking,
                                 const std::string &default_system, const std::string &effort);
}

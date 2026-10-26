#pragma once

#include <atomic>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace zeromoe::server {
using Json = nlohmann::json;

struct ApiError : std::runtime_error {
    int status;
    ApiError(int status, const std::string &message) : std::runtime_error(message), status(status) {}
};

enum class Protocol { OpenAI, Anthropic };
enum class BlockType { Text, Thinking, ToolUse, ToolResult };
struct Block {
    BlockType type = BlockType::Text;
    std::string text;
    std::string id;
    std::string name;
    Json arguments = Json::object();
    bool is_error = false;
};
struct Message {
    std::string role;
    std::vector<Block> blocks;
};
struct Tool {
    std::string name;
    std::string description;
    Json schema = Json::object();
};
struct Request {
    Protocol protocol = Protocol::OpenAI;
    std::string model;
    std::vector<Message> messages;
    std::vector<Tool> tools;
    std::vector<std::string> stops;
    bool stream = false;
    bool include_usage = false;
    int max_tokens = 1024;
    float temperature = 0.7f;
    float top_p = 0.9f;
};
struct Usage {
    int prompt_tokens = 0;
    int cached_tokens = 0;
    int output_tokens = 0;
};
enum class Finish { Stop, Length, Cancelled };
struct Result {
    Usage usage;
    Finish finish = Finish::Stop;
    std::string stop_sequence;
};
struct ModelInfo {
    std::string id;
    int context_size = 4096;
    int default_max_tokens = 1024;
    int max_output_tokens = 16384;
    float temperature = 0.7f;
    float top_p = 0.9f;
    bool tools = false;
    bool starts_thinking = false;
};
struct Prepared {
    std::vector<int> tokens;
    std::vector<size_t> checkpoints;
};

// Called only by the inference owner, including prepare()/tokenization.
class Backend {
public:
    virtual ~Backend() = default;
    virtual Prepared prepare(const Request &) = 0;
    virtual Result generate(const Request &, const Prepared &, const std::atomic_bool &cancel,
                            const std::function<bool(const std::string &)> &emit) = 0;
};

Request parse_request(const Json &, Protocol, const ModelInfo &, bool count_only = false);
Json error_body(Protocol, int status, const std::string &message);
std::string new_id(const std::string &prefix);
}

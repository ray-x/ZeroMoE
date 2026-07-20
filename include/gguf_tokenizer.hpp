#pragma once
#include "engine.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <array>
#include <climits>
#include <cctype>
#include <iostream>

// Byte-level BPE tokenizer (GPT-2/Llama-3/GLM/DeepSeek-style — the vocab
// convention this project's target architectures all use, per
// tokenizer.ggml.model in their GGUFs). NOT SentencePiece: decode() has a
// fallback for that ("▁" U+2581 as a space marker), used when
// tokenizer.ggml.model isn't a BPE variant, but encode() only implements
// byte-level BPE.
//
// encode()'s pretokenizer is a pragmatic approximation of the reference
// regex-based splitter (llama.cpp/HF each model uses a specific Unicode-
// property regex named by tokenizer.ggml.pre) — std::regex has no \p{L}/\p{N}
// support and this project has no external Unicode library dependency, so
// multi-byte UTF-8 runs are grouped as one opaque "word" rather than split by
// Unicode category. Correct for plain ASCII/Latin input; produces valid
// (in-vocab) tokens for anything else via BPE merging, just not necessarily
// split identically to the reference tokenizer.
class GGUFTokenizer {
public:
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;
    std::unordered_map<std::string, int> merge_rank;  // "sym1 sym2" -> priority (lower = merge first)
    std::string model_type;  // tokenizer.ggml.model, e.g. "gpt2", "llama"
    int bos_token_id = 1;
    int eos_token_id = 2;
    // GLM-3/4/GLMEdge-style chat special tokens, populated (if present) by
    // detect_chat_special_tokens() after the vocab loads -- confirmed
    // against llama.cpp's actual chat-template source (llama-chat.cpp,
    // LLM_CHAT_TEMPLATE_CHATGLM_4): "[gMASK]<sop><|user|>\n{msg}<|assistant|>\n".
    // Detected by literal vocab string lookup rather than assumed outright,
    // since GLM-5.2 specifically could have evolved this further and this
    // project has no way to verify against the real checkpoint. -1 if absent.
    int gmask_id = -1, sop_id = -1, user_id = -1, assistant_id = -1;
    // GLM-5.2's real HF config.json has THREE stop tokens in eos_token_id
    // (an ARRAY: endoftext, user, observation -- confirmed via colibri, a
    // working GLM-5.2 engine that reads the raw HF config directly). A
    // single scalar tokenizer.ggml.eos_token_id (bos_token_id/eos_token_id
    // below) only captures one of them -- generation would run past the
    // model's real turn-boundary markers into undefined continuation
    // territory. Detected by literal vocab lookup, same pattern as the chat
    // template tokens above, and merged with eos_token_id into stop_ids.
    std::vector<int> stop_ids;
    int observation_id = -1;

    GGUFTokenizer() { init_byte_unicode(); }

    bool has_chat_template() const { return user_id >= 0 && assistant_id >= 0; }

    bool is_stop_token(int id) const {
        for (int s : stop_ids) if (s == id) return true;
        return false;
    }

    bool load_from_gguf_metadata(std::ifstream& f, uint64_t kv_count) {
        LOG_STREAM_INFO() << "[Tokenizer] Extracting embedded vocabulary...\n";

        for (uint64_t i = 0; i < kv_count; ++i) {
            std::string key = read_string(f);
            uint32_t value_type;
            f.read(reinterpret_cast<char*>(&value_type), 4);

            if (value_type == 9) {
                uint32_t element_type;
                uint64_t array_len;
                f.read(reinterpret_cast<char*>(&element_type), 4);
                f.read(reinterpret_cast<char*>(&array_len), 8);

                if (key == "tokenizer.ggml.tokens" && element_type == 8) {
                    id_to_token.resize(array_len);
                    for (uint64_t j = 0; j < array_len; ++j) {
                        id_to_token[j] = read_string(f);
                        token_to_id[id_to_token[j]] = (int)j;
                    }
                    LOG_STREAM_INFO() << "[Tokenizer] Loaded " << array_len << " tokens.\n";
                    continue;
                }
                if (key == "tokenizer.ggml.merges" && element_type == 8) {
                    for (uint64_t j = 0; j < array_len; ++j) {
                        merge_rank[read_string(f)] = (int)j;
                    }
                    LOG_STREAM_INFO() << "[Tokenizer] Loaded " << array_len << " BPE merge rules.\n";
                    continue;
                }
                // Some multi-EOS models (GLM-5.2's real HF config.json has
                // eos_token_id as a 3-element array) may carry this through
                // to GGUF as an array KV instead of a scalar -- handle both
                // so it's never silently skipped into the default (2).
                if (key == "tokenizer.ggml.eos_token_id" && (element_type == 4 || element_type == 5)) {
                    for (uint64_t j = 0; j < array_len; ++j) {
                        int32_t v; f.read(reinterpret_cast<char*>(&v), 4);
                        stop_ids.push_back(v);
                        if (j == 0) eos_token_id = v;
                    }
                    LOG_STREAM_INFO() << "[Tokenizer] Loaded " << array_len
                                       << " eos_token_id(s) from array KV.\n";
                    continue;
                }

                for (uint64_t j = 0; j < array_len; ++j) skip_kv_value(f, element_type);
            } else {
                if (key == "tokenizer.ggml.bos_token_id") f.read(reinterpret_cast<char*>(&bos_token_id), 4);
                else if (key == "tokenizer.ggml.eos_token_id") {
                    f.read(reinterpret_cast<char*>(&eos_token_id), 4);
                    stop_ids.push_back(eos_token_id);
                }
                else if (key == "tokenizer.ggml.model" && value_type == 8) model_type = read_string(f);
                else skip_kv_value(f, value_type);
            }
        }
        LOG_STREAM_INFO() << "[Tokenizer] model=\"" << model_type << "\" merges=" << merge_rank.size() << "\n";

        detect_chat_special_tokens();
        if (has_chat_template()) {
            LOG_STREAM_INFO() << "[Tokenizer] Chat template detected: [gMASK]=" << gmask_id
                               << " <sop>=" << sop_id << " <|user|>=" << user_id
                               << " <|assistant|>=" << assistant_id
                               << " -- will wrap prompts GLM-chat-style.\n";
        } else {
            LOG_STREAM_WARN() << "[Tokenizer] No <|user|>/<|assistant|> chat tokens found in vocab -- "
                                  "prompts will be sent as raw BOS+text. If this is an instruction-tuned "
                                  "model, output may look unfocused without the expected chat structure.\n";
        }
        return !id_to_token.empty();
    }

    void detect_chat_special_tokens() {
        auto find = [&](const std::string& s) -> int {
            auto it = token_to_id.find(s);
            return it != token_to_id.end() ? it->second : -1;
        };
        gmask_id = find("[gMASK]");
        sop_id = find("<sop>");
        user_id = find("<|user|>");
        assistant_id = find("<|assistant|>");
        observation_id = find("<|observation|>");

        // GLM-5.2's real HF config.json has THREE eos_token_id entries:
        // <|endoftext|>, <|user|>, <|observation|> (confirmed via colibri,
        // a working GLM-5.2 engine reading the raw HF config directly).
        // <|user|>/<|observation|> mark "the model handed the turn back" --
        // treating them as generation-stop points too, not just the primary
        // <|endoftext|>, matters because generating past them runs into
        // undefined continuation territory the model was never trained to
        // produce (colibri's own comment: "5-10x wasted/garbage tokens").
        auto add_stop = [&](int id) {
            if (id < 0) return;
            for (int s : stop_ids) if (s == id) return;
            stop_ids.push_back(id);
        };
        add_stop(find("<|endoftext|>"));
        add_stop(user_id);
        add_stop(observation_id);

        if (!stop_ids.empty()) {
            LogStream ls(LogLevel::INFO);
            ls << "[Tokenizer] Stop tokens (" << stop_ids.size() << "):";
            for (int id : stop_ids) ls << " " << id;
            ls << "\n";
        }
    }

    // Encodes raw text into token ids via byte-level BPE (see class comment
    // for the pretokenizer's known limitations).
    std::vector<int> encode(const std::string& text) {
        std::vector<int> out;
        for (const auto& chunk : pretokenize(text)) {
            if (chunk.empty()) continue;
            std::vector<std::string> symbols = to_unicode_symbols(chunk);
            bpe_merge(symbols);

            bool all_found = true;
            for (const auto& sym : symbols) {
                if (!token_to_id.count(sym)) { all_found = false; break; }
            }
            if (!all_found) {
                // Anomaly (merge produced something not in vocab) -- fall
                // back to the unmerged single-byte symbols, which are
                // guaranteed present in any complete byte-level BPE vocab
                // (every byte-level BPE trainer seeds the vocab with all 256
                // base byte-symbols). Never silently drops input.
                symbols = to_unicode_symbols(chunk);
            }
            for (const auto& sym : symbols) {
                auto it = token_to_id.find(sym);
                if (it != token_to_id.end()) out.push_back(it->second);
            }
        }
        return out;
    }

    std::string decode(int token_id) {
        if (token_id < 0 || static_cast<size_t>(token_id) >= id_to_token.size()) return "";
        const std::string& token = id_to_token[token_id];

        // Byte-fallback tokens, e.g. "<0x0A>" for a raw newline -- common
        // even in byte-level BPE vocabs for a handful of control bytes.
        if (token.size() == 6 && token[0] == '<' && token[1] == '0' && token[2] == 'x' && token[5] == '>') {
            int hi = hex_digit(token[3]), lo = hex_digit(token[4]);
            if (hi >= 0 && lo >= 0) return std::string(1, (char)((hi << 4) | lo));
        }

        if (model_type != "llama" && model_type != "spm") {
            // Byte-level BPE (default assumption -- matches this project's
            // target architectures, all GPT-2/tiktoken-style vocabs): parse
            // the token's UTF-8 chars back through the byte-level mapping.
            std::string out;
            size_t i = 0;
            while (i < token.size()) {
                unsigned char c0 = (unsigned char)token[i];
                size_t len = (c0 < 0x80) ? 1 : (c0 < 0xE0) ? 2 : (c0 < 0xF0) ? 3 : 4;
                len = std::min(len, token.size() - i);
                std::string uni_char = token.substr(i, len);
                auto it = uni_to_byte.find(uni_char);
                if (it != uni_to_byte.end()) out.push_back((char)it->second);
                else out += uni_char;  // not a mapped byte-symbol (e.g. a special token) -- pass through
                i += len;
            }
            return out;
        }

        // SentencePiece-style fallback: "▁" (U+2581, UTF-8 E2 96 81) marks a
        // leading space. (The pre-existing code here searched for a literal
        // ASCII " " instead of this 3-byte marker -- a no-op/corrupting bug,
        // fixed here.)
        std::string out = token;
        static const std::string SP_MARKER = "\xE2\x96\x81";
        size_t pos = 0;
        while ((pos = out.find(SP_MARKER, pos)) != std::string::npos) {
            out.replace(pos, SP_MARKER.size(), " ");
            pos += 1;
        }
        return out;
    }

private:
    // GPT-2 standard byte<->"unicode" mapping: printable bytes map to
    // themselves, everything else (control chars, space, etc.) maps to a
    // private codepoint range above 255. This lets whitespace/control bytes
    // survive as ordinary-looking multi-byte UTF-8 "characters" through
    // text-based BPE merge rules and vocab lookups. Standard algorithm
    // (identical to HF tokenizers' bytes_to_unicode()), computed once.
    std::array<std::string, 256> byte_to_uni;
    std::unordered_map<std::string, uint8_t> uni_to_byte;  // unicode-char string -> original byte

    void init_byte_unicode() {
        std::vector<int> bs;
        for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
        for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
        for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
        std::array<bool, 256> present{};
        for (int b : bs) present[b] = true;

        std::vector<int> cs = bs;
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (!present[b]) {
                bs.push_back(b);
                cs.push_back(256 + n);
                ++n;
            }
        }
        for (size_t i = 0; i < bs.size(); ++i) {
            std::string utf8 = codepoint_to_utf8((uint32_t)cs[i]);
            byte_to_uni[bs[i]] = utf8;
            uni_to_byte[utf8] = (uint8_t)bs[i];
        }
    }

    static std::string codepoint_to_utf8(uint32_t cp) {
        std::string s;
        if (cp < 0x80) {
            s.push_back((char)cp);
        } else {  // max codepoint used here is ~323, so 2 bytes always suffice
            s.push_back((char)(0xC0 | (cp >> 6)));
            s.push_back((char)(0x80 | (cp & 0x3F)));
        }
        return s;
    }

    std::vector<std::string> to_unicode_symbols(const std::string& chunk) const {
        std::vector<std::string> syms;
        syms.reserve(chunk.size());
        for (unsigned char b : chunk) syms.push_back(byte_to_uni[b]);
        return syms;
    }

    void bpe_merge(std::vector<std::string>& symbols) const {
        while (symbols.size() >= 2) {
            int best_rank = INT_MAX;
            size_t best_i = SIZE_MAX;
            for (size_t i = 0; i + 1 < symbols.size(); ++i) {
                auto it = merge_rank.find(symbols[i] + " " + symbols[i + 1]);
                if (it != merge_rank.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_i = i;
                }
            }
            if (best_i == SIZE_MAX) break;
            symbols[best_i] += symbols[best_i + 1];
            symbols.erase(symbols.begin() + best_i + 1);
        }
    }

    // Approximate GPT-2-style pretokenizer: a leading run of whitespace
    // (beyond one space) becomes its own chunk; a single leading space
    // attaches to the following word (matching the "Ġ" convention); each
    // word chunk is then a run of digits, a run of "word bytes" (ASCII
    // alnum or any UTF-8 non-ASCII byte, treated opaquely), or a single
    // punctuation/symbol character.
    static std::vector<std::string> pretokenize(const std::string& text) {
        std::vector<std::string> chunks;
        size_t i = 0, n = text.size();
        while (i < n) {
            size_t ws_start = i;
            while (i < n && std::isspace((unsigned char)text[i])) ++i;
            size_t ws_len = i - ws_start;

            if (i >= n) {
                if (ws_len > 0) chunks.push_back(text.substr(ws_start, ws_len));
                break;
            }

            if (ws_len > 1) chunks.push_back(text.substr(ws_start, ws_len - 1));
            size_t word_start = (ws_len > 0) ? i - 1 : ws_start;

            unsigned char c0 = (unsigned char)text[i];
            size_t j = i;
            if (std::isdigit(c0)) {
                while (j < n && std::isdigit((unsigned char)text[j])) ++j;
            } else if (std::isalnum(c0) || c0 >= 0x80) {
                while (j < n && (std::isalnum((unsigned char)text[j]) || (unsigned char)text[j] >= 0x80)
                       && !std::isdigit((unsigned char)text[j])) ++j;
            } else {
                ++j;  // single punctuation/symbol char per chunk
            }
            chunks.push_back(text.substr(word_start, j - word_start));
            i = j;
        }
        return chunks;
    }

    static int hex_digit(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    }

    std::string read_string(std::ifstream& f) {
        uint64_t len;
        f.read(reinterpret_cast<char*>(&len), 8);
        std::string s(len, '\0');
        f.read(&s[0], len);
        return s;
    }

    void skip_kv_value(std::ifstream& f, uint32_t type) {
        switch (type) {
            case 0: case 1: case 7: f.seekg(1, std::ios::cur); break;  // u8/i8/bool
            case 2: case 3: f.seekg(2, std::ios::cur); break;          // u16/i16
            case 4: case 5: case 6: f.seekg(4, std::ios::cur); break;  // u32/i32/f32
            case 8: { uint64_t len; f.read(reinterpret_cast<char*>(&len), 8); f.seekg(len, std::ios::cur); break; }
            case 10: case 11: case 12: f.seekg(8, std::ios::cur); break; // u64/i64/f64
            default: break;
        }
    }
};

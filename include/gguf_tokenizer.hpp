#pragma once
#include "logger.hpp"
#include <array>
#include <cctype>
#include <climits>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

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
  std::unordered_map<std::string, int>
      merge_rank;         // "sym1 sym2" -> priority (lower = merge first)
  std::string model_type; // tokenizer.ggml.model, e.g. "gpt2", "llama"
  std::string pre_type;   // tokenizer.ggml.pre, e.g. "glm4" (selects the pretokenizer regex)
  int bos_token_id = 1;
  int eos_token_id = 2;
  // tokenizer.ggml.add_bos_token: whether the model's OWN tokenizer config
  // wants a leading BOS prepended at all. Defaults to true (matching the
  // convention most GGUFs rely on when the key is absent), but MUST be
  // respected when present -- confirmed via a real bug: Qwen3.6-35B-A3B's
  // GGUF carries `tokenizer.ggml.add_bos_token = False`, which llama.cpp
  // reads via llama_vocab_get_add_bos() and honors (a real llama.cpp run
  // logged "number of input tokens = 1" for the prompt "hi", i.e. no BOS).
  // Before this field existed, cpu_main.cpp/main.cpp unconditionally pushed
  // bos_token_id regardless of this key, silently feeding the model a
  // 2-token prompt (phantom BOS + "hi") it never saw start a sequence
  // during training/inference elsewhere -- this corrupted every windowed op
  // (causal_conv1d's receptive field) from the very first forward pass and
  // permanently poisoned the recurrent linear-attention state for the rest
  // of generation, which is believed to be the root cause of incoherent
  // decode output for this model.
  bool add_bos_token = true;
  // GLM-3/4/GLMEdge-style chat special tokens, populated (if present) by
  // detect_chat_special_tokens() after the vocab loads -- confirmed
  // against llama.cpp's actual chat-template source (llama-chat.cpp,
  // LLM_CHAT_TEMPLATE_CHATGLM_4): "[gMASK]<sop><|user|>\n{msg}<|assistant|>\n".
  // Detected by literal vocab string lookup rather than assumed outright,
  // since GLM-5.2 specifically could have evolved this further and this
  // project has no way to verify against the real checkpoint. -1 if absent.
  int gmask_id = -1, sop_id = -1, user_id = -1, assistant_id = -1;
  // GLM-5.2's real HF config.json has THREE stop tokens in eos_token_id
  std::vector<int> stop_ids;
  int observation_id = -1;
  // DeepSeek-V3/V4-style chat special tokens -- confirmed against the real
  // HF chat template (models/templates/deepseek-ai-DeepSeek-V4-Flash-0731.jinja
  // in the llama.cpp checkout): "{bos}<｜User｜>{msg}<｜Assistant｜></think>".
  // NOTE these use FULLWIDTH VERTICAL LINE (｜ U+FF5C), not ASCII '|' -- a
  // plain "<|user|>" lookup (the GLM check above) will never match DeepSeek's
  // vocab, which is exactly the bug this was added to fix: without this,
  // has_chat_template() was false for every DeepSeek checkpoint and prompts
  // fell through to raw BOS+text, feeding an instruction/reasoning-tuned
  // model a completion-style prompt it was never trained to see -- observed
  // on a real DeepSeek-V4-Flash run as confident, fluent, but completely
  // off-topic/multilingual-garbage decode output for the plain prompt "hi",
  // with every tensor-math stage (attention, hyper-connections, hash-
  // routing, weight-gathering, embedding) independently reviewed and found
  // correct -- the corruption was in prompt construction, not the model.
  // -1 if absent. ds_think_end_id ("</think>") is appended after the
  // assistant tag to select the template's default non-reasoning path
  // (thinking=false in the jinja source); this project has no "thinking"
  // toggle so always requesting the non-reasoning branch is the correct
  // default absent one.
  int ds_user_id = -1, ds_assistant_id = -1, ds_think_start_id = -1,
      ds_think_end_id = -1;

  // ChatML (Qwen3.8-Flash-Next / qwen4exp): <|im_start|>/<|im_end|> turn
  // markers. The template is ChatML with thinking enabled by default:
  //   <|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n<think>\n
  // (a system turn with the reasoning-effort instructions precedes it; see
  // the model's tokenizer_config.json chat_template).
  int im_start_id = -1, im_end_id = -1;

  GGUFTokenizer() { init_byte_unicode(); }

  bool has_chat_template() const { return user_id >= 0 && assistant_id >= 0; }
  bool has_deepseek_chat_template() const {
    return ds_user_id >= 0 && ds_assistant_id >= 0;
  }
  // ChatML (qwen4exp and Qwen chat models generally) -- turn markers only;
  // roles are plain text after <|im_start|>.
  bool has_chatml_template() const { return im_start_id >= 0 && im_end_id >= 0;
  }
  bool is_loaded() const { return !id_to_token.empty(); }

  bool is_stop_token(int id) const {
    for (int s : stop_ids)
      if (s == id)
        return true;
    return false;
  }

  // Convenience: load tokenizer from an already-parsed GGUFFile by re-reading
  // its header metadata.  This keeps GGUFTokenizer usable by code paths that
  // do not hold an open ifstream themselves.
  bool load(const GGUFFile &file) {
    std::ifstream f(file.resolved_file_path, std::ios::binary);
    if (!f.is_open())
      return false;

    char magic[4];
    f.read(magic, 4);
    if (std::string(magic, 4) != "GGUF")
      return false;

    uint32_t version = 0;
    f.read(reinterpret_cast<char *>(&version), 4);

    uint64_t tensor_count = 0, kv_count = 0;
    f.read(reinterpret_cast<char *>(&tensor_count), 8);
    f.read(reinterpret_cast<char *>(&kv_count), 8);

    return load_from_gguf_metadata(f, kv_count);
  }

  bool load_from_gguf_metadata(std::ifstream &f, uint64_t kv_count) {
    LOG_STREAM_INFO() << "[Tokenizer] Extracting embedded vocabulary...\n";

    for (uint64_t i = 0; i < kv_count; ++i) {
      std::string key = read_string(f);
      uint32_t value_type;
      f.read(reinterpret_cast<char *>(&value_type), 4);

      if (value_type == 9) {
        uint32_t element_type;
        uint64_t array_len;
        f.read(reinterpret_cast<char *>(&element_type), 4);
        f.read(reinterpret_cast<char *>(&array_len), 8);

        if (key == "tokenizer.ggml.tokens" && element_type == 8) {
          id_to_token.resize(array_len);
          for (uint64_t j = 0; j < array_len; ++j) {
            id_to_token[j] = read_string(f);
            token_to_id[id_to_token[j]] = (int)j;
          }
          LOG_STREAM_INFO()
              << "[Tokenizer] Loaded " << array_len << " tokens.\n";
          continue;
        }
        if (key == "tokenizer.ggml.merges" && element_type == 8) {
          for (uint64_t j = 0; j < array_len; ++j) {
            merge_rank[read_string(f)] = (int)j;
          }
          LOG_STREAM_INFO()
              << "[Tokenizer] Loaded " << array_len << " BPE merge rules.\n";
          continue;
        }
        // Some multi-EOS models (GLM-5.2's real HF config.json has
        // eos_token_id as a 3-element array) may carry this through
        // to GGUF as an array KV instead of a scalar -- handle both
        // so it's never silently skipped into the default (2).
        if (key == "tokenizer.ggml.eos_token_id" &&
            (element_type == 4 || element_type == 5)) {
          for (uint64_t j = 0; j < array_len; ++j) {
            int32_t v;
            f.read(reinterpret_cast<char *>(&v), 4);
            stop_ids.push_back(v);
            if (j == 0)
              eos_token_id = v;
          }
          LOG_STREAM_INFO() << "[Tokenizer] Loaded " << array_len
                            << " eos_token_id(s) from array KV.\n";
          continue;
        }

        for (uint64_t j = 0; j < array_len; ++j)
          skip_kv_value(f, element_type);
      } else {
        if (key == "tokenizer.ggml.bos_token_id")
          f.read(reinterpret_cast<char *>(&bos_token_id), 4);
        else if (key == "tokenizer.ggml.add_bos_token" && value_type == 7) {
          uint8_t v = 0;
          f.read(reinterpret_cast<char *>(&v), 1);
          add_bos_token = (v != 0);
        } else if (key == "tokenizer.ggml.eos_token_id") {
          f.read(reinterpret_cast<char *>(&eos_token_id), 4);
          stop_ids.push_back(eos_token_id);
        } else if (key == "tokenizer.ggml.model" && value_type == 8)
          model_type = read_string(f);
        else if (key == "tokenizer.ggml.pre" && value_type == 8)
          pre_type = read_string(f);
        else
          skip_kv_value(f, value_type);
      }
    }
    LOG_STREAM_INFO() << "[Tokenizer] model=\"" << model_type
                      << "\" merges=" << merge_rank.size() << "\n";

    detect_chat_special_tokens();
    if (has_chat_template()) {
      LOG_STREAM_INFO() << "[Tokenizer] Chat template detected: [gMASK]="
                        << gmask_id << " <sop>=" << sop_id
                        << " <|user|>=" << user_id
                        << " <|assistant|>=" << assistant_id
                        << " -- will wrap prompts GLM-chat-style.\n";
    } else if (has_deepseek_chat_template()) {
      LOG_STREAM_INFO() << "[Tokenizer] Chat template detected: <｜User｜>="
                        << ds_user_id << " <｜Assistant｜>=" << ds_assistant_id
                        << " </think>=" << ds_think_end_id
                        << " -- will wrap prompts DeepSeek-chat-style.\n";
    } else if (has_chatml_template()) {
      LOG_STREAM_INFO() << "[Tokenizer] Chat template detected: <|im_start|>="
                        << im_start_id << " <|im_end|>=" << im_end_id
                        << " -- will wrap prompts ChatML-style.\n";
    } else {
      LOG_STREAM_WARN() << "[Tokenizer] No <|user|>/<|assistant|> or "
                           "<｜User｜>/<｜Assistant｜> chat tokens "
                           "found in vocab -- "
                           "prompts will be sent as raw BOS+text. If this is "
                           "an instruction-tuned "
                           "model, output may look unfocused without the "
                           "expected chat structure.\n";
    }
    return !id_to_token.empty();
  }

  void detect_chat_special_tokens() {
    auto find = [&](const std::string &s) -> int {
      auto it = token_to_id.find(s);
      return it != token_to_id.end() ? it->second : -1;
    };
    gmask_id = find("[gMASK]");
    sop_id = find("<sop>");
    user_id = find("<|user|>");
    assistant_id = find("<|assistant|>");
    observation_id = find("<|observation|>");
    ds_user_id = find("<｜User｜>");
    ds_assistant_id = find("<｜Assistant｜>");
    ds_think_start_id = find("<think>");
    ds_think_end_id = find("</think>");
    im_start_id = find("<|im_start|>");
    im_end_id = find("<|im_end|>");

    // GLM-5.2's real HF config.json has THREE eos_token_id entries:
    // <|endoftext|>, <|user|>, <|observation|>
    // <|user|>/<|observation|> mark "the model handed the turn back" --
    // treating them as generation-stop points too, not just the primary
    // <|endoftext|>, matters because generating past them runs into
    // undefined continuation territory the model was never trained to
    // produce
    auto add_stop = [&](int id) {
      if (id < 0)
        return;
      for (int s : stop_ids)
        if (s == id)
          return;
      stop_ids.push_back(id);
    };
    add_stop(find("<|endoftext|>"));
    add_stop(user_id);
    add_stop(observation_id);
    // DeepSeek's real turn-end marker (fullwidth-pipe, distinct from the
    // ASCII <|endoftext|> checked above) -- see ds_user_id's comment for the
    // Unicode-token-family context. tokenizer.ggml.eos_token_id from the
    // GGUF metadata should already point here in a well-formed checkpoint,
    // but add_stop dedups, so this is a harmless extra safety net rather
    // than a behavior change when that's already true.
    add_stop(find("<｜end▁of▁sentence｜>"));
    // ChatML turn-end marker (qwen4exp): the assistant turn closes with
    // <|im_end|> -- a real stop point, same "handed the turn back"
    // reasoning as GLM's <|user|> above. Dedup makes this a no-op when the
    // GGUF's own eos_token_id already IS <|im_end|>.
    add_stop(im_end_id);

    if (!stop_ids.empty()) {
      LogStream ls(LogLevel::INFO);
      ls << "[Tokenizer] Stop tokens (" << stop_ids.size() << "):";
      for (int id : stop_ids)
        ls << " " << id;
      ls << "\n";
    }
  }

  // Encodes raw text into token ids via byte-level BPE (see class comment
  // for the pretokenizer's known limitations).
  std::vector<int> encode(const std::string &text) {
    std::vector<int> out;
    for (const auto &chunk :
         (pre_type == "glm4" ? pretokenize_glm4(text) : pretokenize(text))) {
      if (chunk.empty())
        continue;
      std::vector<std::string> symbols = to_unicode_symbols(chunk);
      bpe_merge(symbols);

      bool all_found = true;
      for (const auto &sym : symbols) {
        if (!token_to_id.count(sym)) {
          all_found = false;
          break;
        }
      }
      if (!all_found) {
        // Anomaly (merge produced something not in vocab) -- fall
        // back to the unmerged single-byte symbols, which are
        // guaranteed present in any complete byte-level BPE vocab
        // (every byte-level BPE trainer seeds the vocab with all 256
        // base byte-symbols). Never silently drops input.
        symbols = to_unicode_symbols(chunk);
      }
      for (const auto &sym : symbols) {
        auto it = token_to_id.find(sym);
        if (it != token_to_id.end())
          out.push_back(it->second);
      }
    }
    return out;
  }

  std::string decode(int token_id) {
    if (token_id < 0 || static_cast<size_t>(token_id) >= id_to_token.size())
      return "";
    const std::string &token = id_to_token[token_id];

    // Byte-fallback tokens, e.g. "<0x0A>" for a raw newline -- common
    // even in byte-level BPE vocabs for a handful of control bytes.
    if (token.size() == 6 && token[0] == '<' && token[1] == '0' &&
        token[2] == 'x' && token[5] == '>') {
      int hi = hex_digit(token[3]), lo = hex_digit(token[4]);
      if (hi >= 0 && lo >= 0)
        return std::string(1, (char)((hi << 4) | lo));
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
        if (it != uni_to_byte.end())
          out.push_back((char)it->second);
        else
          out += uni_char; // not a mapped byte-symbol (e.g. a special token) --
                           // pass through
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
  std::unordered_map<std::string, uint8_t>
      uni_to_byte; // unicode-char string -> original byte

  void init_byte_unicode() {
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; ++b)
      bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b)
      bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b)
      bs.push_back(b);
    std::array<bool, 256> present{};
    for (int b : bs)
      present[b] = true;

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
    } else { // max codepoint used here is ~323, so 2 bytes always suffice
      s.push_back((char)(0xC0 | (cp >> 6)));
      s.push_back((char)(0x80 | (cp & 0x3F)));
    }
    return s;
  }

  std::vector<std::string> to_unicode_symbols(const std::string &chunk) const {
    std::vector<std::string> syms;
    syms.reserve(chunk.size());
    for (unsigned char b : chunk)
      syms.push_back(byte_to_uni[b]);
    return syms;
  }

  void bpe_merge(std::vector<std::string> &symbols) const {
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
      if (best_i == SIZE_MAX)
        break;
      symbols[best_i] += symbols[best_i + 1];
      symbols.erase(symbols.begin() + best_i + 1);
    }
  }

  // llama.cpp's LLAMA_VOCAB_PRE_TYPE_GLM4 split, implemented by hand (no \p{..} support in std::regex):
  //   (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|
  //   ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
  // \p{L} is approximated by ASCII letters plus any non-ASCII byte, \p{N} by ASCII digits, \s by ASCII whitespace.
  static std::vector<std::string> pretokenize_glm4(const std::string &text) {
    const size_t n = text.size();
    auto is_ws = [&](size_t k) { return k < n && std::isspace((unsigned char)text[k]); };
    auto is_nl = [&](size_t k) { return k < n && (text[k] == '\r' || text[k] == '\n'); };
    auto is_digit = [&](size_t k) { return k < n && std::isdigit((unsigned char)text[k]); };
    auto is_letter = [&](size_t k) {
      return k < n && (std::isalpha((unsigned char)text[k]) || (unsigned char)text[k] >= 0x80);
    };
    std::vector<std::string> out;
    size_t i = 0;
    while (i < n) {
      size_t j = i;
      // contractions
      if (text[i] == '\'' && i + 1 < n) {
        const char a = (char)std::tolower((unsigned char)text[i + 1]);
        const char b = i + 2 < n ? (char)std::tolower((unsigned char)text[i + 2]) : '\0';
        if (a == 's' || a == 't' || a == 'm' || a == 'd') j = i + 2;
        else if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) j = i + 3;
        if (j > i) { out.push_back(text.substr(i, j - i)); i = j; continue; }
      }
      // [^\r\n\p{L}\p{N}]?\p{L}+
      {
        size_t k = i;
        if (!is_nl(k) && !is_letter(k) && !is_digit(k) && is_letter(k + 1)) ++k;
        if (is_letter(k)) {
          while (is_letter(k)) ++k;
          out.push_back(text.substr(i, k - i)); i = k; continue;
        }
      }
      // \p{N}{1,3}
      if (is_digit(i)) {
        size_t k = i;
        while (k < n && k - i < 3 && is_digit(k)) ++k;
        out.push_back(text.substr(i, k - i)); i = k; continue;
      }
      // ' ?[^\s\p{L}\p{N}]+[\r\n]*'
      {
        size_t k = i;
        if (text[k] == ' ') ++k;
        const size_t body = k;
        while (k < n && !is_ws(k) && !is_letter(k) && !is_digit(k)) ++k;
        if (k > body) {
          while (is_nl(k)) ++k;
          out.push_back(text.substr(i, k - i)); i = k; continue;
        }
      }
      // \s*[\r\n]+  (whitespace up to the last newline of the run)
      if (is_ws(i)) {
        size_t k = i, last_nl = n;
        while (is_ws(k)) { if (is_nl(k)) last_nl = k; ++k; }
        if (last_nl != n) { out.push_back(text.substr(i, last_nl + 1 - i)); i = last_nl + 1; continue; }
        // \s+(?!\S): leave the final whitespace char for the next word when a non-space follows
        size_t end = k;
        if (end < n && end - i > 1) --end;
        else if (end < n) end = k; // single space before a word: falls to \s+ below
        out.push_back(text.substr(i, end - i)); i = end; continue;
      }
      out.push_back(text.substr(i, 1)); ++i;
    }
    return out;
  }

  // Approximate GPT-2-style pretokenizer: a leading run of whitespace
  // (beyond one space) becomes its own chunk; a single leading space
  // attaches to the following word (matching the "Ġ" convention); each
  // word chunk is then a run of digits, a run of "word bytes" (ASCII
  // alnum or any UTF-8 non-ASCII byte, treated opaquely), or a single
  // punctuation/symbol character.
  static std::vector<std::string> pretokenize(const std::string &text) {
    std::vector<std::string> chunks;
    size_t i = 0, n = text.size();
    while (i < n) {
      size_t ws_start = i;
      while (i < n && std::isspace((unsigned char)text[i]))
        ++i;
      size_t ws_len = i - ws_start;

      if (i >= n) {
        if (ws_len > 0)
          chunks.push_back(text.substr(ws_start, ws_len));
        break;
      }

      if (ws_len > 1)
        chunks.push_back(text.substr(ws_start, ws_len - 1));
      size_t word_start = (ws_len > 0) ? i - 1 : ws_start;

      unsigned char c0 = (unsigned char)text[i];
      size_t j = i;
      if (std::isdigit(c0)) {
        while (j < n && std::isdigit((unsigned char)text[j]))
          ++j;
      } else if (std::isalnum(c0) || c0 >= 0x80) {
        while (j < n &&
               (std::isalnum((unsigned char)text[j]) ||
                (unsigned char)text[j] >= 0x80) &&
               !std::isdigit((unsigned char)text[j]))
          ++j;
      } else {
        ++j; // single punctuation/symbol char per chunk
      }
      chunks.push_back(text.substr(word_start, j - word_start));
      i = j;
    }
    return chunks;
  }

  static int hex_digit(char c) {
    if (c >= '0' && c <= '9')
      return c - '0';
    if (c >= 'a' && c <= 'f')
      return c - 'a' + 10;
    if (c >= 'A' && c <= 'F')
      return c - 'A' + 10;
    return -1;
  }

  std::string read_string(std::ifstream &f) {
    uint64_t len;
    f.read(reinterpret_cast<char *>(&len), 8);
    std::string s(len, '\0');
    f.read(&s[0], len);
    return s;
  }

  void skip_kv_value(std::ifstream &f, uint32_t type) {
    switch (type) {
    case 0:
    case 1:
    case 7:
      f.seekg(1, std::ios::cur);
      break; // u8/i8/bool
    case 2:
    case 3:
      f.seekg(2, std::ios::cur);
      break; // u16/i16
    case 4:
    case 5:
    case 6:
      f.seekg(4, std::ios::cur);
      break; // u32/i32/f32
    case 8: {
      uint64_t len;
      f.read(reinterpret_cast<char *>(&len), 8);
      f.seekg(len, std::ios::cur);
      break;
    }
    case 10:
    case 11:
    case 12:
      f.seekg(8, std::ios::cur);
      break; // u64/i64/f64
    default:
      break;
    }
  }
};

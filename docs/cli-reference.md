# Command-line reference

Flags accepted by `zeromoe`. Run `zeromoe --help` for the same list on
a built binary.

| Flag | Description |
| --- | --- |
| `--config <file>` | Load engine config from a TOML file (preferred over legacy CLI flags). |
| `--gen-config [output.toml]` | Generate a sample config file (default `zeromoe_config.toml`) and exit. |
| `--model <path>` | Path to a GGUF model file or directory (legacy CLI mode, used without `--config`). |
| `--ram <GB>` | Strict RAM budget in GB (default: auto-detect 88% of free system memory). |
| `-c, --ctx-size <N>` | Maximum context size / token window (default: 4096). |
| `-n, --predict <N>` | Maximum new tokens to generate per response (default: 1024). |
| `-t, --threads <N>` | CPU thread limit (default: automatic). |
| `--temp <F>` | Sampling temperature; `0.0` for greedy decode (default: 0.7). |
| `--top-p <F>` | Nucleus sampling top-p threshold (default: 0.90). |
| `--repeat-penalty <F>` | Repetition penalty (default: 1.0, disabled). |
| `--main-gpu <ID>` | CUDA device ID for single-GPU mode (default: 0). |
| `--serve` | Serve the OpenAI Chat Completions and Anthropic Messages APIs from the loaded model. |
| `--host <addr>` | API bind address (default `127.0.0.1`; a non-loopback bind requires `--api-key` or `ZEROMOE_API_KEY`). |
| `--port <port>` | API port (default 8080). |
| `--alias <name>` | API model name (default `zeromoe`). |
| `--api-key <key>` | API key checked on API routes; also read from `ZEROMOE_API_KEY`. |
| `--spec-type <none|draft-mtp>` | Speculative mode supported by ZeroMoE. |
| `--spec-draft-model <path>` | Optional external draft checkpoint. |
| `--spec-draft-n-max <N>` | Maximum speculative draft length (default: 3). |
| `--drop-pages` | Drop expert pages from the OS page cache after use. |
| `--direct-io` | `O_DIRECT` expert-slab reads (Linux only; requires a model repacked with `tools/gguf_tool pack`). Falls back to buffered reads automatically if unsupported/unaligned. (`unpacked_expert_format` is TOML-config-only — see `--config`.) |
| `--io-uring` | Batch the async prefetcher's per-expert reads through `io_uring` (Linux only, kernel ≥ 5.6; independent of `--direct-io`). Falls back to sequential `pread`s automatically if unsupported. |
| `--prefetch-threads <N>` | Concurrent async prefetcher worker threads (default: 2). |
| `--promote-threshold <N>` | Number of host re-hits before an async host→device (H2D) promote (default: 1). |
| `--pin-threshold <N>` | VRAM-resident hit-count before permanently pinning an expert (never evicted; default: 8, `0` = off). |
| `--pin-file <file>` | Usage-frequency file for VRAM hot-set preload at startup (default: `.zeromoe_usage`). |
| `--expert-fetch-threads <N>` | OMP threads for parallel disk reads of a layer's top-k cache misses (default: 4). |
| `--raw-prompt` | Skip the chat-template wrap; tokenize stdin verbatim. Useful for token-count-fair benchmarking and for clients that apply their own chat formatting. |
| `--system-prompt <text>` | Prepend a system-role message ahead of the user turn (chat-template modes only; ignored with `--raw-prompt`). Empty by default — ZeroMoE never invents one on its own. Use this to steer behavior/language, e.g. `--system-prompt "Respond in English unless asked otherwise."` |
| `--log-file <file>` | Redirect logging to a file instead of stderr (llama.cpp-compatible name). |
| `--verbosity <0..5>` | Set log threshold (llama.cpp-compatible numeric verbosity). |
| `--help`, `-h` | Show the CLI help interface. |

At DEBUG (`--verbosity 5`), numerical decode comparisons include
`[TEACHER-FORCE-LOGITS]` fingerprints and Qwen route-hit summaries use the
`[ROUTE-HIT]`, `[MOE-REQUIRED]`, and `[MOE-PROMPT]` headers. Filter these records from a verbose log by their header.

> Most of these flags have a matching TOML key of the same name (e.g. `raw_prompt`, `system_prompt`, `pin_threshold`) for use with `--config`; a config file's values are used unless the equivalent CLI flag is also passed, in which case the CLI flag wins. Run `--gen-config` to see the full set of TOML-only options (e.g. `unpacked_expert_format`, `pin_strategy`) that have no CLI-flag equivalent.

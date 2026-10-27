# Native C++ API server

ZeroMoE can serve one loaded model through OpenAI Chat Completions and
Anthropic Messages. HTTP workers submit typed jobs to a bounded queue; one
inference owner performs tokenization and generation. Model state never belongs
to an HTTP worker. There is no subprocess gateway or continuous batching.

This is an initial implementation, not a claim of drop-in compatibility with
every agent client. Protocol integration and synthetic CPU checkpoint tests
exist; real-GGUF tool/template behavior, reference-logit parity, and the native
Linux/CUDA serving build still require validation.

## Build

The server is built by default. Building it fetches pinned cpp-httplib and
nlohmann/json dependencies; it does not download model weights. Configure with
`-DZEROMOE_BUILD_SERVER=OFF` to omit it.

Linux/CUDA:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DZEROMOE_USE_GGML_CUDA=ON
cmake --build build --target zeromoe server_tests -j"$(nproc)"
```

macOS/CPU:

```bash
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release \
  -DZEROMOE_CPU_ONLY=ON
cmake --build build-cpu --target zeromoe_cpu server_tests cpu_api_tests \
  -j"$(sysctl -n hw.ncpu)"
ctest --test-dir build-cpu --output-on-failure
```

The CPU API currently accepts GGUF models using the generic Qwen MoE backend
(including Qwen3.6 MoE), not every model accepted by the CPU CLI. EXL3 and
safetensors serving are rejected. CUDA serving uses the registered full-graph
adapter for the exact `qwen35moe` GGUF architecture; other architectures still
need their own adapter and memory plan.

## Start

Use a working model configuration with real local weight paths and enough
context for the complete conversation plus the requested output. Before a real
run, check existing inference processes and available RAM; on CUDA hosts also
check competing GPU processes/free VRAM. Do not start another engine against an
already allocated memory budget.

```bash
export ZEROMOE_API_KEY='replace-with-a-long-random-value'
./build/zeromoe --config /path/to/model.toml --serve \
  --host 127.0.0.1 --port 8080 --api-model zeromoe
```

For the supported CPU backend:

```bash
./build-cpu/zeromoe_cpu --config /path/to/qwen.toml --serve \
  --host 127.0.0.1 --port 8080 --api-model zeromoe
```

`max_new_tokens` in the config is both the default output limit and the maximum
allowed per request (`--ngen` can override it). Set it high enough for agent tool
responses and reasoning. Requests exceeding the model context or output limit
fail rather than truncating history. Thinking mode is set by the engine config,
not a per-request reasoning parameter. Raw-prompt mode is incompatible with
serving.

The default address is `127.0.0.1:8080`. Without a key, only loopback binding is
allowed; this is for trusted local development. With a key, API routes accept
`Authorization: Bearer ...` or `x-api-key: ...`. `/health` is unauthenticated and
contains no model prompts. A configured key is required even for local API
requests.

HTTP is plaintext. Keep it on loopback or a trusted private network and use a
TLS reverse proxy for remote access. Disable proxy response buffering for SSE.
One configured key means one trust/cache namespace: this is not a multi-tenant
service. Request IDs are returned in `X-Request-Id`.

## Endpoints

| Endpoint | Supported behavior |
| --- | --- |
| `GET /health` | Readiness status |
| `GET /v1/models` | One configured model ID |
| `POST /v1/chat/completions` | Text, reasoning content, function tools, SSE, usage |
| `POST /v1/messages` | Text/thinking/tool blocks, SSE, usage |
| `POST /v1/messages/count_tokens` | Tokenize the rendered prompt without generation |

OpenAI example:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $ZEROMOE_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"zeromoe","messages":[{"role":"system","content":"You are a coding assistant."},{"role":"user","content":"Explain RAII briefly."}],"max_tokens":128}'
```

Anthropic streaming example:

```bash
curl -N http://127.0.0.1:8080/v1/messages \
  -H "x-api-key: $ZEROMOE_API_KEY" \
  -H 'anthropic-version: 2023-06-01' \
  -H 'Content-Type: application/json' \
  -d '{"model":"zeromoe","system":"You are a coding assistant.","messages":[{"role":"user","content":"Explain RAII briefly."}],"max_tokens":128,"stream":true}'
```

For an OpenAI-compatible client use base URL `http://127.0.0.1:8080/v1`.
For an Anthropic-compatible client use `http://127.0.0.1:8080`.
Set the model to the `--api-model` value; no model alias lookup is performed.
OpenAI streaming usage is opt-in with
`"stream_options":{"include_usage":true}`.

## Tools and protocol limits

Tool rendering is currently supported only for the Qwen ChatML adapters. Other
native family adapters support text conversations and reject tools. Templates
follow the engine's family conventions; this implementation does not execute
arbitrary GGUF Jinja templates.

Supply OpenAI `tools` with `type: "function"` and an object `parameters` schema,
or Anthropic `tools` with an object `input_schema`. Return tool results in the
next conversation using the generated call ID. The server validates call/result
relationships, buffers generated tool JSON until it is complete, and rejects
malformed or undeclared calls. It never executes tools on the host.

Only automatic tool choice is supported. Tool schemas guide the prompt; they
are not grammar-enforced. `strict: true`, forced tool choice, and
`parallel_tool_calls: false` are rejected. Tool arguments are emitted only after
the complete call parses, not character by character before validation.

OpenAI supports `max_tokens` or `max_completion_tokens`, not both. Anthropic
requires `max_tokens` for generation. Sampling supports temperature and top-p;
stop sequences work across token fragments. Unsupported request fields are
rejected instead of silently ignored.

Not implemented: OpenAI Responses, embeddings, multimodal input, hosted tools,
JSON-schema constrained generation, multiple completions, logprobs, or
request-controlled reasoning. Agents requiring those features need adaptation.
Anthropic thinking blocks are native model reasoning, not Anthropic-signed
reasoning.

## Prompt-state caching

Caching means reusing actual inference state, not returning a previous answer.
Keep stable system instructions and tool definitions first, maintain their
ordering/content, and append conversation turns without rewriting earlier
messages. Reuse requires an identical rendered token prefix.

**CPU Qwen:** the experimental cache is disabled by default pending real-model
teacher-forced validation. Opt in with a RAM budget:

```bash
./build-cpu/zeromoe_cpu --config /path/to/qwen.toml --serve \
  --api-cache-mib 256
```

It stores token IDs, dense KV prefixes, recurrent state, convolution history,
position, and last logits. Checkpoints are taken at message boundaries and
forwarded generation prefixes. A longest-prefix match is restored before
prefilling the remaining tokens. Unrelated requests reset live sequence state;
they cannot inherit another conversation's rolling state.

The cache uses LRU eviction, a byte budget, and at most 16 checkpoints. Oversize
checkpoints are skipped with a diagnostic. The budget covers retained snapshots,
not model weights, active KV, inference scratch, or temporary allocation
overhead. Budgeting snapshots does not make the rest of inference fit in RAM.
There is no cache persistence or tenant isolation.

**Native CUDA:** API prompt caching is disabled. The REPL checkpoint is not a
complete graph-aware session snapshot, so the server does not advertise its
reuse as safe. Graph-owned state and stable captured bindings require separate
checkpoint integration.

OpenAI reports actual reused tokens in `usage.prompt_tokens_details.cached_tokens`.
Anthropic reports them in `cache_read_input_tokens`, excluding them from
`input_tokens`; `cache_creation_input_tokens` remains zero. In streaming
responses the final usage event is authoritative. Anthropic `cache_control`
blocks are accepted as compatibility hints only; they do not enable caching,
guarantee retention, implement provider billing semantics, or enforce a `5m`/`1h`
TTL.

## Scheduling and cancellation

There is one active generation and up to 16 queued inference requests by
default. Full queues return HTTP 429. HTTP worker/socket admission and per-job
output buffers are also bounded. Defaults include an 8 MiB body limit, a
256 KiB raw-output queue per job, and a 600-second queue-plus-generation
deadline.

SSE uses protocol-native events and terminal markers. Errors after HTTP headers
are sent are delivered as stream errors, not successful completion markers.
Slow clients apply backpressure; disconnects, expired deadlines, and shutdown
cancel pending work. Cancellation is cooperative between engine steps (and CPU
layers), not preemptive inside a GPU operation or disk read.

SIGINT/SIGTERM stops admission, cancels queued/active requests, and shuts down
the listener. A fatal backend exception stops the service rather than continuing
with potentially corrupted state. Full-graph request failures never authorize a
silent layered fallback.

## Validation

The standalone protocol tests do not need weights or CUDA:

```bash
cmake -S server -B build-api -DCMAKE_BUILD_TYPE=Release
cmake --build build-api -j4
ctest --test-dir build-api --output-on-failure
```

`server_tests` covers request validation, auth, tool/result round trips,
fragmented UTF-8 and tool output, SSE, stop sequences, queue limits, timeout and
disconnect cancellation, shutdown, and fatal backend failures.
`cpu_api_tests` uses real CPU Qwen recurrent/convolution and dense-attention
operations with tiny synthetic F32 weights to compare cached and uncached
logits bitwise over successive forced decode frames, interleaved conversations,
and eviction. It does not validate real quantized MoE weights.

Before enabling caching by default, validate a real GGUF using teacher-forced,
per-frame comparisons against the reference, including repeated prompts,
appended tool turns, interleaved sessions, cancellations, and eviction. Follow
`scripts/validate_vs_llama.sh` where applicable. CUDA full-fit runs must also
pass `tools/check_full_graph_moe_run.py <run.log> --full-fit`.

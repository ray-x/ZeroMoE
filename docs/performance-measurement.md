# Measuring performance

Three layers, each using the one below it:

1. **The per-request stats record**: the engine measures every request it serves.
2. **`zeromoe --bench`**: runs a fixed set of requests in one process and summarises their records.
3. **`bench/perf_report.py`**: runs `--bench` on a quiet machine, adds the machine and build description, and renders tables.

## 1. The per-request stats record

`core/request_stats.hpp`. Every request produces one record when it ends (also when it fails: `status` is then `error`).
It is written to the log as a single `[STATS] {json}` line, appended to a file as one JSON object per line when you pass
`--stats-file PATH`, and kept in memory for in-process callers.

| Field | Meaning |
| --- | --- |
| `status`, `seq`, `model`, `family` | `ok` or `error`; request number in this process; model file name; graph family |
| `prompt_tokens`, `prompt_cached_tokens` | prompt size, and how much of it came from the prompt-state cache |
| `prefill_ms`, `prefill_tok_s` | time to process the uncached prompt tokens, and the rate |
| `decode_setup_ms` | decode-graph setup this request paid before its first token (Qwen graph; `null` for families that do not separate it) |
| `ttft_ms` | request start to first generated token |
| `generated_tokens` | tokens produced |
| `decode_tok_s` | steady decode speed: 1000 / mean(step ms) over steps >= 3 |
| `decode_peak_tok_s` | best 32-step window |
| `decode_p50_ms`, `decode_p95_ms` | step-time percentiles over steps >= 3 (a high p95 against the median means stalls) |
| `decode_bucket_tok_s` | speed in six equal consecutive parts of a long generation (only with >= 120 steps) |
| `decode_wall_ms`, `decode_wall_tok_s` | end of prefill to end of request, setup included |
| `total_ms` | whole request |
| `start_unix_ms`, `decode_start_unix_ms`, `end_unix_ms` | wall-clock instants, so an outside observer can match machine noise to the request |
| `extra` | engine-specific numbers noted during the request, for example `soft_released_gib` (Qwen: how much of the soft expert tier the prefill released) |

A value that does not apply is `null`, never 0: the decode figures of a 2-token request, the prefill rate of a 1-token prompt.

Engine code adds to `extra` with `zeromoe::stats::note("key", value)`.

## 2. `zeromoe --bench`

```sh
zeromoe --config configs/qwen38-flash.toml --bench \
    --bench-prompt bench/prompts/short.txt --bench-prompt bench/prompts/mid.txt --bench-prompt bench/prompts/long.txt \
    --bench-ngen 200 --bench-reps 3 --bench-warmup 1 --bench-long 1500 --bench-out result.json
```

It loads the model once and gives the request loop a scripted queue instead of stdin, so chat templates, tokenization,
prefill, decode-graph setup and teardown are exactly what a user's request runs. It forces full-length generation, switches
off per-token printing and VRAM sampling (as llama-bench does) and the prompt-state cache (a repeated prompt must be prefilled
every time). The output is a table on stdout plus `--bench-out` JSON: min, median and max per prompt, and every request's record.

**Random tokens** (llama-bench's tests): `--bench-random-pp N` processes a prompt of N random token ids;
`--bench-random-tg N` starts from one random token and generates N tokens, feeding a random token at each step instead of the
sampled one. Ids come from a seeded generator (`--bench-seed`, default 1). `--bench-random-pp 512 --bench-random-tg 128` is
the `llama-bench -p 512 -n 128` equivalent. Random ids route to random experts, so a MoE model whose hot-expert placement was
tuned on real text is slower on them (Qwen 3.8 Flash: about 41% lower decode). Report both.

Real text is the default because a MoE model's speed depends on which experts the text routes to.

## 3. `bench/perf_report.py`

```sh
bench/perf_report.py run --name qwen38-flash --config configs/qwen38-flash.toml --model /path/to/model \
    --binary build/zeromoe --env CUDA_VISIBLE_DEVICES=0 --env ZEROMOE_QWEN4EXP_FULL_GRAPH=1 ... \
    --set omp_max_threads=10 --note ram_type="DDR4-2666 4ch"
bench/perf_report.py render bench/results/*.json        # Markdown for README / PERFORMANCE.md
```

It runs `zeromoe --bench` (text prompts, then random tokens), reads the engine's records, and adds what only an outside
observer can: the CPU with AVX-512 flags, RAM, GPUs (PCIe link, power limit, which were used), OS, CUDA, ZeroMoE and ggml
commits (and whether the binary matches HEAD), model files, the config with its checksum, every override and environment
variable, and the engine's resolved-settings log lines. It also decides whether the machine was quiet: every request is
checked against outside CPU load (everything except the engine) during its prefill and its decode, and against another
`zeromoe` or a compiler running. A request counts only if it is valid and quiet (at most 0.5 cores outside on average, 1.0 in
any second); the rest are kept under rejected and replaced by another run, up to three attempts per mode. The command can be
rerun to continue where it stopped. Results go to `bench/results/<name>.json`; the working files stay out of git.

To add a model to the summary: run the same command with its config, `--model` and environment. No per-model code is needed.

## How the fields compare with other servers

| ZeroMoE | llama.cpp server `timings` | Ollama | LM Studio (`stats`) |
| --- | --- | --- | --- |
| `prompt_tokens` - `prompt_cached_tokens` | `prompt_n` (`cache_n` is the cached part) | `prompt_eval_count` (`prompt_eval_cached_count`) | `input_tokens` |
| `prefill_ms`, `prefill_tok_s` | `prompt_ms`, `prompt_per_second` | `prompt_eval_duration` (ns) | not given |
| `generated_tokens` | `predicted_n` | `eval_count` | `total_output_tokens` |
| `decode_tok_s` | `predicted_per_second` | `eval_count / eval_duration * 1e9` | `tokens_per_second` |
| `ttft_ms` | not given | not given | `time_to_first_token_seconds` |
| `decode_setup_ms` | not given | not given | not given |
| `total_ms` | | `total_duration` (ns) | |

The definitions differ in detail: llama.cpp and Ollama divide by the whole generation time, while `decode_tok_s` leaves out
the first three steps. Compare orders of magnitude and trends across engines, not the last digit. llama.cpp also publishes
Prometheus counters at `/metrics` (`--metrics`) and per-slot state at `/slots`; ZeroMoE has no server yet, so the JSONL file is
the interface.

## Pitfalls

- **The first request of a process is slower** (graph and kernel set-up, cold caches). `--bench` runs a warmup request and
  reports a warm process, which is what a running server sees; a fresh-process-per-request measurement reads prefill 3 to 15%
  lower.
- **Noise.** A compile or another model server on the same machine changes decode by 10% or more, and a single stall can
  turn one request into 4 tok/s. The record keeps p95 and the machine screen exists for this reason; do not publish numbers from
  a busy machine.
- **Greedy decoding on a fixed prompt** produces the same text each time, so differences between parts of a long decode reflect
  what is generated as well as context length.
- **Set the model path** with `--model` or a real `model_path` in TOML; the shipped configs carry a placeholder.

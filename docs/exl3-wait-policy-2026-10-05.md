# Native EXL3 decoder waiting-policy comparison

This report covers ZeroMoE's native `exl3_gpu=true` decoder, not the
delegated upstream ExLlamaV3 model/cache/generator backend documented in
[`exllamav3-backend.md`](exllamav3-backend.md).

The 26 CPU / 22 GPU EXL3 profile remains below the 28 tok/s target. On the
Xeon W-2235, RTX 3090 and RTX 5060 Ti host, passive OpenMP waiting improved
the warmed standard profile from about 6.40 to 7.10 tok/s. HC fusion was
disabled in all runs.

The binary, checkpoint, prompt, 12 CPU threads, context 256, temperature zero,
CPU int8 VNNI path and expert placement were identical. CUDA scheduling was
explicitly `auto`; `GOMP_SPINCOUNT` was unset. Each run emitted 128 tokens,
and throughput covers the 127 first-to-last generated-token intervals.

| Run, in execution order | OpenMP waiting | Decode seconds | tok/s | Raw log |
| --- | --- | ---: | ---: | --- |
| Cold filesystem cache | Passive | 27.0130 | 4.70143 | [log](../logs/exl3_wait_policy/passive_cold.log) |
| Warm default | Default | 19.8895 | 6.38528 | [log](../logs/exl3_wait_policy/default_warm.log) |
| Warm passive | Passive | 17.8913 | 7.09842 | [log](../logs/exl3_wait_policy/passive_warm.log) |
| Default repeat | Default | 19.8004 | 6.41402 | [log](../logs/exl3_wait_policy/default_repeat.log) |

The cold run is excluded from the policy comparison. The two default runs
bracket the warm passive run and average about 6.40 tok/s. Passive waiting
increased throughput by approximately 10.9%. All 128 token IDs and the
printed selected-token logits match across the three warm runs.

| Wall-time category per decode interval | Default mean, ms | Passive, ms |
| --- | ---: | ---: |
| CPU expert matmul | 37.783 | 39.649 |
| CPU weight lookup/loading | 12.789 | 8.175 |
| GPU executor calls | 43.676 | 42.589 |
| GPU weight upload | 8.526 | 8.265 |
| Remaining forward work | 53.485 | 42.198 |

These are host-wall counters, not isolated CUDA-event kernel durations.
The CPU weight category includes cache lookup and dispatch overhead. The
remaining category also includes GPU expert payload reads and host packing.
Passive waiting saves host work despite slightly slower CPU matmuls.

Use this launch environment for the measured EXL3 hybrid profile:

```bash
env -u GOMP_SPINCOUNT OMP_WAIT_POLICY=PASSIVE ZEROMOE_CUDA_SCHEDULE=auto \
  ./build_ggmlcuda/zeromoe --config configs/qwen4exp_exl3_gpu.toml \
  --prompt "Write a detailed, continuous explanation of how a small garden changes through the seasons. Keep going with practical details and examples, and do not conclude early." \
  --ngen 128
```

This recommendation is specific to the EXL3 hybrid decoder. The GGUF full
graph measurements in [runtime-settings.md](runtime-settings.md) retain their
own waiting-policy guidance. The source-cache lookup optimization added after
this comparison needs a separate throughput check.

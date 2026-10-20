# GLM-5.3 decode baseline (3090 + 5060 Ti, SSD-offloaded experts)

Full results, hardware, memory use and method: `docs/GLM5.3-baseline.md`.

Files:
- `glm53_dsa_core5060_retarget.toml` -- the baseline config (5060 Ti core + 3090 expert vault, `host_ram_gb = 100`).
- `run_dsa_graph2gpu.sh` -- launches one request (env recipe, strict full graph, O_DIRECT, offload store).
- `run_ab.sh <tag> [KEY=VALUE ...]` -- one benchmark run with the pilot defaults; the log is copied to `logs/split_ab_<tag>.{log,stdout}`. Overrides apply after the defaults.
- `preflight.sh` -- refuses to start when another engine or GPU user is running, and drops the page cache of the model files.

Peak recipe (short prompt, 64 GiB DDR pool):

    systemd-run --user --scope -p MemorySwapMax=0 bench/glm53/run_ab.sh peak \
      MOESTREAM_GLM5NEXT_GPU1_HOT_GIB=22 OFFLOAD_DDR_GIB=64 NGEN=80 \
      MOESTREAM_GLM5NEXT_PILOT_PREFETCH=3 MOESTREAM_GLM5NEXT_PILOT_STAGE=3 \
      MOESTREAM_GLM5NEXT_MODEL_DIRECT=1 # MOESTREAM_OFFLOAD_RETARGET is on by default (=0 disables)

Long prompt (426 tokens): `PROMPT_TEXT=<prompt> OFFLOAD_DDR_GIB=74 MOESTREAM_MEM_GUARD_FLOOR_MB=16384` plus the recipe above.
Needs MOESTREAM_BIN (default `./build/moestream`) built Release with the ggml tree that carries patches 0006/0007, and the paths in the
toml / scripts (model dir, offload store `/data/glm53_offload.gguf`, usage files) adjusted for your machine.

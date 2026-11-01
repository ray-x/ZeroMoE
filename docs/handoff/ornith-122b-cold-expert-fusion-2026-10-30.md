# Ornith 1.5 122B tuning handoff: cold expert fusion

## Goal and current status

Continue recovery of Ornith 1.5 122B support in the unmerged
`recovery/ornith-122b-offload` worktree. This tuning change targets the
repeated CPU work for routed experts that miss the usage-ranked VRAM hotset.
It does not yet have a GPU throughput measurement.

CUDA access has been intermittent. At 18:06–18:13, a stable three-probe
window allowed a DEBUG run on an idle RTX 3090; its full-graph gate passed
and it captured fused-node timings. Host load then became severe, so the
DEBUG rate is not valid performance evidence. At 18:15 the matching
`llama-bench --list-devices` again reported no CUDA device; no uninstrumented
post-fusion throughput run has completed. `/mnt/usb-ssd` remains readable.

## Finding

The shared `qwen4exp_cold::supported()` path in
`models/qwen/cold_ffn_fused.cpp` previously accepted only Q2_0 expert weights
using Q8_0 activation rows. That let the Qwen3.8 Q2_0 profile fuse each cold
expert's gate/up, SwiGLU, and down projection into one custom CPU node with an
internal barrier. Ornith's IQ3_XXS/IQ2_S/IQ4_XS weights were rejected and used
three separate `MUL_MAT_ID` nodes plus mask/route weighting in the graph.

This is a material code-path difference for Ornith's 22.77% surveyed cold
route share at the 16.5 GiB profile. Existing decode logs did not measure
the cost of those CPU nodes by layer, so the size of the speedup remains
unproven.

## Change made

The fused CPU node now reads `ggml_get_type_traits_cpu()` for gate, up, and
down separately. It accepts tensors when all three CPU dot kernels use the
same activation quantization type and their dimensions meet each weight and
activation block alignment. This covers the Ornith combinations:

- IQ3_XXS / IQ3_XXS / IQ3_XXS, using Q8_K activations.
- IQ2_S / IQ2_S / IQ3_XXS, using Q8_K activations.
- IQ3_XXS / IQ3_XXS / IQ4_XS, using Q8_K activations.

The phase-one activation chunk follows the activation block size (256 rows
for Q8_K). Dot calls use each kernel's row contract. IQ3_XXS requires
`nrc=1`; the Q2_0 VNNI fast path retains its patched four-row call when
AVX-512 VNNI is available. Kernels with incompatible activation types or
unsupported alignment continue to use the stock graph chain.

A DEBUG log marker, `[GGML-QWEN-FULL-COLD-FUSED]`, reports the layer,
weight types, activation type, and CPU thread count when this path is selected.
The next GPU run can verify path coverage directly.

## Correctness verification

Built and ran the CPU-only `cold_ffn_fused_test` target from:

```sh
cmake --build build_mtp_results/2026-10-29-ornith-122b-smoke-5060ti/build \
  --target cold_ffn_fused_test -j4
stdbuf -oL \
  build_mtp_results/2026-10-29-ornith-122b-smoke-5060ti/build/cold_ffn_fused_test
```

Result: exit code 0; all 180 fused-vs-stock cases were bit-identical:

| Type combination | Cases | Result |
| --- | ---: | --- |
| Q2_0 / Q2_0 / Q2_0 | 60 | PASS |
| IQ3_XXS / IQ3_XXS / IQ3_XXS | 60 | PASS |
| IQ2_S / IQ3_XXS / IQ4_XS | 60 | PASS |

Each type combination covers 1, 2, 3, 6, and 10 CPU threads; S=1, 2, and 4;
and all-hot, sparse-cold, all-cold, and mixed route masks. The initial IQ3
attempt used four-row calls and failed because the GGML IQ3_XXS kernel
requires one row. The implementation was corrected to honor that contract;
the final expanded matrix passes.

The main `zeromoe` target then built after reconfiguring the existing CMake
tree to capture the current dirty worktree snapshot. Provenance check passed:

- Project commit: `7d48e01de3ea2804ea6375b37858c23c909ad1d2` (dirty worktree).
- Source snapshot SHA256: `c5672e44ab40695b672bcaf14c89a90f1ca6d87a275476df292bbb157d235888`.
- Binary SHA256: `65947859c774df82695c172199b7b981db009ddb47f1a8a8651cd8ecbbcf8a67`.
- GGML base: `8baabd53b77d1c48c6ef8d3b620436292f3f6231`.
- Applied GGML patch manifest SHA256:
  `bc15a89242db58386ab4335048ea55bb15233f18c40fe23881802097fe1f4f71`.
- CUDA targets: sm_86; sm_120a.

Build output is saved at
`build_mtp_results/2026-10-30-ornith-122b-3090-match-24p200g10t/official-survey-5/cold-fused-zeromoe-build.log`.

The Ornith GGUF log lists 13 unique gate/up/down quantization triplets across
its 48 layers. The test source now includes all 12 combinations beyond the
already-tested IQ3_XXS triple (covering IQ2_XXS, IQ1_S, IQ2_S, IQ3_S, Q2_K,
Q3_K, and IQ4_XS); the updated target compiles, but those 72 cases have not
run. Two additional short cases use Ornith's actual D=3072, FF=1024 shape.
The latest shared-host load was 15.76 on 12 allowed CPUs, so only the original
180 cases are currently verified. Run the expanded CPU matrix when the shared
host is quieter.

Static comparison of the saved quant log against the test cases found all 13
model triplets present. The Q2_0 control and the IQ2_S/IQ3_XXS/IQ4_XS mixed
case are extra coverage. This verifies the test definition, not the pending
case results.

## Existing performance baseline

The last accepted official-histogram hotset measurements predate this fusion
change: 16.25 GiB measured 21.33 tok/s and 16.5 GiB measured 20.99 tok/s on
the RTX 3090. Both passed the full-graph and 90%-of-llama.cpp gates. They do
not show whether IQ3_XXS cold fusion helps. Qwen3.8's Q2_0 profile reported
about 42–43 tok/s with 14,554 expert records / 18.74 GiB resident; the Ornith
profile holds 5,375 / 16.498 GiB and reports 77.23% route coverage. The model
and workload comparison, including the larger active matrix work in Ornith,
is documented in `docs/models/ornith-1.5-122b.md` and the detailed audit at
`build_mtp_results/2026-10-29-ornith-122b-perf-5060ti/ornith-qwen38-timing-and-profile-audit.md`.

Do not quote a speedup for this code change until it is measured on GPU.

The existing matched hotset logs were gated against the user's absolute
42 tok/s hotset target with `--minimum-tps 42`; both fail at 21.33 and
20.99 tok/s. Outputs are saved in `official-survey-5/` as
`official-hotset16_25-minimum42-gate.txt` and
`official-hotset16_5-minimum42-gate.txt`. Their relative llama.cpp gates still
pass, but the user target is not met.

## Release readiness blockers

The adapter is recovered and its previous full-fit run passed the graph and
relative llama.cpp checks, but this worktree is not release-ready:

- The user-set 42 tok/s hotset target fails on both saved profiles. The 16.25
  GiB run measured 21.33 tok/s; the 16.5 GiB run measured 20.99 tok/s. The
  `--minimum-tps 42` failures are saved as
  `official-survey-5/official-hotset16_25-minimum42-gate.txt` and
  `official-survey-5/official-hotset16_5-minimum42-gate.txt`.
- The user-set 28 tok/s split target also remains unmet: the corrected
  19-GPU-layer / 29-CPU-layer profile measured 18.26–18.62 tok/s. Its
  16-token / 400-token workload still lacks a matched same-GGUF llama.cpp row.
- The new mixed-IQ cold-expert fusion has CPU equivalence evidence only. It
  has not been run in the model graph on a GPU, so its performance effect is
  unknown.
- The original 180-case fused-vs-stock matrix passed, and the low-priority
  one-thread smoke passed the 12 remaining model type combinations plus two
  actual-shape cases. The remaining 70 concurrency/mask cases have not run.
- CUDA enumeration is intermittent: it succeeded briefly, then failed
  again. During the successful probe another worktree held the RTX 3090,
  and the host load was high. Do not compete with that process; see the
  latest device probes below.
- The recovery worktree is dirty and unmerged. No release commit was made.

## Next step

The CPU matrix is still incomplete. Rerun its focused expanded mode when
the shared host is quieter. For GPU validation, first require stable CUDA
enumeration and an idle 3090; a brief successful probe is insufficient.

1. Validate the new binary with `tools/build_provenance.py check`.
2. The exact DEBUG profile is prepared at
   `build_mtp_results/2026-10-30-ornith-122b-3090-match-24p200g10t/official-survey-5/official-hotset16_5-cold-fused-debug.toml`;
   its command and build manifest sit beside it. It preserves the accepted
   16.5 GiB workload and sets `ZEROMOE_QWEN_EARLY_READY_ALL_SPLIT=0`. Run it
   for CPU per-layer/op timing and fused-node coverage, not as a speed result.
3. Confirm the full graph actually uses the fused cold node for mixed IQ
   layers; check that no cold-route weights or node outputs are missing.
4. Run an uninstrumented three-request 24-prompt / 200-generation test on the
   same binary and configuration, then rerun the full-graph and performance
   gates against the clean llama.cpp reference.
5. Compare per-token cold CPU time, graph synchronization, and steady decode
   to the pre-change 20.99 tok/s hotset record. Keep onboarding in progress
   until the measurements show the effect and the Qwen3.8 speed gap is
   addressed.

## Latest expanded CPU smoke (2026-10-30)

This update supersedes the earlier note that all expanded cases were still
pending.

- The saved model quant inventory has 13 unique gate/up/down triplets. The
  test matrix contains all 13, a Q2_0 control, one extra mixed-format control,
  and two actual-shape cases (`D=3072`, `FF=1024`, E=8, top-8).
- The fixture now uses `ggml_quantize_chunk` with a unit importance matrix for
  IQ types whose `from_float_ref` is unavailable (IQ2_XXS and IQ1_S).
- Ran the low-priority `--smoke-ornith-types` mode at one CPU thread,
  S=1, all-cold routing. All 14 pending type/shape combinations passed
  bit-identically, including actual-shape cases. Saved output:
  `build_mtp_results/2026-10-30-ornith-122b-3090-match-24p200g10t/official-survey-5/cold-fused-type-smoke-rerun.log`.
- The full multi-thread/token/mask matrix still has 70 cases pending; only the
  original 180-case matrix and this one-thread smoke are verified. Current
  shared-host load is high, so don't run the remaining matrix until it falls.
- CUDA remains unavailable (`nvidia-smi` cannot reach the driver). The DEBUG
  timing profile is prepared but unrun; there is still no GPU speed result for
  cold-expert fusion.

## Full matrix rerun attempt (2026-10-30)

The expanded full CPU matrix was started at low priority after the one-minute
load showed 1.86. The Q2_0 control completed 60/60 bit-identical cases, but
host load rose to 14.63 on 12 allowed CPUs. I stopped the run with SIGINT to
avoid further contention; exit code 130. This is an interrupted run, not an
operation failure. The earlier one-thread all-cold smoke passed all 14 pending
quant/shape cases. Artifacts: `official-survey-5/cold-fused-full-matrix-rerun.command.txt`,
`cold-fused-full-matrix-rerun.log`, and `cold-fused-full-matrix-rerun.exit-code.txt`.
The remaining multi-thread/token/mask cases still need a quieter host.

## Earlier host CUDA failure probe (2026-10-30)

Read-only checks show the NVIDIA kernel module is loaded at 610.57.04 and
`/proc/driver/nvidia/gpus` still lists the RTX 3090 (`0000:4f:00.0`) and RTX
5060 Ti (`0000:bd:00.0`). Their `/dev/nvidia0`, `/dev/nvidia1`, `/dev/nvidiactl`,
UVM, and capability nodes exist, and `CUDA_VISIBLE_DEVICES` is unset. However,
`nvidia-smi` fails to communicate with the driver and the clean matching
`llama-bench --list-devices` reports no CUDA-capable device. The boot-kernel
journal search showed module initialization but no XID entry. This points to a
host CUDA/driver state failure; it is not a model or ZeroMoE graph error. The
host is shared, so no driver reload, device reset, or node recreation was
attempted. GPU acceptance requires the host CUDA state to recover.

## Focused expanded-matrix progress (2026-10-30)

To avoid rerunning the 180 established cases, `tests/test_cold_ffn_fused.cpp`
now accepts `--ornith-expanded-matrix`, which targets the 14 additional
Ornith quant/shape combinations (84 thread/mask cases). The test target builds
with this mode.

The first focused run completed six combinations before interruption:

- Both actual-shape cases (`D=3072`, `FF=1024`) passed all six cases bit-for-bit.
- `iq2_s/iq2_s/iq3_xxs`, `iq3_xxs/iq3_xxs/iq4_xs`, and
  `iq2_xxs/iq2_xxs/q2_k` each passed all six cases bit-for-bit.
- `iq2_xxs/iq2_xxs/iq3_xxs` also had bit-identical fused and stock outputs in
  all six cases, but the fixture's sparse-cold coverage assertion failed in
  one case: randomized route IDs happened to select no cold expert. This was a
  test-fixture false positive, not an output mismatch.

The route fixture now assigns deterministic IDs for the sparse-cold pattern so
that each case includes both cold and hot experts. The updated test target
builds. Its full focused rerun was interrupted when shared-host load rose to
14.92 on 12 allowed CPUs, before it emitted a case result. Do not count that
rerun as verification. The focused rerun, earlier partial output, command
lines, and exit codes are preserved under
`build_mtp_results/2026-10-30-ornith-122b-3090-match-24p200g10t/official-survey-5/`
with `cold-fused-ornith-expanded-matrix*` filenames. Re-run the focused matrix
when host load is lower; CUDA validation remains blocked by the driver state
recorded above.

## Transient device and provenance probe (2026-10-30 14:39)

A read-only `nvidia-smi` probe now succeeds with driver 610.57.04. It
reported 12,742 MiB in use on the RTX 3090 by PID 997405, running
`ZeroMoE-wt-rank/build/zeromoe`; GPU utilization was 0%. The RTX 5060 Ti
had 4 MiB in use. This is another worktree/session, so do not terminate it
or start a competing 3090 workload. Host load was 17.38 on the 12-CPU
affinity set.

The prepared engine binary passed
`python3 tools/build_provenance.py check --binary
build_mtp_results/2026-10-29-ornith-122b-smoke-5060ti/build/zeromoe --repo .`
for dirty source commit `7d48e01de3ea2804ea6375b37858c23c909ad1d2`, binary
SHA256 `65947859c774df82695c172199b7b981db009ddb47f1a8a8651cd8ecbbcf8a67`,
and GGML base `8baabd53b77d1c48c6ef8d3b620436292f3f6231`.

## Latest CUDA recheck (2026-10-30 14:40:47)

`nvidia-smi` failed again at 14:40:47 AEDT, after succeeding at 14:39.
The NVIDIA kernel module still reports version 610.57.04, `/proc/driver/nvidia/gpus`
still lists both PCI devices, and `/dev/nvidia0`, `/dev/nvidia1`, and
`/dev/nvidiactl` still exist. Host load was 13.59 on the 12-CPU affinity
set. CUDA availability is unstable; do not start GPU validation until
enumeration remains stable and the 3090 has been released by the other
worktree process.

## Per-layer cold-fusion timing and rebuilt binary (2026-10-30)

The DEBUG profile enables the existing scheduler CPU-node timing callback.
The fused cold node previously had no `layer_timing_layer_of` tag, so its
per-layer time could be grouped under a generic node name. The graph builder
now tags the node as `moe_cold_fused` for that callback. This should emit
per-layer and per-op cold-node time alongside the existing fused-layer
selection marker; this instrumentation has not yet been exercised in a
model run.

The first incremental link attempt rejected its stale configure-time source
snapshot and deleted the old executable. I reconfigured the same CMake tree
from this worktree and rebuilt `zeromoe` successfully. The new binary SHA256
is `2fa0691c01d9c78741f5605f7440bc39e244cd10a8800768a772ad4cbb7bf7c4`; its
source snapshot SHA256 is `1e45140d9db77b6cbce65cbd75f1fd94411ed7e185de7fbdaf5436f2ed719c84`.
`tools/build_provenance.py check` passes for source commit
`7d48e01de3ea2804ea6375b37858c23c909ad1d2`, dirty worktree, and GGML base
`8baabd53b77d1c48c6ef8d3b620436292f3f6231`. The prepared DEBUG profile's
adjacent build manifest was refreshed from this build.

## Latest CUDA recheck (2026-10-30 14:44:19 AEDT)

`nvidia-smi` failed again; host load was 13.98 on the 12-CPU affinity set.
The driver state remains intermittent. Do not run the GPU profile until CUDA
enumeration is stable and the 3090 has been released by the other session.

## Per-case expanded matrix update (2026-10-30 14:57)

`cold_ffn_fused_test --ornith-expanded-matrix <case-name>` now selects one
expanded tuple, so interrupted shared-host runs can resume without repeating
completed tuples. The target builds. After the deterministic sparse-cold
fixture fix, all 12 additional quant tuples pass their six thread/mask cases;
the IQ3_XXS actual Ornith shape also passes all six. Per-case logs are saved in
`official-survey-5/cold-fused-*-case.log` and `*-det.log`.

The mixed-IQ actual-shape case is the one remaining verification. Its earlier
pre-fix run passed, but the final-fixture rerun was interrupted when host load
reached 16.50. I then changed the test harness to quantize each tuple's weight
bytes once and reuse them across the six thread/mask cases; the test target
builds, but no test has run after that optimization. Rerun the mixed actual
shape after host load falls, then rerun all cases if the shared-weight change
reveals any discrepancy.

At 14:58 AEDT, host load was 12.98 on the 12-CPU affinity set and
`nvidia-smi` still failed to communicate with the driver. No CPU matrix or GPU
profile was started under that condition.

## Cached-weight fixture recheck (2026-10-30 15:00)

The current test harness, which quantizes each tuple's weights once and reuses
the packed bytes, passes the `iq2_xxs/iq2_xxs/iq3_xxs` tuple across all six
thread/mask cases. A bounded low-priority run then passed four more small
quant tuples (exit code 0 for each); the per-case exit manifest is
`official-survey-5/cold-fused-expanded-cached-fixture-results.tsv`. The runner
stopped before starting the next tuple when the load threshold was reached.
Thus five of 12 non-actual-shape tuples have current-harness results; seven
small tuples and both actual-shape tuples still need current-harness runs.

At 15:00 AEDT, load was 13.38 on the 12-CPU affinity set and `nvidia-smi`
still failed. No further CPU matrix or GPU run was started.

## User-designated Qwen split profile comparison (2026-10-30)

Compared Qwen's measured split profile (`configs/bench/qwen4exp_tg400_iq1m_22_26.toml`)
with Ornith's corrected 16-prompt / 400-generation candidate
(`ornith-qwen-map-split40-thread80.toml`). The comparison applies the Qwen
run's `graph_device_expert_layers=40`, trailing placement, and 10-thread
runtime overrides, plus the candidate's observed 10 process/graph threads.
The CPU-affinity map, context 416, output length 400, 10 effective threads, and
40-layer trailing placement request match. Before memory admission both
profiles choose the same 22 eligible layers; Ornith's larger expert matrices
make the 3090 planner admit 19 of them, leaving 29 CPU/DDR layers. The measured
18.26–18.62 tok/s Ornith split rate remains below Qwen's 28.66 tok/s. The
active matrix-work estimate (1.54x) predicts 18.61 tok/s, which is consistent
with those runs but does not prove all overhead is explained.

The review also flags `host_ram_gb` 100 vs 80, `expert_fetch_threads` 10 vs 4,
prefill batch 8 vs unspecified, model-specific pin files, and fixed GPU 0 vs
auto-selection. `ram_preload_all_experts=true` in both profiles; the measured
candidate selected the RTX 3090 through auto-selection. These differences need
separate consideration; the config comparator does not establish their decode
impact. Qwen usage/prefill pin files are model-specific and must not be copied
to Ornith. The output and exact command are saved as
`official-survey-5/qwen38-split-profile-compare.txt` and
`qwen38-split-profile-compare.command.txt`.

## Cached-fixture case continuation (2026-10-30 15:07)

The guarded low-priority continuation passed the remaining seven small
quant tuples under the cached-weight fixture. Together with the earlier five,
all 12 additional small-shape tuples now pass six thread/mask cases each on
the current harness (72 cases). Their exit codes and log names are in
`official-survey-5/cold-fused-expanded-cached-fixture-results.tsv` and the
adjacent `*-cached-fixture.log` files.

Both actual-shape tuples still need current-harness runs after the packed
weight reuse change. Earlier runs provide evidence for the unchanged fused
node with the former fixture; they do not finish this current test matrix.

The profile comparator now accepts `--candidate-override` for effective
settings resolved from startup logs or hardware policy. A manual Qwen split
comparison with effective 10-thread values confirms the matching affinity
map, 40-layer trailing placement request, context 416, and 400-token
workload; the planner admits 19 Ornith GPU layers versus Qwen's 22. The
comparison artifact is saved under `official-survey-5/`.

At 15:07 AEDT, host load had risen to 15.32 on the 12-CPU affinity set;
the last `nvidia-smi` probe at 15:06:43 still failed. No further CPU or GPU
run was started.

## Latest shared-host probe (2026-10-30 15:10 AEDT)

Host load is 17.47 on the 12-CPU affinity set. `nvidia-smi` still cannot
communicate with the driver. Read-only checks show NVIDIA kernel module
610.57.04 loaded, both PCI GPU directories under `/proc/driver/nvidia/gpus`,
and `/dev/nvidia0`, `/dev/nvidia1`, and `/dev/nvidiactl` present. No NVRM/Xid
lines were visible in the accessible kernel log. No driver or device action was
taken; the two actual-shape cached-fixture tests and all GPU validation remain
pending.

## Expanded cold-fusion CPU matrix complete (2026-10-30 16:07)

The current cached-weight test harness passed all 14 expanded tuples across
six thread/mask cases each: **84/84 bit-identical** fused-versus-stock cases.
Coverage includes all 12 additional model quant tuples plus both actual
Ornith-sized shapes (`D=3072`, `FF=1024`, 8 experts, top-8). Every per-case
process exited 0. The verified test binary SHA256 is
`30031e850926ecaa8e6c69394a0a6f8cb9837039f8b903323fb1b1dc280889cd`.
The auditable case list is
`official-survey-5/cold-fused-ornith-expanded-84case-summary.txt`; individual
logs and exit results are beside it.

This completes the CPU fused-node equivalence matrix. It does not establish
that the GPU full graph selects/runs the node in the target environment or
prove a throughput improvement. At 16:07, CUDA was still unavailable
(`nvidia-smi` could not communicate with the driver); host load was 6.69 on
the 12-CPU affinity set. Next: with stable CUDA and an idle 3090, run the
prepared DEBUG profile, confirm per-layer `moe_cold_fused` timing/coverage,
then run repeated uninstrumented split and hotset measurements with the 28 and
42 tok/s gates.

## Split and hotset GPU verification packages prepared (2026-10-30 16:12)

Prepared a DEBUG split profile cloned from the corrected Qwen-map candidate,
with `log_level=DEBUG`, the exact 16-token `Tell a story.` prompt, context 416,
400 generated tokens, and `ZEROMOE_QWEN_EARLY_READY_ALL_SPLIT=0` to exercise the
fused cold node. Three separate INFO repeat configs/commands use the same
workload and environment. Their startup must confirm auto-selection of the
3090, ten graph CPU threads, and fused-node coverage; DEBUG results are timing
evidence only.

Also prepared three INFO reruns of the usage-ranked 16.5 GiB hotset profile.
The gate command files run the full-graph check and then `--minimum-tps 28`
for split or `--minimum-tps 42` for hotset. A matched llama.cpp 16/400 command
is prepared for the same GGUF, CUDA0/3090, ten threads, batch/ubatch 8, and
`-ngl 99 -ncmoe 29`; it must produce the exact JSON row before the split
performance gate is valid. The hotset gate uses the saved matched 24/200,
context-224, ten-thread llama-bench JSON with `-ncmoe 31`.

Configs, commands, prompt and build manifests are saved in
`build_mtp_results/2026-10-30-ornith-122b-3090-match-24p200g10t/official-survey-5/`
under `official-split-qwen-map-cold-fused-*` and
`official-hotset16_5-cold-fused-r*`. TOML parsing and current ZeroMoE binary
provenance check pass. Nothing was launched: at 16:12 AEDT load was 2.41, but
`nvidia-smi` still failed to communicate with the driver.

## Latest status check (2026-10-30 16:22 AEDT)

The 84/84 current-harness CPU correctness matrix and split/hotset GPU run
packages are complete. Host load is 13.99 on the 12-CPU affinity set, and
`nvidia-smi -L` still fails to communicate with the driver. The clean matching
`llama-bench --list-devices` probe reports no CUDA-capable device. No GPU run was
started; the 3090's current ownership/availability cannot be verified while
CUDA enumeration is down. The target throughput gates remain outstanding.

## Retry probe (2026-10-30 18:02 AEDT)

After the user resumed the goal, I rechecked the same recovery worktree and
run artifacts. The current-harness CPU matrix remains recorded as 84/84 PASS.
Host load is now 1.10 on the 12-CPU affinity set, but the CUDA gate is still
closed: `nvidia-smi` fails, the matching `llama-bench --list-devices` reports
`(none)`, and `/dev/nvidia0`, `/dev/nvidia1`, and `/dev/nvidiactl` are absent.
`/proc/driver/nvidia/gpus` still lists both PCI GPU entries. No GPU process was
started and no device/driver changes were attempted. The remaining work needs
CUDA device access: run the prepared split DEBUG profile, then the matched
split/hotset repeats and 28/42 tok/s gates.

## Hotset DEBUG graph execution (2026-10-30 18:06–18:13 AEDT)

After three stable CUDA preflight probes showed the idle RTX 3090 with
23,858 MiB free, the prepared DEBUG profile ran three repeated 24-prompt /
200-generation requests. Startup auto-selected the 3090; the memory plan was
`Fits` at 23.21 / 23.28 GiB with the 16.50 GiB usage-ranked expert budget and
complete preloaded host experts. The debug marker selected the IQ-quantized
`moe_cold_fused` node in all 48 layers on each of three graph builds (144
markers total). `check_full_graph_moe_run.py --full-fit
--require-full-graph-prefill` passes.

The new layer timing tag emitted 57 `moe_cold_fused` samples: 21.5–533.0 ms,
median 413.8 ms. Early samples were tens of milliseconds; later ones were
around 400–530 ms while sampled host load rose above 20 on 12 CPUs. The debug
run reports 2.61 decode tok/s / 2.41 overall for 600 tokens, but this is
instrumented and load-contended; it is not a throughput acceptance result and
does not establish a fusion speedup. The exact stdout and engine logs are
`official-hotset16_5-cold-fused-debug.stdout.log` and
`official-hotset16_5-cold-fused-debug.engine.log`.

After the process exited, CUDA enumeration failed again: `nvidia-smi` could
not communicate and the matching `llama-bench --list-devices` reported no
CUDA device at 18:15. The prepared uninstrumented hotset repeats and split
DEBUG/repeat profiles have not been run. The 42 and 28 tok/s gates remain
unmet/unverified.


## Post-diagnostic CUDA recheck (2026-10-30 18:25 AEDT)

After the DEBUG graph run exited, the matched `llama-bench --list-devices`
again reported no CUDA-capable device; `nvidia-smi` could not communicate with
the driver. Host load was 5.89 on the 12-CPU affinity set. The uninstrumented
split/hotset runs remain pending.


The DEBUG engine log's three per-request `REQUEST-DECODE-WALL` rates were
2.01, 4.08, and 2.52 tok/s (200 generated tokens each). The shutdown aggregate
was 2.61 decode / 2.41 overall tok/s for 600 tokens. These highly variable
instrumented results confirm this run cannot be used as an acceptance
measurement; retain only its graph-path and timing attribution evidence.

## Post-fusion teacher-forced parity package prepared (2026-10-30)

The saved 190-row parity report predates the cold-fusion graph integration.
Prepared a follow-up run under
`build_mtp_results/2026-10-30-ornith-122b-postfusion-parity-5060ti/` that
reuses the exact llama.cpp `Hello there.` prompt and forced decode token 353,
compares pre-MoE hidden states and router outputs, and uses a 7 GiB
usage-ranked hotset on the reference RTX 5060 Ti to exercise both hot and cold
expert branches. The copied comparator reads the saved reference tensors via a
read-only symlink; the new ZeroMoE dump directory is separate. The TOML parses.

The package contains the exact prompt, config, command/environment, binary
manifest, comparator, and notes. It has not been launched because the latest
CUDA enumeration failed; run the preflight in `zeromoe.command.txt` first. This
parity rerun is required before accepting the fused graph path, even though the
CPU fused-versus-stock matrix and the DEBUG full-graph gate pass.


## Post-fusion parity preflight recheck (2026-10-30 18:33 AEDT)

Host load was 0.85 on the 12-CPU affinity set, but the matching sm_120a
`llama-bench --list-devices` probe with `CUDA_VISIBLE_DEVICES=1` again found no
CUDA device. The post-fusion parity package remains prepared and unrun.

## Latest status probe (2026-10-30 18:34 AEDT)

Host load is low (0.45 on the 12-CPU affinity set), but CUDA remains
unavailable: `nvidia-smi` fails and the matching sm_120a `llama-bench`
`--list-devices` probe with `CUDA_VISIBLE_DEVICES=1` reports no device. The
NVIDIA module (610.57.04), both `/proc/driver/nvidia/gpus` PCI entries, and
`/dev/nvidia0`, `/dev/nvidia1`, `/dev/nvidiactl` exist; no NVRM/Xid lines are
visible in the accessible kernel log. No device reset or driver action was
taken. The post-fusion parity and uninstrumented performance packages remain
ready but unrun.

## Latest device-state audit (2026-10-30 18:37 AEDT)

Host load is now low (0.18 on the 12-CPU affinity set), but the fresh
`nvidia-smi` and matched llama.cpp CUDA enumeration checks still fail. The
610.57.04 NVIDIA module, both `/proc/driver/nvidia/gpus` PCI entries, and the
core `/dev/nvidia*` nodes are present; no NVRM/Xid lines are visible in the
accessible kernel log. No driver reset, module reload, or device-node change
was attempted. The post-fusion teacher-forced parity and uninstrumented 28/42
tok/s gates remain impossible to verify until CUDA enumeration is restored.

## Retry after user-reported `nvidia-smi` availability (2026-10-30 18:38 AEDT)

A fresh `nvidia-smi` succeeded and showed both GPUs essentially idle (3090:
1 MiB used; 5060 Ti: 2 MiB; no processes). CUDA runtime enumeration still
failed independently: the matching sm_120a llama-bench with
`CUDA_VISIBLE_DEVICES=1` and the clean sm_86 build with
`CUDA_VISIBLE_DEVICES=0` both returned `ggml_cuda_init: no CUDA-capable
device`. Thus NVML visibility alone is not a go signal. No model run was
started; do not reset/reload the shared driver. Retry the parity or performance
commands only after the matching CUDA runtime lists the intended device.


## Qwen 3.8 full-graph reference and timing audit (2026-10-30)

### Layout, decode flow, and expert storage

- `general.architecture=qwen4exp` resolves to `Qwen4ExpGraphModel` in `models/graph/full_graph_models.cpp`; it shares the Qwen full-graph builder/runtime under `models/qwen/ggml_qwen4exp_graph_*.cpp`.
- A decode graph contains all 48 layers, including the model's GDN/full-attention blocks, low-rank hyper-connections, PLE state, MoE, and shared expert. `qwen4exp_full_graph_decode_run()` updates the single-token input, position, mask, and KV/GDN state, submits the scheduler graph asynchronously, synchronizes, reads outputs, then the next token replays the graph.
- The documented UD-IQ1_M split profile marks 26 layers CPU-affinity and requests a trailing 40-layer block. CPU affinity wins, leaving 22 complete expert layers in VRAM (17.74 GiB) and 26 layers' expert weights in the preloaded host DDR source. The distinct Q2_0 hotset profile is usage-ranked across all 48 layers: K=14,554 experts / 18.74 GiB in VRAM, remainder in DDR.
- The 22/26 split and the 14,554-expert hotset are different placement recipes. Their 28.66 and 43.12 tok/s references cannot be combined into one configuration.

### Timing evidence and missing views

The archived Qwen 3.8 UD-IQ1_M split baseline gives 16-token prefill at 25.7 tok/s, about 65 ms decode setup, and 400-token decode in 13.96 s (28.66 tok/s). The Q2_0 hotset reference is 43.12 tok/s over 196 measured steps. Neither archived INFO run provides a complete CPU/GPU per-layer timing map.

A new short diagnostic used the available IQ3_XXS Qwen3.8 shards, not the UD-IQ1_M baseline. With the reduced 37-layer candidate, the planner admitted 19 VRAM layers (15.47 GiB) and 29 CPU/DDR layers. For 15 prompt tokens, context 128, and 20 generated tokens, its stats were: prefill 1,345.9 ms; decode setup 121.0 ms; decode p50/p95 587.8/651.5 ms; decode wall 11,415.0 ms (1.75 tok/s); total 12,819.8 ms. This is diagnostic only. The full-graph execution gate recorded one prefill and 19 decode replays, but `memory_plan=unknown`, so `check_full_graph_moe_run.py --full-fit` fails and this is not acceptance evidence.

Nsight Systems recorded 268.1 ms of summed GPU kernel execution across the short request. The largest buckets are BF16/float matrix-vector kernels and quantized matrix-vector kernels; the report also identifies MoE top-k, GDN, hyper-connection, and normalization kernels. The report is `build_mtp_results/2026-10-30-qwen38-flash-3090-attribution/gpu-node-short/nsys-kernel-summary.txt`; the raw 29.8 MB `.nsys-rep` is beside it. This breakdown is by kernel name, not model layer.

The CPU callback can report CPU node time per layer/op and gaps between CPU nodes. A 400-token DEBUG IQ3_XXS run emitted these lines, but the engine log was overwritten by a later run because its copied TOML kept the parent `log_path`. The 20-token CPU retry also inherited that bad log path, so no trustworthy per-layer CPU report is currently preserved. The earlier 400-token diagnostic was 165.7 s / 2.41 tok/s and had 399 graph replays, but its raw engine log is not recoverable; do not use it as a baseline.

### Onboarding implication

Qwen3.8 is useful as the user's selected runtime/profile reference. Its current Qwen4Exp adapter reports `memory_plan=unknown` in the full-fit gate, however, so it does not satisfy the stricter current full-graph memory-plan contract despite successful graph replays. Ornith's Qwen35 adapter must retain its positive `Fits` byte plan. For both models, future performance diagnosis needs CPU per-layer/op timing, GPU device timing, and request workflow timing in separate, correctly isolated artifacts; no per-layer GPU mapping exists yet.


### Follow-up short CPU timing sample (2026-10-30)

A path-correct Qwen IQ3_XXS DEBUG run is now under `build_mtp_results/2026-10-30-qwen38-flash-3090-attribution/cpu-layer-corrected/`. It captured one CPU timing row at position 24: CPU node work 28.283 ms, CPU-node gaps 36.851 ms, and 29 CPU-affinity layers at 0.62–1.66 ms each. L20 was the highest displayed at 1.66 ms. The 20-token workflow was 15 prompt tokens, 651.951 ms prefill, 67.41 ms decode setup, 13.52 tok/s steady decode, 12.19 tok/s decode wall, and 2.354 s total. This is diagnostic only: Qwen IQ3_XXS differs from the archived UD-IQ1_M baseline and `memory_plan=unknown` makes the full-fit checker fail. Host load increased during the surrounding session, so no speed claim follows.

The separate Nsight node trace is under `gpu-node-short/`; the kernel summary totals 268.106 ms across its short request. It reports GPU kernel categories but not per-layer GPU time. The first attempt at Nsight's NVTX capture range produced no report; the follow-up bounded 20-token trace succeeded.

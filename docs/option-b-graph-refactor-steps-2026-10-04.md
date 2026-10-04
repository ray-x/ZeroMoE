# Graph refactor: staged A/B work

User instruction: follow the reviewed recommendations and run A/B at each step.
This continues [the early-ready handoff](handoff-2026-10-04-option-b-early-ready.md).
Preserve all generated tests, logs and discarded measurements. Do not commit
unrelated work or enable experimental defaults without evidence.

## Requirements and sequence

1. All-hot CPU skip: after the input-ready event, detect zero cold routes,
   clear the cold result and skip the prepared CPU graph. Preserve the current
   H2D and weighted merge. Compare against early-ready enabled without the skip.
2. Transfer/join costs: profile the existing scheduler copy path, then compare
   an owned GPU cold-output buffer and asynchronous upload joined by an event.
   Preserve input ownership, arithmetic and replay epochs. If CPU route/mask
   bookkeeping remains material, transport GPU-remapped cold IDs/weights as a
   separate A/B change rather than combining unrelated changes.
3. Next-layer prediction: compare an explicit predictor against static usage
   on actual routed IDs, report cold recall, wasted transfers and deadline
   success, and measure predictor overhead separately. Avoid assuming smooth
   hidden states imply accurate predictions.
4. Dynamic staging: implement only with stable GPU slots, coherent route maps,
   publication after transfer completion and holds throughout compute. Begin
   with a bounded target layer before expanding. Late/incorrect predictions
   fall back to CPU; selected experts must never change. GDS/cuFile is excluded
   because the host does not support it.

Each code stage requires identical-history logit comparison, appropriate native
checks and a guarded six-pair throughput A/B. Both arms use the same binary,
config, history and VRAM policy. No dump-I/O timing is used for acceptance.
Keep BF16 experimentation disabled and use 10 CPU threads. Reuse successful
measurements; retry only invalidated samples.

## Step 1 implementation

`MOESTREAM_QWEN_EARLY_READY_SKIP_ALL_HOT=1` selects the new path. The context
copies the static cold mask at construction. Every replay counts its cold
routes from the already-read IDs, and zeros the weighted cold output if the
count is zero. Original H2D and merge remain. Diagnostic counters distinguish
all-hot visits from actual skips. The default is off.

`bench/option_b_early_ready_check.py` now accepts `--baseline-env NAME=VALUE`
and `--candidate-env NAME=VALUE` to compare adjacent stages with early-ready
on in both arms. Overrides are recorded and checked during resume.

Numerical A/B: `logs/option_b_step1_all_hot_logits/`.
All 64 raw records were byte-identical; KL, TV and logit deltas were zero.
The baseline's dump timings are excluded. Model verification passed:
`logs/option_b_steps_model_verify.log`.

Reproduction (choose a new output directory, or add `--resume` for the same run):

```bash
python3 bench/option_b_early_ready_check.py \
  --engine ../moestream-build-upstream-master/moestream \
  --out-dir logs/option_b_step1_all_hot_perf \
  --prompt-ids logs/quant_decode_compare_chat_20261004/prompt.tokens \
  --continuation logs/quant_decode_compare_chat_20261004/Q2.tokens \
  --reps 6 --all-split --wait-idle 300 \
  --baseline-env MOESTREAM_QWEN_EARLY_READY_ALL_SPLIT=1 \
  --candidate-env MOESTREAM_QWEN_EARLY_READY_SKIP_ALL_HOT=1
```

Step 1 completed: native suite passed (`logs/option_b_step1_all_hot_native.log`).
Six guarded pairs measured 32.442 versus 32.451 tok/s (+0.03%), with 2/6
paired wins. It fails the 1.5% acceptance bar and stays off. Steady scheduler
profiling, derived from cumulative 32/64 reports, showed CPU work decreasing
15.337 -> 15.093 ms while GPU wait/copy increased 1.718 -> 1.898 ms. The saved
all-hot CPU work was largely hidden under the already-running GPU branch.
Results: `logs/option_b_step1_all_hot_perf/summary.json` and its manifest/logs.

## Step 2a: asynchronous cold-output join

`MOESTREAM_QWEN_EARLY_READY_ASYNC_OUTPUT=1` owns a fixed GPU cold-output buffer
per layer. The CPU marker uploads the identical weighted cold result from its
pinned output on a nonblocking stream, records completion and queues a wait on
the producer stream before the GPU merge. The marker remains explicitly
ordered after hot/shared work and before the merge. Input, ID and output buffers
are not allocator scratch. Replay synchronization protects reuse at the next
token. Ordinary scheduler output copying remains the control path.

This step is compared with all-hot skipping disabled in both arms, because
Step 1 did not meet the acceptance bar. The existing GPU wait/copy cost is about
1.72 ms/token on the Step-1 control, so the join is material enough to measure.

Step 2a correctness: the completed dumps at
`logs/option_b_step2_async_output_logits/` have 64 byte-identical records;
`comparison.json` reports zero delta. The benchmark's timing admission failed
because another build started, so those timings are excluded. All 44 contexts
reported 32 queued uploads. Native checks passed:
`logs/option_b_step2_async_output_native.log`.

Step 2a completed all six guarded pairs. It won 6/6, but the mean gain is below
1.5%, so the asynchronous-output flag remains optional. See the exact values
in `logs/option_b_step2_async_output_perf/summary.json`; interrupted build
samples were retained and retried by `/tmp/run_graph_ab_retry.py`.

## Step 2b: immutable route-table residency

The main graph uses immutable remap/mask tables created through `host_input`,
which marks them as user inputs. The scheduler repeats input copies and waits
for these tables on every token. `MOESTREAM_QWEN_EARLY_READY_RESIDENT_TABLES=1`
uploads those tables once to a fixed GPU buffer. The prepared CPU graph still
reads the original host tables. This is a separate A/B with asynchronous output
and all-hot skipping off, so the measured effect belongs to table residency.

Step 2a exact result: 32.361 -> 32.800 tok/s (+1.356%), 6/6 wins. Step 2b continues against asynchronous output disabled. The first Step-2b build saw a transient Exl3CpuMatrixCache forward-declaration error during concurrent header edits; the declaration is now present in the authoritative header and the build was retried without editing that unrelated work.

## Pending later work

Prediction evaluation will include the next layer's router projected from the
current FFN input, static cold-expert usage and previous-token routes as controls.
Prediction must be recorded alongside the actual next-layer routes and the
static cold mask; compare at a matched cold-expert transfer budget. Trace I/O
is excluded from throughput runs. Measure predictor overhead separately before
using predictions to move weights.

Any staging implementation must make the captured matmuls read the staged
weights through a valid slot remap. A bitmap alone is insufficient. Reset/publish
an epoch before consumption, publish readiness only after all three planes are
copied, and prevent slot reuse during current-token compute. A miss or late
transfer remains a CPU route. First validate a bounded target layer before
extending cache ownership across the graph.

Step 2b raw logits matched exactly for 64 records. Its first native suite
failed because Test 23 passed no cache budget and tried to allocate almost the
entire GPU for two 128-KiB buffers; the resulting allocation OOM also affected
the following KDA test. The pinning test now passes an explicit 512-KiB cache
budget. This is a localized test-fixture correction in `src/main_test.cpp`, not
a change to production cache allocation. Native validation was rerun.

The bounded-cache native retry passed all tests:
`logs/option_b_step2b_resident_tables_native_retry.log`. Test 23 now uses
0.000524288 GB and verifies uploaded bytes correctly. Step-2b six-pair A/B is
running with the original early-ready path in both arms and only residency
changed in the candidate.

Step 2b completed all six guarded pairs and won 6/6. Its exact summary is in
`logs/option_b_step2b_resident_tables_perf/summary.json`. It meets the acceptance
bar and is the retained base for later A/Bs: GPU table residency on in both
arms, all-hot skip and asynchronous-output experiments off unless explicitly
compared again. Table residency does not change expert placement or selected
routes and all 64 numerical records matched exactly.

## Step 3: predictor probe

`MOESTREAM_QWEN_EARLY_READY_PREDICT=1` projects each split layer's FFN input
through the next split layer's existing router and bias to forecast its top-K.
This is a proxy input, not a claim that it matches the next layer's actual
post-attention input. The auxiliary GPU work is ordered after the early input
fork and before hot/shared work, so it can overlap current cold CPU work.
`MOESTREAM_QWEN_EARLY_READY_PREDICT_LAYER` can restrict the target.

`MOESTREAM_QWEN_EARLY_READY_ROUTE_TRACE=path` records static cold masks and
actual/predicted IDs after replay synchronization. Trace I/O is excluded from
throughput. Static cold usage and previous-token routes will be controls at
matched cold-candidate counts; measure forecast overhead before staging.

Step 2b exact result: 32.302 -> 33.111 tok/s (+2.505%), 6/6 paired wins.
It is the accepted experimental base for Step 3. The predictor A/B keeps table
residency on in both arms and includes trace only for correctness/accuracy.
Throughput measurements will have trace and logit dumps disabled.

Step 3 correctness/native checks passed. All 64 raw logits match exactly.
`logs/option_b_step3_prediction_logits/prediction_scores.json` evaluates 2,646
next-layer visits: proxy cold recall 62.72%, static cold usage 2.80%, previous
same-layer token routes 37.74% at matched candidate counts. Proxy precision is
53.87%; 35.84% of mixed visits have all cold experts predicted. Transfer-byte
lower bound is 300.34 us/visit at measured 12.3 GB/s. These are forecast metrics,
not achieved deadlines or model speedup. Layer 13 proxy recall is 64%, with a
174.83-us transfer lower bound, making it a suitable bounded staging proof.

A separate staged GPU branch can consume ready cold experts without copying
all static hot weights into a new arena. Its per-route IDs must be -1 for
unready/static-hot routes; the CPU complement must be updated from the actual
GPU selection. Epoch-tagged double buffers and a background pinned-copy worker
can preserve fallbacks when predictions or transfers miss. This avoids both
GDS and host callbacks inside conditional bodies. Preserve all three-plane
readiness and compute holds before reusing a slot.

Step 3 overhead A/B completed: 33.019 -> 33.003 tok/s (-0.05%), with 3/6 paired
wins. The forecasting work is effectively hidden under current CPU work on
this history. It still needs topical accuracy checks and realized staging
lead/deadline measurement.

## Step 4 bounded implementation under development

A third GPU expert branch reads an epoch-tagged staged pool while the static
hot branch remains intact. Only actual routes whose three planes have been
published for the current epoch move out of the CPU complement. Two banks
prevent overwrite while either DMA or the consuming token is in flight. A
background worker packs immutable DDR rows into owned pinned buffers and
issues H2D copies on a nonblocking stream. No bitmap-only residency change is
used. Begin with Layer 13 from Layer-12 forecasts, then scale only after runtime
coherence, late-transfer fallback and A/B checks.

The first bounded staging run stopped during buffer binding: CUDA quantized
weights require more bytes than their logical payload. Stage pools now use
`ggml_backend_buft_get_alloc_size` before allocating, including CUDA padding.
The failed attempt is retained at `logs/option_b_step4_stage13_logits/`; a new
same-binary correctness pair is running at `..._logits_v2/`.

The padded Layer-13 staging run completed. At 32 visits it uploaded 39 predicted
cold experts and consumed 14 staged routes. Logits are not bit-identical after
CPU->GPU expert placement changes: 63/64 top-1 agreement, mean KL 0.0005867;
the one disagreement had a 0.0109 baseline margin. Treat this as unresolved
until an independent GPU reference confirms exact staged outputs and forced
late-transfer fallback preserves the original CPU/GPU placement.

`MOESTREAM_QWEN_EARLY_READY_STAGE_VERIFY=1` snapshots staged weighted output
and compares it with a separate GPU graph freshly loaded from the immutable
DDR planes for the actual staged route IDs. Verification timing is excluded.
`MOESTREAM_QWEN_EARLY_READY_STAGE_DELAY_US` injects a bounded copy-stream delay
for fallback/epoch/bank-reuse checks. Both controls default off.

Latest build with GPU-reference verification and delayed-transfer injection
completed: `logs/option_b_step4_stage13_verify_build.log`. Verification run is
`logs/option_b_step4_stage13_verified/` with `STAGE_VERIFY=1`; its dump/verification
timing is excluded from all acceptance comparisons. Remaining gates: reference
match, forced-late fallback, topical forecasts, Layer-13/full-split A/B and native
checks. The active goal remains the complete graph refactor, not just these
completed substeps.

Independent reference validation passed for the actual staged outputs:
`logs/option_b_step4_stage13_verified/`. At the diagnostic boundary, all 32
verified outputs had max delta exactly zero. This validates packing, slot
selection and GPU FFN computation against the original DDR weights; the prior
logit differences are consistent with CPU/GPU placement arithmetic rather than
wrong staged weights. Forced 100-ms late DMA is now testing fallback and bank
holds at `logs/option_b_step4_stage13_late/`.

Forced-late validation passed: `logs/option_b_step4_stage13_late/comparison.json`
has 64 byte-identical records, zero KL/TV/delta. At 32 visits, staged routes were
zero, 16 requests were dropped because banks remained held, and all 32 GPU
reference checks had zero delta. This verifies late publication cannot change
selected experts or consume stale slots. Native staging validation is running
as unified exec session 85263, log `logs/option_b_step4_stage13_native.log`.

Next actions: finish staging native checks; run Layer-13 six-pair A/B with
verification/delay/trace off; collect proxy accuracy on code and creative-writing
prompts; validate all-eligible-layer staging and compare its end-to-end benefit,
including predictor/third-branch costs. Preserve earlier measurements and all
artifacts. Do not mark the refactor goal complete: staging scaling, broader
quality checks and final cleanup/adoption remain unfinished.

Staging native suite passed (`logs/option_b_step4_stage13_native.log`). The
Layer-13 six-pair timing proof is running as unified exec session 82699; logs and results
are `logs/option_b_step4_stage13_perf/`. Verification, trace and delay are off;
both arms keep the same Layer-13-only predictor. All generated files remain.

Layer-13 staging timing completed: 33.390 -> 33.246 tok/s (-0.432%), 2/6 paired wins. It is not adopted. Results: `logs/option_b_step4_stage13_perf/summary.json`. Broader forecasts, all-eligible-layer correctness/A/B, and final cleanup/adoption remain unfinished; the full graph-refactor goal stays active.

## Asynchronous output on the resident-table base

The adjacent-step retest completed at 33.225 -> 33.830 tok/s (+1.820%),
6/6 wins: `logs/option_b_resident_async_combined_perf/summary.json`. It clears
the 1.5% average threshold for this batch. Individual gains span 0.977% to
3.879%; the standalone result remains +1.356%. One compilation-contaminated
attempt was excluded, archived, and retried by the admission/resume harness.
Combined raw-logit parity still needs verification before adoption.

All 42 eligible target-layer staging reference checks are now running in
`logs/option_b_step4_stage_all_verified/`. This pair keeps resident tables and
the predictor in both arms; the candidate adds staging and independent GPU
expert-output verification. Its timing is excluded. The new reproducible
`bench/option_b_topic_check.py` captures code/creative histories and scores
forecast controls before teacher-forced or free-running staging comparisons.

The first all-layer reference attempt declined graph allocation: free VRAM
was 0.73 GiB, below the existing 1-GiB floor. It ran the layered fallback and
is invalid as staging evidence; preserve `..._stage_all_verified/`. An unrelated
compilation also contaminated it. Keep the safety floor. The verifier now
shares one independently loaded reference graph/weight workspace across
compatible layers (checks backend, quantization, dimensions and slot count),
while each actual staged output has its own snapshot. No inference buffers
are shared or reduced by this diagnostic change.

Code review also found producer prediction-mailbox reads needed a lifetime
guard: after whole-model GPU synchronization, wait for worker forecast reads
to finish before another token can overwrite pinned predictions. This does
not wait for expert DMA; banks remain held through late copies and compute.
The route trace now records replay epoch and actual staged slot IDs; the
analyzer reports realized cold coverage and the fraction of predicted hits
that were ready at consumption, separate from forecast recall.

The shared-reference all-layer rerun succeeded in full-graph mode:
`logs/option_b_step4_stage_all_verified_v2/`. All 42 eligible target layers
passed independent staged-output verification; at the 32-visit diagnostic,
1,344 checks report maximum delta zero. The complete run passed its later
verification assertions too. Raw model logits have 62/64 top-1 agreement,
mean KL 0.0008802 and mean TV 0.0107834. Placement moves quantized expert work
between CPU/GPU arithmetic, so this does not establish identical logits or
free-running quality. Verification/dump timing remains excluded. Host GPU
access is now available through the host interface; the prior sandbox driver
failure does not describe host hardware availability.

Combined resident-table + asynchronous-output numerical parity passed:
`logs/option_b_resident_async_combined_logits/comparison.json` has 64
byte-identical raw records and zero KL/TV/delta. This completes its numerical
gate; throughput reconfirmation and the broader runtime checks remain.

Topical teacher-forced validation uses 256-token captured code/creative
histories at `logs/option_b_topics_stage_all_verified/`. A concurrent Exl3
compile interrupted the first code comparison after its seed capture
succeeded. The capture and discarded comparison are preserved. Topic resume
now checks engine/config/settings hashes and reuses successful captures;
dump-only comparisons may tolerate CPU contention with timing invalid.
The A/B harness also rejects `layered decode remains active` and allocator
refusals, which the earlier generic fallback check did not catch.

Both 256-token topical reference comparisons completed. On code, proxy cold
recall is 54.39%; actual staged coverage is 17.43% of cold routes, and 32.03%
of correctly predicted cold routes are ready at use. On creative writing,
recall is 74.85%; actual staged coverage is 28.61%, and 38.22% of predicted
hits are ready. These are diagnostic runs with trace and independent reference
verification enabled; do not substitute them for normal-run deadline or
throughput measurements. Model-logit top-1 agreement is 255/256 for code and
252/256 for creative writing. Original-weight GPU verification assertions
passed in both. See each topic's `prediction_scores.json`,
`staging_scores.json`, and `comparison/comparison.json` in
`logs/option_b_topics_stage_all_verified/`. Free-running coherence remains to
be checked. The next native suite is `logs/option_b_stage_all_native.log`.

The native suite passed (`logs/option_b_stage_all_native.log`). The all-layer
100-ms GPU-copy-stream delay run also passed independent staged-output
assertions, but does not prove all routes take CPU fallback:
`logs/option_b_stage_all_late/` reports two staged routes at the 32-visit
diagnostic. Forty-two GPU delay kernels slow the decoder enough that some
transfers meet their consumer deadline. Retain this as a contention test.
The new debug-only `MOESTREAM_QWEN_EARLY_READY_STAGE_WORKER_DELAY_US` delays
the worker after snapshotting its prediction, before packing/queuing DMA.
It avoids occupying GPU resources and will support a strict all-late fallback
check. Normal execution has both delay controls off.

The worker-delay all-layer fallback check passed:
`logs/option_b_stage_all_worker_late/` has 64 byte-identical baseline/candidate
records, zero KL/TV/delta, and zero staged routes in the 32-visit diagnostics.
This completes the strict late-publication/epoch fallback gate.

Bounded code capture continued past its first EOS because the timing-style
harness deliberately ignores stops. Its natural prefix ends at generated
token 188. Derived `natural_prefix_summary.json` and filtered route artifacts
are retained under each topic. For the code prefix, all 189 logit choices
(including EOS) agree; mean KL is 0.0001760, proxy cold recall 48.92%, actual
staged coverage 16.06%. Creative capture has no stop within 256 tokens, so its
previous figures stand. Do not evaluate the post-EOS synthetic continuation
as a real answer-quality regression. `bench/option_b_request_check.py` now
checks code/creative/code in one engine process with natural stops enabled,
retaining complete responses and reference diagnostics; run
`logs/option_b_multi_request_free/` is in progress.

Three consecutive code/creative/code requests completed in both modes:
`logs/option_b_multi_request_free/results.json`. Both first/third code answers
are identical; code output is also identical across modes. The creative
continuations remain coherent but differ and reach the 512-token limit.
Staging passed 1,050 diagnostic reference batches during the sequence,
including replay epochs continuing across request resets. Do not describe
the truncated creative scene as a complete answer to every prompt constraint.

Six-pair all-layer staging throughput is 32.642 -> 28.517 tok/s (-12.636%),
0/6 wins: `logs/option_b_stage_all_perf/summary.json`. Keep staging off.
Compilation/test interference was excluded and retried. The final timing
guard was expanded to cover Exl3 and Qwen benchmark executables, compiler
helpers, and compute processes on every GPU during inference. Its manifest
admission version is 2; preserve version-1 batches and use a new directory
rather than mixing samples with different admission monitoring.

The final six-permutation three-arm comparison is running in
`logs/option_b_scheduler_final_three_arm/`: original path (`baseline`),
early-ready with resident tables (`early`), and the latter plus asynchronous
output (`extra`). Each arm appears six times and every position/order is
balanced. Staging and its predictor are off. This provides a fresh overall
refactor measurement and a contemporaneous async-output incremental A/B.

## Validated runtime choice

Final admission-version-2 timing completed in
`logs/option_b_scheduler_final_three_arm/summary.json`:
original 31.848, early-ready + resident tables 33.294 (+4.538%), and the latter
+ async output 33.874 tok/s (+6.361% overall, +1.744% incremental). Both
comparisons won all six paired repetitions. Adopt the early-ready + resident
tables + async-output bundle as the opt-in Q2_0 0/44/4 runtime profile. Keep
all-hot skip, forecasting and dynamic staging off; retain their experiments
and all generated evidence for future work.

The profile is `configs/qwen4exp_q2_0_option_b_scheduler.env`; use it with the
existing TOML and the patched upstream integration build:

```bash
source configs/qwen4exp_q2_0_option_b_scheduler.env
../moestream-build-upstream-master/moestream \
  --config configs/qwen4exp_q2_0_option_b_rightsize_split44_x0_cov80.toml
```

This preserves the 8,421 static VRAM experts (10.84 GiB). The refactor adds
small persistent routing/input/output buffers and stream/event handles,
rather than increasing expert residency. Global defaults remain off because
the opt-in scheduler hook requires the patched integration build. The BF16
dispatch experiment remains deferred. Results apply to this single-3090 Q2_0
profile and fixed teacher-forced history with ten CPU threads; other layouts,
quantizations, prompt distributions and devices need their own measurements.

The final three-mode numerical check completed in
`logs/option_b_scheduler_final_three_arm_logits/`. Original, table and
table+async modes each produced the same 64 raw logit records, byte for byte;
both comparisons have zero KL, TV and maximum delta. These dump timings are
invalid for throughput. The final native suite passed in
`logs/option_b_scheduler_final_native.log` against the current working tree.
Shell/Python syntax and diff whitespace checks passed. The completed handoff
is `docs/handoff-2026-10-05-graph-refactor-complete.md`; it supersedes the
historical pending-state notes earlier in this worklog.

Post-native current-binary parity also passed in
`logs/option_b_scheduler_post_native_logits/`: original/table/table+async
64-record dumps match both each other and the prior final dumps byte for
byte. The refreshed binary hash matches this manifest. Heavy compilation
from a separate session explains the slow dump run; its timing is invalid
and does not replace the guarded throughput result. No further graph-scheduler
acceptance check is pending for the selected profile.

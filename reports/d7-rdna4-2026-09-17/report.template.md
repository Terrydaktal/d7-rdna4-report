# Restoring M1/M8 Numerical Agreement in RDNA4 Speculative Decoding

**Terrydaktal · 17 September 2026 · Technical report, version 2**

## Abstract

A pinned Radiance/vLLM inference stack produced different target logits when the
same saved token history was processed one position at a time (M1) or eight
positions at a time under D7 speculative verification (M8). The investigation
localized numerical differences in GDN prefill and decode, convolution, gated
normalization, attention, residual normalization and the BF16 vocabulary head.
The repaired, compiled implementations now agree on top-1/10/20 membership,
ordering, retained scores, inclusive boundary ties and full-vocabulary logit
hashes at 10,000 decode positions and 23 initial-prefill predictions. Four
subsequent performance changes reduced a repaired verification round from
approximately 76 ms to 60 ms in the measured 60K-input control.

This report documents an empirical result on a pinned configuration. It does
not establish arbitrary-input equivalence, equality to an independently
specified mathematical model, or improved task accuracy. The separate
eager-versus-compiled reference discrepancy is reported below.

## 1. Configuration and reference

- One AMD Radeon AI PRO R9700, gfx1201, TP1.
- Radiance 1.0.16 / vLLM 0.28, PyTorch 2.12.0, ROCm 7.14, Triton 3.7.1.
- Qwen3.8-27B-Uncensored-MXFP4-awq; 64 layers, including 48 GDN and 16 attention
  layers; hidden width 5,120; vocabulary size 248,320.
- The target vocabulary head uses the full original BF16 weights in both arms.
  INT2 candidate selection is excluded from the equality claim. Other model
  weights, activation quantization and cache formats remain part of the pinned
  quantized backend; this is not an unquantized BF16-model comparison.
- The compiled comparisons use Inductor and piecewise GPU graphs, with capture
  sizes `[1, 2, 4, 8]`. Actual target graph replay is audited.
- The numerical reference is the explicitly selected serial implementation.
  Repairs also affect M1/preparation paths, so repaired M1 is measured afresh.

The 10K corpus comprises 23 saved Pi continuations with 57,008–65,527-token
prefixes. Both arms consume identical saved token histories at aligned logical
positions. Each M8 group processes eight positions with all seven supplied
proposals accepted. The denominator is 10,000 processed decode positions,
excluding intermediate-prefill head calls. Final initial-prefill predictions
have a separate denominator of 23. No generated tools execute during replay.

## 2. Agreement definitions

For each position t, let z1(t) and z8(t) be the full vocabulary logit vectors.
T_k sorts logits by decreasing score and then increasing token ID for ties.

```text
set agreement(k)   = count(set(T_k(z1)) == set(T_k(z8))) / N
order agreement(k) = count(T_k(z1) == T_k(z8)) / N
mean shared(k)     = sum(|set(T_k(z1)) intersect set(T_k(z8))|) / N
```

Retained-score equality, inclusive boundary-tie membership and full-vector
SHA-256 equality are separate checks. Matching hashes are strong empirical
evidence, not a collision-free mathematical proof. Agreement measures execution
consistency; it is not a percentage of questions answered correctly.

## 3. Top-1/10/20 and ordering results

### Final compiled, optimized repair: full 10K corpus

| Prediction | Same token set | Same ordering | Mean shared tokens |
| --- | ---: | ---: | ---: |
| Top 1 | 10,000 / 10,000 (100%) | 10,000 / 10,000 (100%) | 1 / 1 |
| Top 10 | 10,000 / 10,000 (100%) | 10,000 / 10,000 (100%) | 10 / 10 |
| Top 20 | 10,000 / 10,000 (100%) | 10,000 / 10,000 (100%) | 20 / 20 |

Retained scores, inclusive tie sets and full-vocabulary hashes also match at
all 10,000 positions. All measures separately match for **23/23 initial-prefill
predictions**. Both workers exited successfully. The completion audit checked
all per-continuation receipts, fixture identities, widths, aligned positions,
runtime receipts and the recomputed aggregate.

### Compiled before/after control: 320 positions

This is the directly comparable compiled control, separate from the larger
historical eager run. Each implementation is compared with its corresponding
M1 reference.

| Prediction | Original M8: same set | Original M8: same order | Fixed optimized M8: same set | Fixed optimized M8: same order |
| --- | ---: | ---: | ---: | ---: |
| Top 1 | 318/320 (99.375%) | 318/320 (99.375%) | 320/320 (100%) | 320/320 (100%) |
| Top 10 | 161/320 (50.3125%) | 17/320 (5.3125%) | 320/320 (100%) | 320/320 (100%) |
| Top 20 | 76/320 (23.75%) | 0/320 (0%) | 320/320 (100%) | 320/320 (100%) |

Full-vocabulary hashes match at 0/320 original positions and 320/320 repaired
positions. The original initial-prefill vectors also differ. Original compiled
M1 versus repaired compiled M1 agrees on top-1 at 319/320 positions and differs
in every full-vector hash; reference changes are material to this experiment.

### Historical unfixed eager baseline: same 10K corpus

| Prediction | Same token set | Same ordering | Mean shared tokens |
| --- | ---: | ---: | ---: |
| Top 1 | 9,852/10,000 (98.52%) | 9,852/10,000 (98.52%) | 0.9852 / 1 |
| Top 10 | 4,857/10,000 (48.57%) | 799/10,000 (7.99%) | 9.3487 / 10 |
| Top 20 | 2,270/10,000 (22.70%) | 9/10,000 (0.09%) | 18.6781 / 20 |

These historical results use eager execution. They must not be presented as an
unfixed **compiled** 10K measurement. The final compiled run uses a fresh
compiled M1 reference, not this saved eager reference.

## 4. Compiled stage timings and isolated correctness measurements

Both timing profiles use the same 60,000-input-token Pi fixture and the
optimized compiled configuration: **`enforce_eager=false`, Inductor, PIECEWISE
GPU graphs, capture sizes `[1, 2, 4, 8]`**. The actual traces contain **65 target
GPU graph launches in every captured round**, in both arms. Runtime receipts
are recorded in [compiled-execution-audit.json](evidence/compiled-execution-audit.json).
These are not eager timings. The full BF16 target head is used on both sides.

The finer audit found incomplete expected target-kernel inventories in old
round 8 and fixed round 7 (one-based): respectively two and three missing
recorded events relative to the modal inventory. This does not establish that
the backend skipped those computations. The table below uses the **six paired
rounds with complete inventories on both sides**, rounds 1–6. Selection depends
on event counts, not speed. Both original eight-round receipts and the
[original aggregate table](historical-eight-round-stage-table.md) are retained;
all recorded dispatches, including excluded rounds, remain in the evidence.

Each time is the sum of recorded GPU dispatch durations divided by six. Layer
stages sum all their layer instances in a round. The 256 projection dispatches
per round have been mapped using the checked four-projection sequence in each
of 64 layers and the intervening GDN, attention and SiLU markers. The roughly
25 ms folded-projection total is **the joint MLP gate/up projection across all
64 layers**, rather than an unexplained collection of unrelated stages.

**“No correctness repair” describes the code change; it does not certify that
the stage has no numerical differences.** The isolated columns require a
separate 320-position experiment. A completed end-to-end 320-position match
cannot be copied into every stage's column. Intermediate tensors are not
vocabulary logits: their largest 20 coordinates are not the model's top-20
predictions. The intended isolated measurement holds the stage input and
initial state at the reference values, substitutes only that stage, and uses
an identical reference remainder to obtain vocabulary top-20. Direct tensor
and state equality is recorded separately. Cells without that evidence remain
explicitly unmeasured. Drafter and unattributed bookkeeping rows do not claim
target-top-20 equivalence.

The capture prerequisite has now passed: a diagnostic that observes already
compiled launches (with graph replay disabled for host copies) reproduced the
graph-enabled release at **320/320 decode positions and the initial prefill
prediction**, including full-logit digests. It recorded 27,800 operation
boundaries across 40 eight-position groups. This establishes output agreement
for the captured sample, not hidden-state or isolated-stage equivalence.
See [capture output bridges](evidence/compiled-capture-output-bridges.json).
The first capture prototype altered module boundaries and failed during
compiler warm-up; it produced no qualifying positions. Its results have not
been substituted for this successful capture.

The subsequent isolated-head probe was interrupted by a host OOM-killer event
after reaching its 320-position progress marker and before final validation and
result publication. It contributes **no completed isolated-stage result** to
the table. The original compiled reference evidence is preserved in verified
encrypted archives; repeating the intermediate capture and completing isolated
comparisons remain outstanding. No eager-versus-compiled M8 stage comparison
has yet been completed.

{{STAGE_TABLE}}

Target-body subtotal: **{{TARGET_TOTAL}}**. Total recorded GPU kernel duration:
**{{ALL_KERNEL_TOTAL}}**. These are subtotals, not extra stages. Kernel durations
can overlap and exclude host/queue gaps; they are not wall-clock round latency.

### Timing changes that are not yet causally isolated

Activation quantization still uses the same named kernel and **256 calls per
round** in each arm. Across the six complete rounds its sum is **2.437 ms old
versus 0.814 ms fixed**. The traces establish the difference, not its cause.
There was no quantizer replacement or removal. Execution context, layouts,
cache behavior and measurement effects have not been separated by an isolated
quantizer experiment, so the report does not claim a threefold quantizer fix.

Attention decode averages **4.112 ms old versus 5.262 ms fixed**, but the fixed
per-round totals are **3.546, 7.054, 3.570, 6.916, 3.588 and 6.901 ms**. The
repaired source forms one or two query groups according to the eight queries'
16-token tile boundaries, preserving each query's M1 causal/softmax decisions.
A second active group repeats cached-context work. This is a plausible
mechanism for the two timing bands, not a measured attribution of every slow
round; the original traces do not record the kernel's sequence-length argument.
The split-KV merge is separate: **0.253 ms old versus 0.121 ms fixed**.

Unchanged projection timings also vary. A timing delta by itself does not
identify a correctness repair, an optimization, or its causal benefit.

### Every layer, including the 64 contributions to the gate/up total

Layer numbers are zero-based. “Other three projections” means the input,
attention/GDN output, and MLP down projections; all individual semantic stage
values are also retained in `stage-times.json`. Non-projection work includes
normalization, gating, quantization, attention/GDN operations and copies.

<details>
<summary>Expand all 64 layers</summary>

{{LAYER_TABLE}}

</details>

### Every recorded compiled kernel within each semantic stage

Calls are totals over the six retained rounds; times are per-round means.
Compiler-fused operations remain indivisible. The tables do not invent times
for separate constituents of a single GPU dispatch. Individual dispatch
durations, layer IDs and round IDs are retained in the two
`evidence/*-compiled-dispatches.json` exports; they contain no token IDs, raw
activations, absolute timestamps or process identifiers.

<details>
<summary>Expand the complete kernel breakdown</summary>

{{KERNEL_TABLE}}

</details>

Prefill is outside this steady-decode timing table and has a separate
causal/chunk-partition repair. CPU scheduling has no GPU-kernel duration here.
The last row retains GPU bookkeeping outside the attributed model scopes;
its individual kernels are listed without inventing missing semantic labels.

## 5. Brief unprofiled engine controls — 60K input, not 60K output

**The approximately 80–87 tok/s figures are short controls on one 60,000-input-
token Pi fixture. They are not results from generating 60,000 tokens.**

{{SPEED_TABLE}}

{{SPEED_SUMMARY}}

These direct-engine controls exclude HTTP/Pi, tools, snapshot publication,
cold prefill and warm-up. Rates pool tokens and time after the first output
chunk; round latency is the median of the per-response steady-round medians.
Throughput also depends on speculative acceptance, so it is not a pure kernel
speed ratio. Clean timing passes do not record GPU profiles or copy activations.

### Separate, earlier 60K-generated-token verify-head experiment

This earlier compiled experiment used 115 natural responses per method across
11 intact Pi request boundaries, with 57,008–65,527 input tokens. It predates
the final D7 repairs and **does not measure the final corrected M8's long-run
speed**. The head methods also have different numerical guarantees.

| Earlier head method | Natural responses | Generated tokens | Timed post-first seconds | Post-first tok/s |
| --- | ---: | ---: | ---: | ---: |
| Full BF16 reference head | 115 | 60,598 | 946.230 | 63.920 |
| INT2 block-8 / rerank-80 | 115 | 60,075 | 892.548 | 67.178 |
| INT2 global-128 / BF16 rerank | 115 | 60,348 | 892.266 | 67.506 |
| INT2 global-256 / BF16 rerank | 115 | 60,675 | 906.873 | 66.779 |

The first output chunks are excluded from both the rate numerator and timed
denominator. See [earlier-60k-output-speed.json](evidence/earlier-60k-output-speed.json).
No 60K-output throughput claim is made for the final repaired M8.

## 6. Repairs and upstream ownership

1. **GDN prefill:** raw per-token gates and an ordered transition avoid the
   diagnosed dependence on future gates, cumulative-gate rounding and chunk
   partition. Core implementation belongs with libr4d; adapters belong in Radiance.
2. **GDN decode:** match serial convolution arithmetic, gate precision, recurrent
   reductions, output rounding and retained-prefix state. Rejecting a speculative
   suffix must not make its state persistent. Packed QKV views remove copies.
3. **Gated normalization:** retain the serial row tile without serializing all
   eight rows. The vendored FLA dispatcher belongs in vLLM.
4. **Attention:** retain each query's causal range and serial reduction/split
   policy; share KV reads to recover performance. Core ownership is libr4d.
5. **Residual/QK/final normalization:** preserve the chosen finite-precision
   boundaries; retain FP32 residual sums in registers. Existing vLLM rounding
   PRs must be checked for overlap before proposing another repair.
6. **Full BF16 head:** interleave two arithmetic-preserving M4 groups in one
   HIP launch. This avoids the first repair's duplicated launch/concatenation
   overhead. The upstream skinny-GEMM implementation belongs in vLLM.
7. **Compiled integration:** install dispatch before tracing/capture, use opaque
   bindings where necessary, and validate startup-only fallbacks separately from
   real requests. Integration and conformance adapters belong in Radiance.

These are submission scopes, not a claim that the pinned adapters apply
unchanged to current upstream main. The new ports require their own tests.
Upstream links and submission state are maintained in `submissions.md`.

## 7. Remaining discrepancies and limits

**The completed 10K equality result is compiled M1 versus compiled M8 only.**
It does not establish eager/compiled equivalence or choose an independently
proved arithmetic reference. Corrected eager M1 versus corrected compiled M1
has **98.11% top-1 agreement**
on the same 10K corpus. Top-10 set/order agreement is 48.45%/7.56%; top-20
set/order agreement is 22.27%/0.04%. Every full-vector hash differs, including
all 23 prefills. This separate discrepancy is unresolved and is not evidence
that either execution is an independently proved mathematical reference.

The final 10K replay covers the all-seven-accepted D7 path. Separate small
operator tests exercise acceptance boundaries, state/history comparisons,
graph replay and injected faults, but this report does not claim full-model
coverage of every rejection width, arbitrary input, long-context range,
concurrency, snapshot restore or hardware platform. It does not prove that
model-generated loops are eliminated or measure improved coding accuracy.

Native correctness is tied to source, binary and configuration identities.
Unsupported changes require requalification. A replay corpus and matching hashes
cannot establish a universal theorem about all future optimizations.

## 8. Evidence and reproduction

The `evidence/` directory contains aggregate comparisons and kernel profiles,
with file SHA-256 values in `evidence-sha256.json`. Recompute the stage table,
kernel accounting and aggregate assertions without GPU use:

```sh
uv run python reports/d7-rdna4-2026-09-17/build_report.py
```

The original Pi transcripts, token IDs, ranked token IDs and raw traces remain
private. The aggregate measurements can be audited from the published receipts;
independent end-to-end reproduction needs a public replacement workload.
Synthetic operator regressions accompany the corresponding source submissions.

Pinned arithmetic repair SHA-256:
`8deef78cd58b355d85d142caac58a88ea68ad576a31fe7769cc3db49b18235e9`.
Final performance bundle SHA-256:
`d5e74aedda327d5543a0f4a47aeff80ec853c3eea5f9de49051b2d58ec109068`.
10K summary seal:
`740f52b749130d91ecb870153a02e8b28c232489c87997423146c2af2d5496f8`.

## References and attribution

- [Radiance](https://github.com/magiccodingman/vllm-radiance), the qualified downstream stack.
- [libr4d](https://codeberg.org/StillDeadcode/libr4d), RDNA4 attention and recurrent kernels.
- [vLLM](https://github.com/vllm-project/vllm), including ROCm skinny GEMMs and vendored FLA.
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention), serial GDN/normalization foundations.
- [vLLM batch invariance](https://docs.vllm.ai/en/latest/features/batch_invariance/), related existing work.
- [PyTorch numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html), batched/reduction-order limitations.
- Existing normalization work: [vLLM #49639](https://github.com/vllm-project/vllm/pull/49639) and [#52243](https://github.com/vllm-project/vllm/pull/52243).

This contribution builds on those implementations. Batch invariance itself is
not introduced by this report. Earlier DFlash RNG work in Radiance PR #8
backports [vLLM #54282](https://github.com/vllm-project/vllm/pull/54282) and is
not presented as a new upstream discovery here.

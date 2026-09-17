# Restoring M1/M8 Numerical Agreement in RDNA4 Speculative Decoding

**Terrydaktal · 17 September 2026 · Technical report, version 1**

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

## 4. Every compiled GPU stage/group, with repairs

Both profiles use the same 60K-input Pi fixture, compiled execution and actual
piecewise graph replay. The table sums GPU kernel durations over eight profiled
rounds and divides by eight. It covers every recorded kernel dispatch exactly
once. Target layer stages are totals across all layers per round, not per layer.
The full BF16 head is used on both sides.

Some original operations were fused by Inductor, so separate times for their
constituents do not exist in the captured profile. Those operations retain a
shared timing group. Matrix projections are likewise grouped by actual kernel
dispatch rather than assigning unsupported timings to individual projections.
`kernel-dispatches.csv` contains every kernel name, group, call count and time.

{{STAGE_TABLE}}

Target-body subtotal: **47.798 ms old; 47.726 ms fixed**. This subtotal includes
the target rows above and must not be added again. Total recorded GPU kernel
duration: **59.156 ms old; 58.819 ms fixed**. Kernel durations can overlap and
exclude host/queue gaps; they are not the wall-clock round latency.

Changes in timings of unchanged kernels, especially activation quantization,
are observations from these short profiles. They do not establish a causal
speedup from a code change in that kernel. The repair does not make every stage
faster: attention remains more expensive than the original arithmetic, while
other savings recover the total round cost.

Prefill has a separate causal/chunk-partition repair and is outside this steady
decode profile. Its speed is not inferred from the decode table. Sampling,
accept/reject bookkeeping and state metadata outside the attributed model
scopes are retained in the final accounting group; CPU-only scheduling has no
GPU-kernel duration in this table.

## 5. Unprofiled serving-speed controls

| Compiled configuration | Median round | Measured output rate | Evidence scope |
| --- | ---: | ---: | --- |
| Original, full BF16 target head | 59.619 ms | 85.212 tok/s | Three natural responses, 2,518 output tokens |
| Correctness repair, before performance recovery | 76.038 ms | 68.612 tok/s | Three natural responses, 2,237 output tokens |
| Repair after first three performance changes | 60.985 ms | 83.907 tok/s | Same three repaired responses, byte-identical outputs |
| Final repair, all four performance changes | 59.997 ms | 86.665 tok/s | One 743-token natural response |

The matching pre-optimization repeat for that final response was 76.025 ms and
69.687 tok/s. The output digest is unchanged. The final round is about 0.64%
above the original full-BF16 control's three-repeat median, but the different
repeat counts limit precision. The output rates also reflect acceptance, so
they must not be interpreted as pure kernel speed ratios.

These direct-engine controls exclude HTTP/Pi, snapshot publication, cold prefill
and warm-up. They are separate from the earlier 60K-generated-token verify-head
study, which also used compilation. The old legacy INT2 block-8/rerank-80
control measured 56.982 ms and 89.128 tok/s here, but uses an approximate target
head and is not the numerical reference for this report.

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

Corrected eager M1 versus corrected compiled M1 has **98.11% top-1 agreement**
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

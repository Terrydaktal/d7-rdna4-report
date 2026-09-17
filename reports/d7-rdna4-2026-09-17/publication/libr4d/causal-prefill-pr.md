# Add causal GDN prefill reference and partition reproducer

Rounded cumulative gates, midpoint scaling and a whole-call extreme-decay
fallback can change an earlier prefill prefix when its future suffix or chunk
partition changes. This experimental reference computes raw per-token gates and
uses a causal ordered recurrence, including identity for empty sequences.

The target is the existing unfused **R4D M1** arithmetic, including FP32 beta and
its output rounding. It is deliberately kept distinct from the later stock-FLA
contract used by the final D7 repair.

The pinned optimized reference passed 52/52 native state/output comparisons,
including complete partitions, causal prefixes, future-gate interventions,
nonzero state/history, strided gates and empty sequences. Prepared normalization
plus recurrence measured 2.380/2.373 ms in the two 2,049-token layer controls.
Current-header HIP compilation also passed without GPU execution; native
execution of that new header binding and public-dispatch integration remain open.

This supplies the reference, source-bound builders and native reproducer for
review. It does not present operator evidence as a complete model proof or
silently install an unqualified production default.

[Wider report and compiled stage table](https://github.com/Terrydaktal/d7-rdna4-report/blob/main/reports/d7-rdna4-2026-09-17/REPORT.md).

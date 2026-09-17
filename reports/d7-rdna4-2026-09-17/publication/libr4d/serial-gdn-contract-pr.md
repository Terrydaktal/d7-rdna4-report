# Add stock-FLA serial GDN contract and speculative reference

The R4D speculative path and stock serial GDN can differ in beta precision,
normalization/reduction order, convolution arithmetic and output conversion.
This reference bundle makes the selected stock-FLA target explicit and supplies
the oracle, ordered Triton scan and adapters used to repair the pinned downstream
implementation.

The state contract covers every acceptance count: row zero processes the pending
token, accepting k proposals retains after-row state k, and the newly emitted
correction/bonus token stays pending. CPU negative controls exercise state
ownership, prefix selection and rejected-suffix isolation.

Publication checks: 78 CPU tests passed; one retained compiler-artifact check was
explicitly skipped. Source identities match the qualified downstream sources.
The original native coordinator passed 38 exact state/output comparisons and
detected both injected faults. The complete compiled downstream repair later
matched top-1/10/20 sets, ordering and full-logit digests at all 10,000 positions,
plus 23 initial-prefill predictions; other stage repairs contribute to that result.

This is an experimental reference submission, **not a completed HIP production
port**. It provides a precise oracle for that port. Production dispatch, native
acceptance/graph checks and full-model qualification of a new libr4d integration
are still required.

This bundle supplies the GDN reference for **Fix 1**. The report also covers
**Fix 2**, a separate eager/compiled rounding alignment, and gives all four
isolated comparison columns on 320 common correct positions.

The tested Radiance integration is already in
[Radiance #11](https://github.com/magiccodingman/vllm-radiance/pull/11), depending
on [#10](https://github.com/magiccodingman/vllm-radiance/pull/10).
[Report and compiled stage timings](https://github.com/Terrydaktal/d7-rdna4-report/blob/main/reports/d7-rdna4-2026-09-17/REPORT.md).

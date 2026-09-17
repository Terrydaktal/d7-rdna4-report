# Add experimental query-isolated M8 attention with shared KV reads

The decode kernel's wave-wide ballot lets one query influence another query's
softmax rescaling decisions. Under a serial-query arithmetic contract, that can
change earlier outputs despite correct causal token masking.

This adds an experimental shared-KV translation unit with query-specific
six-head ballots, grouping at 16-token tile boundaries and per-query causal merge
extents. The public dispatcher is unchanged pending integration qualification.

The original source transformer, ABI wrapper and native probe are preserved with
their identities. A separate source-bound current-header build passed HIP
compilation without GPU execution. Its receipt explicitly remains
`BUILT_UNTESTED`; the new header binding still needs native testing.

The pinned downstream stage passed 192 query positions, future-query
interventions, guards and graph replay across three cache storage layouts. Its
60K FP8 layer time changed from 0.997 ms to 0.243 ms. The complete pinned compiled
Radiance repair later matched top-1/10/20 sets, ordering and full-logit digests on
10,000 positions. Those full-model results include other stage repairs.

This patch covers the attention part of **Fix 1**. The report's fresh final
timings additionally include **Fix 2**, which aligns eager/compiled rounding;
that compiler/RoPE work is a separate integration change.

[Report, four isolated comparisons, complete compiled stage timings and evidence](https://github.com/Terrydaktal/d7-rdna4-report/blob/main/reports/d7-rdna4-2026-09-17/REPORT.md).

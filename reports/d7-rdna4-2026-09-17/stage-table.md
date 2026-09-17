| Compiled GPU stage/group | Old M8 (ms) | Fixed M8 (ms) | Repair / interpretation |
| --- | ---: | ---: | --- |
| MXFP4 folded projections | 24.243 | 25.496 | No projection arithmetic change. |
| MXFP4 projections, split-4 kernel | 7.633 | 7.554 | No projection arithmetic change. |
| MXFP4 projections, split-1 kernel | 4.981 | 4.973 | No projection arithmetic change. |
| Attention decode | 4.104 | 5.673 | Keep each query's serial causal tile/reduction contract; share KV reads across eight queries. |
| Attention split-KV merge | 0.255 | 0.124 | Preserve the serial per-query split/merge arithmetic. |
| Activation FP8 quantization | 2.485 | 0.815 | Same quantization kernel; measured timing change is not evidence of a quantizer repair. |
| GDN recurrent update and gates | 1.150 | 1.211 | Match serial gate precision, reduction order, recurrent transition and output rounding; preserve packed QKV views. |
| GDN convolution | 0.769 | 0.228 | Use the qualified serial product/accumulation order with rolling state; remove unnecessary packed-QKV copies. |
| Attention KV write | 0.148 | 0.047 | No KV-format change; timing includes the same cache-write kernel. |
| Explicit residual and Q/K/final normalization | Included in fused group | 0.813 | Preserve reduction tree, rsqrt and BF16 rounding boundaries; retain FP32 residual sums in registers. |
| Explicit GDN gated normalization | Included in fused group | 0.098 | Fix the row tile at the serial shape while processing all rows in parallel. |
| Compiler-fused normalization, activations, embedding and other pointwise work | 1.398 | 0.336 | Bind repaired normalization as opaque compiled operators before graph capture; remaining operations retain compiler fusion. |
| Target copies, conversions and buffer initialization | 0.633 | 0.359 | Keep packed GDN views and remove three materializations plus concatenation; other copies remain. |
| Full BF16 vocabulary head | 4.106 | 4.134 | Two interleaved M4 groups in one HIP launch preserve serial arithmetic without two launches and concatenation. |
| Drafter, including proposal head | 6.334 | 6.447 | Unchanged drafter; kept separate from target correctness. |
| GPU work outside attributed model stages | 0.918 | 0.512 | Includes sampling/state bookkeeping/copies; no invented per-operation attribution. |

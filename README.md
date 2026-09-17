# D7 numerical agreement and performance report

**[Read the technical report](reports/d7-rdna4-2026-09-17/REPORT.md)** — the
rendered Markdown includes the top-1/10/20 and ordering results, compiled stage
timings, fixes, workload definitions and limitations.

Terrydaktal · 17 September 2026

This public repository contains the report, aggregate evidence and review
materials for the pinned R9700/Radiance M1-versus-M8 investigation. The final
compiled pair matched all 10,000 decode positions and 23 prefill predictions
on the reported measures. A later rounding alignment also produced exact
eager/compiled M8 agreement over 320 positions. The report distinguishes this
new sample from the historical 10K cross-mode discrepancy; a 10K rerun with
the new rounding settings remains pending. A compiled ABBA speed comparison
measured **60.906 ms before versus 60.900 ms after** preserving BF16 casts.
Neither result proves arbitrary-input equivalence.

[Upstream submissions](reports/d7-rdna4-2026-09-17/submissions.md) ·
[Compiled stage table](reports/d7-rdna4-2026-09-17/stage-table.md) ·
[Compiled rounding speed comparison](reports/d7-rdna4-2026-09-17/rounding-speed-table.md) ·
[Every layer](reports/d7-rdna4-2026-09-17/layer-table.md) ·
[Every compiled kernel](reports/d7-rdna4-2026-09-17/kernel-table.md) ·
[Eager/compiled boundaries before and after alignment](reports/d7-rdna4-2026-09-17/common-rounding-boundaries.md) ·
[Aggregate evidence](reports/d7-rdna4-2026-09-17/evidence/) ·
[Optional PDF](reports/d7-rdna4-2026-09-17/report.pdf)

## Repository contents

```text
reports/d7-rdna4-2026-09-17/
├── REPORT.md, report.template.md, report.pdf
├── evidence/                     Aggregate comparisons and GPU profiles
├── evidence-sha256.json          Evidence checksums
├── stage-table.md, stage-times.json, kernel-dispatches.csv
├── rounding-speed-table.md
├── layer-table.md, kernel-table.md
├── execution-mode-boundaries.md, common-rounding-boundaries.md
├── compiled-m1-m8-boundaries.md
├── historical-eight-round-stage-table.md
├── build_report.py, build_detailed_tables.py, analyze_compiled_trace.py
├── analyze_rounding_speed.py
├── build_execution_mode_tables.py
├── build_pdf.py
├── probe_upstream_gdn_norm.py
├── submissions.md
└── publication/
    ├── head/                    Upstream BF16-head probe, source and result
    └── libr4d/                  Prepared patches, descriptions and manifest
```

| Script | Input | Output / role |
| --- | --- | --- |
| `build_report.py` | Aggregate JSON receipts and Markdown template | Verifies aggregate assertions and kernel accounting; rebuilds the report, stage tables, dispatch CSV and evidence checksums. No GPU required. |
| `build_detailed_tables.py` | Sanitized per-dispatch exports and original aggregate receipts | Checks all recorded dispatch totals, selects paired rounds by inventory completeness, and generates semantic-stage, 64-layer and compiled-kernel tables. Called by `build_report.py`. |
| `build_execution_mode_tables.py` | Authenticated activation-boundary comparison receipts | Builds the complete eager/compiled before/after and compiled M1/M8 boundary tables. Called by `build_report.py`. |
| `analyze_compiled_trace.py` | Private profiler traces and their pinned original aggregate receipts | Validates graph replays, kernel/layer attribution and trace identity; exports only timings, symbols, layer IDs and round IDs. Original private traces are not published. This preparation step has already produced the checked-in exports. |
| `analyze_rounding_speed.py` | Archived timing receipts from the four compiled ABBA controls | Checks receipt hashes, sequential GPU leases, matching source/configuration identities, compiled graph replay and natural completion; recomputes per-round and pooled rates into `evidence/rounding-speed-abba.json`. Original receipts remain private; the published aggregate retains the per-round timings. No GPU required. |
| `build_pdf.py` | `REPORT.md` | Optional HTML/PDF rendering, with dependencies pinned in the script. |
| `probe_upstream_gdn_norm.py` | Compatible pinned ROCm/Triton/vLLM environment | Native synthetic gated-normalization comparison and JSON metrics; requires a GPU. |
| `publication/head/probe.py` | Compatible ROCm/PyTorch environment and adjacent HIP source | Native interleaved-head comparison and JSON results; see its README for prerequisites and commands. |

## Rebuild the public tables

From the repository root, run the aggregate check first:

```sh
uv run python reports/d7-rdna4-2026-09-17/build_report.py
```

To render a new PDF after rebuilding the Markdown, optionally run:

```sh
uv run reports/d7-rdna4-2026-09-17/build_pdf.py
```

These commands reconstruct the published report from aggregate evidence. The
original private Pi corpus, raw token IDs and full execution traces are not
included; independently repeating the full model experiment needs a suitable
workload and the pinned backend. Native probes are separate from the report
build and do not run automatically.

The earlier roughly 84 tok/s final result covers three short natural responses
on a 60,000-token input. The rounding comparison uses six responses per setting:
**84.12 → 83.07 tok/s**, with effectively unchanged round latency and fewer
committed tokens per round. Neither is a 60,000-output-token throughput result.
The long-output benchmark is reported separately and predates the final D7 repairs.
Both timing traces use compiled execution with observed GPU graph replay.
The detailed table identifies actual code repairs separately from observed
timing changes. An unmeasured isolated stage has no top-20 correctness claim.

# D7 numerical agreement and performance report

**[Read the technical report](reports/d7-rdna4-2026-09-17/REPORT.md)** — the
rendered Markdown includes the top-1/10/20 and ordering results, compiled stage
timings, fixes, workload definitions and limitations.

Terrydaktal · 17 September 2026

This public repository contains the report, aggregate evidence and review
materials for the pinned R9700/Radiance M1-versus-M8 investigation. The final
compiled pair matched all 10,000 decode positions and 23 prefill predictions
on the reported measures. That is a workload-specific empirical result;
arbitrary-input equivalence and the separate eager/compiled discrepancy remain
outside that claim.

[Upstream submissions](reports/d7-rdna4-2026-09-17/submissions.md) ·
[Compiled stage table](reports/d7-rdna4-2026-09-17/stage-table.md) ·
[Every layer](reports/d7-rdna4-2026-09-17/layer-table.md) ·
[Every compiled kernel](reports/d7-rdna4-2026-09-17/kernel-table.md) ·
[Aggregate evidence](reports/d7-rdna4-2026-09-17/evidence/) ·
[Optional PDF](reports/d7-rdna4-2026-09-17/report.pdf)

## Repository contents

```text
reports/d7-rdna4-2026-09-17/
├── REPORT.md, report.template.md, report.pdf
├── evidence/                     Aggregate comparisons and GPU profiles
├── evidence-sha256.json          Evidence checksums
├── stage-table.md, stage-times.json, kernel-dispatches.csv
├── layer-table.md, kernel-table.md
├── historical-eight-round-stage-table.md
├── build_report.py, build_detailed_tables.py, analyze_compiled_trace.py
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
| `analyze_compiled_trace.py` | Private profiler traces and their pinned original aggregate receipts | Validates graph replays, kernel/layer attribution and trace identity; exports only timings, symbols, layer IDs and round IDs. Original private traces are not published. This preparation step has already produced the checked-in exports. |
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

The roughly 84 tok/s final result covers three short natural responses on a
60,000-token input. It is not a 60,000-output-token throughput result. The earlier
long-output benchmark is reported separately, and predates the final D7 repairs.
Both timing traces use compiled execution with observed GPU graph replay.
The detailed table identifies actual code repairs separately from observed
timing changes. An unmeasured isolated stage has no top-20 correctness claim.

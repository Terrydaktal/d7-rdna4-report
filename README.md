# D7 numerical agreement and performance report

**[Read the technical report](reports/d7-rdna4-2026-09-17/REPORT.md)** — the
rendered Markdown includes the top-1/10/20 and ordering results, compiled stage
timings, fixes, workload definitions and limitations.

Terrydaktal · 17 September 2026

This report covers two major repairs on one pinned R9700/Radiance stack:

1. **Fix 1: compiled M1/M8 agreement.** All 10,000 decode positions and 23
   prefill predictions matched, including full logit vectors.
2. **Fix 2: eager/compiled agreement.** BF16 intermediate and RoPE product
   rounding were aligned. All 320 decode vectors and the prefill prediction
   matched; this is a separate sample from the Fix 1 10K qualification.

The stage table gives fresh **Old compiled M8** and **Final fixed compiled M8**
GPU timings, plus four isolated top-20 set/order comparisons on common correct
inputs. The original/final clean controls measured **59.68 / 60.05 ms per round**.
These brief controls use 60K input tokens. The earlier 60K-generated-token
benchmark is separate. Neither experiment proves arbitrary-input equivalence.

[Upstream submissions](reports/d7-rdna4-2026-09-17/submissions.md) ·
[Compiled stage table](reports/d7-rdna4-2026-09-17/stage-table.md) ·
[Compiled rounding speed comparison](reports/d7-rdna4-2026-09-17/rounding-speed-table.md) ·
[Every layer](reports/d7-rdna4-2026-09-17/layer-table.md) ·
[Per-layer correctness counts](reports/d7-rdna4-2026-09-17/isolated-stage-layer-comparisons.csv) ·
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
├── isolated-stage-layer-comparisons.csv
├── execution-mode-boundaries.md, common-rounding-boundaries.md
├── compiled-m1-m8-boundaries.md
├── build_report.py, build_detailed_tables.py, analyze_compiled_trace.py
├── analyze_rounding_speed.py, analyze_two_fix_profiles.py
├── combine_native_stage_evidence.py
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
| `analyze_two_fix_profiles.py` | Original/final native runtime receipts and compiled trace exports | Validates the matched configuration, three natural controls per arm, graph replay and paired complete profile rounds; writes the fresh timing audit. |
| `combine_native_stage_evidence.py` | Audited isolated-stage sweeps | Checks fixture identity and overlap consistency before combining the four comparison columns. The native replay drivers are in Radiance PR #11. |
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

The isolated diagnostics disable graph replay so that individual native calls
can be substituted. Their ordinary output must first match the graph-enabled
release at all 320 positions and prefill. Their durations are never used as
release timings. The separate ABBA comparison of Fix 2 alone measured
**60.906 / 60.900 ms** per round before/after preserving compiled BF16 casts.

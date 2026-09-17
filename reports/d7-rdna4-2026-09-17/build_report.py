"""Rebuild public numeric tables from aggregate, transcript-free receipts."""

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / "evidence" / name).read_text())


def category(name):
    if "radiance_mxfp4_fp8_gemm_folded" in name:
        return "MXFP4 folded projections"
    if "radiance_mxfp4_fp8_gemm_decode<8, 128, 4" in name:
        return "MXFP4 projections, split-4 kernel"
    if "radiance_mxfp4_fp8_gemm_decode<8, 128, 1" in name:
        return "MXFP4 projections, split-1 kernel"
    if "attn_decode_kernel" in name or "shared_decode" in name:
        return "Attention decode"
    if "attn_splitkv_combine" in name or "shared_merge" in name:
        return "Attention split-KV merge"
    if "dynamic_per_token_scaled_fp8_quant" in name:
        return "Activation FP8 quantization"
    if "gdn_recurrent_update" in name or "stock_gdn_scan" in name:
        return "GDN recurrent update and gates"
    if "gdn_conv_update" in name or "causal_conv1d_update" in name:
        return "GDN convolution"
    if "reshape_and_cache" in name:
        return "Attention KV write"
    if "stock_m1_gemma_norm" in name:
        return "Explicit residual and Q/K/final normalization"
    if "layer_norm_fwd_kernel" in name:
        return "Explicit GDN gated normalization"
    if "triton_" in name:
        return "Compiler-fused normalization, activations, embedding and other pointwise work"
    return "Target copies, conversions and buffer initialization"


FIXES = {
    "MXFP4 folded projections": "No projection arithmetic change.",
    "MXFP4 projections, split-4 kernel": "No projection arithmetic change.",
    "MXFP4 projections, split-1 kernel": "No projection arithmetic change.",
    "Attention decode": (
        "Keep each query's serial causal tile/reduction contract; share KV "
        "reads across eight queries."
    ),
    "Attention split-KV merge": "Preserve the serial per-query split/merge arithmetic.",
    "Activation FP8 quantization": (
        "Same quantization kernel; measured timing change is not evidence of a quantizer repair."
    ),
    "GDN recurrent update and gates": (
        "Match serial gate precision, reduction order, recurrent "
        "transition and output rounding; preserve packed QKV views."
    ),
    "GDN convolution": (
        "Use the qualified serial product/accumulation order with rolling "
        "state; remove unnecessary packed-QKV copies."
    ),
    "Attention KV write": "No KV-format change; timing includes the same cache-write kernel.",
    "Explicit residual and Q/K/final normalization": (
        "Preserve reduction tree, rsqrt and BF16 rounding boundaries; "
        "retain FP32 residual sums in registers."
    ),
    "Explicit GDN gated normalization": (
        "Fix the row tile at the serial shape while processing all rows in parallel."
    ),
    "Compiler-fused normalization, activations, embedding and other pointwise work": (
        "Bind repaired normalization as opaque compiled operators before "
        "graph capture; remaining operations retain compiler fusion."
    ),
    "Target copies, conversions and buffer initialization": (
        "Keep packed GDN views and remove three materializations plus "
        "concatenation; other copies remain."
    ),
}


def main():
    from build_detailed_tables import authenticated_read
    from build_execution_mode_tables import build as build_mode_tables

    modes = authenticated_read(ROOT / "evidence/current-compiled-eager-320.json")
    assert (
        modes["decode"]["positions"] == 320
        and modes["decode"]["full_logits_exact"] == 0
    )
    assert [
        (modes["decode"][k]["set_exact"], modes["decode"][k]["ranked_exact"])
        for k in ("1", "10", "20")
    ] == [(319, 319), (146, 14), (66, 0)]
    silu = authenticated_read(ROOT / "evidence/isolated-silu-modes-320.json")
    assert silu["status"] == "SAMPLE_CHECKED" and silu["negative_control_detected"]
    assert silu["inputs_unchanged"]
    decoded = list(silu["results"]["decode"].values())
    assert len(decoded) == 64 and all(
        row["own_capture_reproduced"] == 320 for row in decoded
    )
    assert (
        sum(
            row["compiled_vs_eager_on_common_input"]["different_elements"]
            for row in decoded
        )
        == 96380748
    )
    assert (
        sum(row["torch_native_vs_eager"]["exact_positions"] for row in decoded) == 20480
    )
    assert (
        sum(row["torch_fp32_vs_compiled"]["exact_positions"] for row in decoded)
        == 20463
    )
    pilot = authenticated_read(ROOT / "evidence/rope-gate-native-pilot.json")
    assert pilot["positions"] == 8 and pilot["gate"]["gate_input_exact"]
    assert pilot["rope"]["True"]["eager_q"]["exact_positions"] == 8
    assert pilot["rope"]["True"]["eager_k"]["exact_positions"] == 8
    assert pilot["rope"]["True"]["compiled_q"]["different_elements"] == 1496
    assert (
        pilot["gate"]["common_input_compiled_vs_eager"]["different_elements"] == 13524
    )
    assert pilot["gate"]["compiled_reproduces_own_output"]["exact_positions"] == 8
    assert pilot["gate"]["eager_reproduces_own_output"]["exact_positions"] == 8
    precision = authenticated_read(ROOT / "evidence/precision-casts-vs-eager.json")
    assert precision["decode"]["positions"] == 320
    assert precision["decode"]["full_logits_exact"] == 0
    assert [
        (precision["decode"][k]["set_exact"], precision["decode"][k]["ranked_exact"])
        for k in ("1", "10", "20")
    ] == [(316, 316), (151, 22), (76, 0)]
    cut = authenticated_read(ROOT / "evidence/precision-attention-cut.json")
    formulas = authenticated_read(ROOT / "evidence/precision-rotary-formulae.json")
    rotary = authenticated_read(ROOT / "evidence/rotary-rne-native-replay.json")
    assert rotary["status"] == "SAMPLE_CHECKED"
    assert rotary["negative_controls_detected"] and rotary["coefficients_unchanged"]
    assert rotary["versions"] == {
        "gpu": "AMD Radeon AI PRO R9700",
        "torch": "2.12.0+rocm7.14",
        "triton": "3.7.1",
    }
    for phase, count in (("decode", 320), ("prefill", 9)):
        for name in (
            "qkv_projection",
            "query_after_normalization",
            "key_after_normalization",
            "value",
            "gate_input",
        ):
            assert cut["results"][phase][name]["exact_positions"] == count
        for kind in ("query", "key"):
            for rounding, side in (("rtz", "left"), ("rne", "right")):
                entry = formulas["rotary_formulae"][phase][
                    f"{kind}/{rounding}_products/{side}"
                ]
                assert entry["positions"] == entry["exact_positions"] == count
                assert entry["different_elements"] == 0
            for variant, side in (
                ("native", "eager"),
                ("rne_products", "compiled_casts"),
            ):
                entry = rotary["results"][phase][f"{variant}/{kind}/{side}"]
                assert entry["positions"] == entry["exact_positions"] == count
                assert entry["different_elements"] == 0
            for variant, side in (
                ("native", "compiled_casts"),
                ("rne_products", "eager"),
            ):
                entry = rotary["results"][phase][f"{variant}/{kind}/{side}"]
                assert entry["positions"] == count
                assert entry["exact_positions"] == (0 if phase == "decode" else 1)
    for name, different in {
        "query_after_rotation": 184437,
        "key_after_rotation": 30984,
        "attention_output": 1335594,
        "gated_attention_output": 1282380,
    }.items():
        assert cut["results"]["decode"][name]["different_elements"] == different
    common = authenticated_read(ROOT / "evidence/rotary-common-rounding-320.json")
    eager_change = authenticated_read(
        ROOT / "evidence/rotary-whole-model-eager-change.json"
    )
    assert common["status"] == "COMPARED_TWO_DECLARED_INTERVENTIONS"
    assert common["observed_rotary"]["calls"] == 1344
    assert common["binding"]["isolated_native_evidence"] == rotary["sha256"]
    assert eager_change["decode"]["1"]["set_exact"] == 316
    assert eager_change["decode"]["full_logits_exact"] == 0
    for phase, count in (("decode", 320), ("prefill", 1)):
        assert common[phase]["positions"] == common[phase]["full_logits_exact"] == count
        for k in ("1", "10", "20"):
            for field in (
                "set_exact",
                "ranked_exact",
                "retained_scores_exact",
                "inclusive_tie_set_exact",
            ):
                assert common[phase][k][field] == count
    build_mode_tables(ROOT)
    profiles = {arm: read(f"{arm}-compiled-profile.json") for arm in ("old", "fixed")}
    groups = {}
    dispatches = []
    for arm, profile in profiles.items():
        by_group = defaultdict(float)
        divisor = 1000 * profile["profile_steps"]
        assert profile["profile_steps"] == 8
        for item in profile["attribution"]["kernel_groups"]:
            group = (
                category(item["kernel"])
                if item["stage"] == "target_body"
                else item["stage"]
            )
            ms = item["kernel_us"] / divisor
            by_group[group] += ms
            dispatches.append(
                {
                    "arm": arm,
                    "group": group,
                    "kernel": item["kernel"],
                    "calls": item["calls"],
                    "milliseconds_per_round": ms,
                }
            )
        total = sum(by_group.values())
        assert abs(total - profile["attribution"]["kernel_us"] / divisor) < 1e-8
        assert (
            sum(x["calls"] for x in profile["attribution"]["kernel_groups"])
            == profile["attribution"]["kernels"]
        )
        groups[arm] = dict(by_group)
    rows = [
        {
            "stage": group,
            "old_ms": groups["old"].get(group),
            "fixed_ms": groups["fixed"].get(group),
            "repair": FIXES[group],
        }
        for group in FIXES
    ]
    for group, label, fix in (
        (
            "target_vocabulary_head",
            "Full BF16 vocabulary head",
            (
                "Two interleaved M4 groups in one HIP launch preserve serial arithmetic without "
                "two launches and concatenation."
            ),
        ),
        (
            "drafter",
            "Drafter, including proposal head",
            "Unchanged drafter; kept separate from target correctness.",
        ),
        (
            "unattributed",
            "GPU work outside attributed model stages",
            "Includes sampling/state bookkeeping/copies; no invented per-operation attribution.",
        ),
    ):
        rows.append(
            {
                "stage": label,
                "old_ms": groups["old"][group],
                "fixed_ms": groups["fixed"][group],
                "repair": fix,
            }
        )
    result = {
        "unit": "sum of GPU kernel durations per profiled round, milliseconds",
        "profile_rounds_per_arm": 8,
        "rows": rows,
        "target_body_ms": {
            a: profiles[a]["attribution"]["stages"]["target_body"]["kernel_us"] / 8000
            for a in profiles
        },
        "all_kernel_ms": {a: sum(groups[a].values()) for a in profiles},
    }
    (ROOT / "stage-times.json").write_text(json.dumps(result, indent=2) + "\n")
    table = [
        "| Compiled GPU stage/group | Old M8 (ms) | Fixed M8 (ms) | Repair / interpretation |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in rows:
        values = [
            "Included in fused group" if row[k] is None else f"{row[k]:.3f}"
            for k in ("old_ms", "fixed_ms")
        ]
        table.append(
            f"| {row['stage']} | {values[0]} | {values[1]} | {row['repair']} |"
        )
    (ROOT / "stage-table.md").write_text("\n".join(table) + "\n")
    with (ROOT / "kernel-dispatches.csv").open("w") as out:
        writer = csv.DictWriter(
            out, fieldnames=list(dispatches[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(dispatches)
    # Retain the original eight-round aggregation, then build the finer table
    # from independently audited individual dispatches and complete rounds.
    (ROOT / "historical-eight-round-stage-table.md").write_text("\n".join(table) + "\n")
    from build_detailed_tables import build

    replacements = build(ROOT)
    speed = read("brief-speed-controls.json")
    speed_table = [
        (
            "| Compiled configuration | Natural responses | Output tokens | Timed "
            "post-first seconds | Median round | Pooled post-first rate |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = {
        "original": "Original, full BF16 target head",
        "initial_repair": "Correctness repair, before performance recovery",
        "first_three_changes": "Repair after first three performance changes",
        "final_four_changes": "Final repair, all four performance changes",
    }
    for name, label in labels.items():
        row = speed["controls"][name]
        passes = row["passes"]
        assert len(passes) == 3 and all(p["finish_reason"] == "stop" for p in passes)
        assert [p["index"] for p in passes] == [1, 2, 3]
        assert row["output_tokens"] == sum(p["output_tokens"] for p in passes)
        assert math.isclose(
            row["median_round_ms"],
            statistics.median(p["steady_median_step_ms"] for p in passes),
        )
        assert math.isclose(
            row["post_first_tps"],
            sum(p["after_first_tokens"] for p in passes)
            / sum(p["after_first_seconds"] for p in passes),
        )
        speed_table.append(
            f"| {label} | 3 | {row['output_tokens']:,} | {row['post_first_seconds']:.3f} | "
            f"{row['median_round_ms']:.3f} ms | {row['post_first_tps']:.3f} tok/s |"
        )
    replacements["{{SPEED_TABLE}}"] = "\n".join(speed_table)
    replacements["{{SPEED_SUMMARY}}"] = (
        "The final control produced 743, 848 and 704 tokens at 86.790, 79.240 and "
        "87.529 tok/s. Its middle response differed from the earlier repaired control "
        "starting at token offset 671. A repeat of the same build reproduced the "
        "earlier 743/790/704 lengths and all three output digests. The cause of this "
        "intermittent continuation difference is unresolved; the table retains the "
        "first complete measurement. The separate forced-token 10K agreement does "
        "not establish repeatability of these natural completions. See "
        "[brief-speed-controls.json](evidence/brief-speed-controls.json)."
    )
    template = ROOT / "report.template.md"
    document = template.read_text()
    for marker, replacement in replacements.items():
        document = document.replace(marker, replacement)
    assert "{{" not in document, "unexpanded report field"
    (ROOT / "REPORT.md").write_text(document)
    summary = read("fixed-compiled-10k-summary.json")
    cross_mode = read("historical-eager-to-compiled-m8.json")
    assert cross_mode["status"] == "COMPARED_SAVED_EVIDENCE"
    assert cross_mode["within_mode_m1_m8_exact"] == {"eager": 10000, "compiled": 10000}
    assert cross_mode["decode"]["positions"] == 10000
    assert cross_mode["prefill"]["positions"] == 23
    assert cross_mode["prefill"]["full_logits_exact"] == 0
    for k, counts in {"1": (9811, 9811), "10": (4845, 756), "20": (2227, 4)}.items():
        assert (
            cross_mode["decode"][k]["set_exact"],
            cross_mode["decode"][k]["ranked_exact"],
        ) == counts
    assert summary["decode"]["positions"] == 10000
    assert summary["prefill"]["positions"] == 23
    for domain in ("decode", "prefill"):
        count = summary[domain]["positions"]
        assert summary[domain]["full_logits_exact"] == count
        for k in ("1", "10", "20"):
            assert all(
                summary[domain][k][field] == count
                for field in (
                    "set_exact",
                    "ranked_exact",
                    "retained_scores_exact",
                    "inclusive_tie_set_exact",
                )
            )
    manifests = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "evidence").glob("*.json"))
    }
    (ROOT / "evidence-sha256.json").write_text(json.dumps(manifests, indent=2) + "\n")
    detailed = json.loads((ROOT / "stage-times.json").read_text())
    print(
        json.dumps(
            {
                "accounting_verified": True,
                "comparison_verified": True,
                "target_body_ms": detailed["target_body_ms"],
                "all_kernel_ms": detailed["all_kernel_ms"],
            }
        )
    )


if __name__ == "__main__":
    main()

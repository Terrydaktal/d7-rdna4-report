"""Render observed activation equality separately from isolated vocabulary tests."""

from build_detailed_tables import authenticated_read


def build(root):
    sources = {
        phase: authenticated_read(root / "evidence" / f"current-mode-boundaries-{phase}.json")
        for phase in ("prefill", "decode")
    }
    assert sources["prefill"]["positions"] == 9
    assert sources["decode"]["positions"] == 320
    indexed = {
        phase: {(row["operation"], tuple(row["owners"])): row for row in source["boundaries"]}
        for phase, source in sources.items()
    }
    keys = list(indexed["decode"])
    keys += [key for key in indexed["prefill"] if key not in indexed["decode"]]
    lines = [
        "# Observed eager/compiled activation boundaries",
        "",
        "Current repaired M8, before the SiLU intervention. Both captures reproduce their",
        "uninstrumented controls. Counts compare complete captured tensor rows exactly;",
        "they are not vocabulary top-20 counts. Inputs can already contain upstream",
        "differences. KV/GDN/conv state and unpaired operations are not certified by this table.",
        "The independently isolated SiLU experiment is described in the report.",
        "",
        "Only unique operation/owner pairs with compatible captured layouts are shown.",
        "Unpaired and ambiguous counts remain in the two source receipts.",
        "",
        "| Logical owner | Operation | Prefill input exact | Prefill output exact | Decode input exact | Decode output exact |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for key in keys:
        cells = []
        for phase in ("prefill", "decode"):
            row = indexed[phase].get(key)
            if row is None:
                cells += ["Unpaired", "Unpaired"]
                continue
            n = row["captured_input_positions"]
            cells.append(f"{row['exact_captured_input_positions']}/{n}" if n else "Unobserved")
            cells.append(f"{row['exact_output_positions']}/{row['positions']}")
        owner = key[1][0].removeprefix("language_model.model.")
        lines.append(f"| `{owner}` | `{key[0]}` | " + " | ".join(cells) + " |")
    (root / "execution-mode-boundaries.md").write_text("\n".join(lines) + "\n")
    serial = authenticated_read(root / "evidence/compiled-m1-m8-boundaries-320.json")
    assert serial["positions"] == 320
    assert len(serial["boundaries"]) == 466
    assert all(r["exact_output_positions"] == r["positions"] == 320 for r in serial["boundaries"])
    serial_lines = [
        "# Observed compiled M1/M8 activation boundaries",
        "",
        "Eight serial M1 observations are concatenated by consecutive logical position",
        "and compared with one M8 group. Both observers reproduce their uninstrumented",
        "controls. The 466 matched boundaries all have exact captured outputs across 320",
        "positions. This does not establish equality of uncaptured KV/GDN/conv state,",
        "nor an isolated old-M8 vocabulary top-20 result. Unpaired operators remain",
        "explicit in the [receipt](evidence/compiled-m1-m8-boundaries-320.json).",
        "",
        "| Logical owner | Operation | Captured input exact | Output exact |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in serial["boundaries"]:
        n = row["captured_input_positions"]
        inputs = f"{row['exact_captured_input_positions']}/{n}" if n else "Unobserved"
        owner = row["owners"][0].removeprefix("language_model.model.")
        serial_lines.append(
            f"| `{owner}` | `{row['operation']}` | {inputs} | "
            f"{row['exact_output_positions']}/{row['positions']} |"
        )
    (root / "compiled-m1-m8-boundaries.md").write_text("\n".join(serial_lines) + "\n")
    return len(keys)

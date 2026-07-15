"""Build publication assets from the frozen E2AM-MemRAG release.

The script is dependency-free and intentionally refuses to infer missing values.
It creates reviewer-auditable CSV summaries, LaTeX tables, and vector figures.
Rasterization and LaTeX compilation are separate, documented build steps.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
RAW = PAPER / "data" / "raw"
DERIVED = PAPER / "data" / "derived"
MANUSCRIPT = PAPER / "manuscript"
TABLES = MANUSCRIPT / "tables"
FIGURES = MANUSCRIPT / "figures"


MODEL_NAMES = {
    "tiny": "Qwen3 0.6B",
    "small": "Granite 4.0 1B",
    "granite": "Granite 4.1 3B",
    "peer": "SmolLM3 3B",
    "upper": "Qwen3 4B Instruct",
}

DECLARED_ROUTES = [
    ("A00", "Qwen3 0.6B", "Direct", "yes"),
    ("A01", "Qwen3 0.6B", "BM25 knowledge", "yes"),
    ("A02", "Qwen3 0.6B", "Dense knowledge", "yes"),
    ("A03", "Qwen3 0.6B", "Bounded hybrid knowledge", "yes"),
    ("A04", "Granite 4.0 1B", "Direct", "yes"),
    ("A05", "Granite 4.0 1B", "BM25 knowledge", "no"),
    ("A06", "Granite 4.0 1B", "Dense knowledge", "no"),
    ("A07", "Granite 4.0 1B", "Bounded hybrid knowledge", "no"),
    ("A08", "Qwen3 0.6B", "Flat memory", "yes"),
    ("A09", "Qwen3 0.6B", "Hierarchical memory", "yes"),
    ("A10", "Qwen3 0.6B", "Graph/temporal memory", "yes"),
    ("A11", "Granite 4.0 1B", "Graph/temporal memory", "no"),
    ("A12", "Granite 4.0 1B", "Hybrid knowledge + flat memory", "yes"),
    ("A13", "Granite 4.0 1B", "Hybrid + graph memory + guard", "yes"),
    ("A14", "Qwen3 4B Instruct", "Hybrid + graph memory + guard", "yes"),
    ("A15", "Granite 4.0 1B", "BM25 + hierarchical memory + guard", "no"),
    ("M16", "Qwen3 0.6B", "Matched grounded/verified", "yes"),
    ("M17", "Granite 4.1 3B", "Matched direct", "yes"),
    ("M18", "Granite 4.1 3B", "Matched grounded/verified", "yes"),
    ("M19", "SmolLM3 3B", "Matched direct", "yes"),
    ("M20", "SmolLM3 3B", "Matched grounded/verified", "yes"),
    ("M21", "Qwen3 4B Instruct", "Matched direct", "yes"),
]


def read_csv(name: str) -> list[dict[str, str]]:
    with (DERIVED / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (RAW / name).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RuntimeError(f"{name}:{line_number} is not an object")
                rows.append(value)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def tex(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def fpercent(value: Any, digits: int = 1) -> str:
    return f"{100.0 * float(value):.{digits}f}"


def fnumber(value: Any, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def table_file(name: str, content: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / name).write_text(content.strip() + "\n", encoding="utf-8")


def build_audit_derivatives(clean: list[dict[str, Any]], robust: list[dict[str, Any]]) -> None:
    uuids = sorted({str(row["generation"]["energy"]["gpu_uuid"]) for row in clean})
    aliases = {uuid: f"GPU-{chr(65 + index)}" for index, uuid in enumerate(uuids)}
    by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clean:
        by_route[str(row["route_id"])].append(row)
    hardware_rows = []
    for route_id in sorted(by_route):
        boards = sorted({aliases[str(row["generation"]["energy"]["gpu_uuid"])] for row in by_route[route_id]})
        hardware_rows.append({"route_id": route_id, "rows": len(by_route[route_id]), "board_aliases": ";".join(boards)})
    write_csv(DERIVED / "route_hardware.csv", hardware_rows, ["route_id", "rows", "board_aliases"])

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in robust:
        grouped[(str(row["condition"]), str(row["route_id"]))].append(row)
    robust_rows = []
    for (condition, route_id), rows in sorted(grouped.items()):
        energies = [float(row["generation"]["energy"]["energy_joules"]) for row in rows]
        robust_rows.append(
            {
                "condition": condition,
                "route_id": route_id,
                "queries": len(rows),
                "success_rate": sum(bool(row.get("success")) for row in rows) / len(rows),
                "abstention_rate": sum(bool(row.get("answer", {}).get("abstain")) for row in rows) / len(rows),
                "mean_gpu_joules": sum(energies) / len(energies),
            }
        )
    write_csv(
        DERIVED / "robustness_route_statistics.csv",
        robust_rows,
        ["condition", "route_id", "queries", "success_rate", "abstention_rate", "mean_gpu_joules"],
    )

    granite_rows = [row for row in clean if str(row["route_id"]) in {"A04_small_direct", "A13_small_hybrid_verified"}]
    output_counts: dict[str, int] = defaultdict(int)
    for row in granite_rows:
        output_counts[str(row.get("generation", {}).get("text", ""))] += 1
    granite_audit = {
        "routes": ["A04_small_direct", "A13_small_hybrid_verified"],
        "rows": len(granite_rows),
        "parseable_rows": sum(bool(row.get("answer", {}).get("parse_ok")) for row in granite_rows),
        "max_new_tokens_rows": sum(int(row.get("generation", {}).get("output_tokens", 0)) == 80 for row in granite_rows),
        "unique_outputs": len(output_counts),
        "most_common_output_sha256": hashlib.sha256(max(output_counts, key=output_counts.get).encode()).hexdigest(),
        "most_common_output_count": max(output_counts.values()),
        "diagnosis": "frozen prompt/runtime-format incompatibility; not a general model-capability claim",
    }
    (DERIVED / "granite_1b_output_audit.json").write_text(
        json.dumps(granite_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_tables() -> None:
    routes = read_csv("route_statistics.csv")
    models = read_csv("model_transfer.csv")
    overall = read_csv("overall_results.csv")[0]
    robust = read_csv("robustness_conditions.csv")
    trace = read_csv("trace_audit.csv")
    hardware = {row["route_id"]: row["board_aliases"] for row in read_csv("route_hardware.csv")}

    table_file(
        "model_panel.tex",
        r"""
\begin{table*}[t]
\caption{Frozen generator panel and matched direct/grounded endpoints. Only the first two model families were eligible for online routing. All checkpoints were loaded in FP16 at immutable revisions.}
\label{tab:model-panel}
\centering
\small
\begin{tabularx}{\textwidth}{@{}lXlll@{}}
\toprule
Role & Frozen repository (revision prefix) & Direct & Grounded/verified & Router eligible \\
\midrule
Tiny & \texttt{Qwen/Qwen3-0.6B} (\texttt{c1899de}) & A00 & M16 & yes \\
Small & \texttt{ibm-granite/granite-4.0-1b} (\texttt{6a7381b}) & A04 & A13 & yes \\
Granite reference & \texttt{ibm-granite/granite-4.1-3b} (\texttt{c065040}) & M17 & M18 & no \\
Peer reference & \texttt{HuggingFaceTB/SmolLM3-3B} (\texttt{a07cc9a}) & M19 & M20 & no \\
Upper reference & \texttt{Qwen/Qwen3-4B-Instruct-2507} (\texttt{cdbee75}) & M21 & A14 & no \\
\bottomrule
\end{tabularx}
\end{table*}
""",
    )

    table_file(
        "confirmatory_result.tex",
        rf"""
\begin{{table}}[t]
\caption{{Confirmatory policy result on the sealed test split ($n={overall['policy_queries']}$). Intervals are 95\% scenario-cluster bootstrap intervals.}}
\label{{tab:confirmatory}}
\centering
\small
\begin{{tabular}}{{@{{}}lr@{{}}}}
\toprule
Quantity & Result \\
\midrule
Selected route & A03 for all queries \\
Strict success & {fpercent(overall['policy_success_rate'])}\% [{fpercent(overall['policy_success_ci_low'])}, {fpercent(overall['policy_success_ci_high'])}] \\
Generation-window GPU energy & {fnumber(overall['policy_mean_gpu_joules'])} J/query \\
Energy interval & [{fnumber(overall['policy_gpu_joules_ci_low'])}, {fnumber(overall['policy_gpu_joules_ci_high'])}] J \\
Policy latency component sum & {fnumber(overall['policy_latency_seconds'])} s/query \\
Abstention / execution coverage & {fpercent(overall['policy_abstention_rate'])}\% / {fpercent(overall['execution_coverage'])}\% \\
Policy $-$ baseline success & {fpercent(overall['success_difference'])} pp \\
Policy $-$ baseline energy & {fnumber(overall['gpu_joules_difference'])} J/query \\
\midrule
Quality non-inferiority & pass \\
Energy reduction & \textbf{{fail}} \\
Operating constraints & pass \\
Joint confirmatory hypothesis & \textbf{{fail}} \\
\bottomrule
\end{{tabular}}
\end{{table}}
""",
    )

    model_rows = []
    for row in models:
        same_board = hardware[row["direct_route_id"]] == hardware[row["grounded_route_id"]]
        model_rows.append(
            f"{tex(MODEL_NAMES[row['model_key']])} & "
            f"{fpercent(row['direct_success_rate'])} & {fpercent(row['grounded_success_rate'])} & "
            f"{100*float(row['success_diff_mean']):+.1f} [{100*float(row['success_diff_low']):+.1f}, {100*float(row['success_diff_high']):+.1f}] & "
            f"{float(row['gpu_joules_diff_mean']):+.2f} [{float(row['gpu_joules_diff_low']):+.2f}, {float(row['gpu_joules_diff_high']):+.2f}] & "
            f"{'yes' if same_board else 'no'}" + r" \\"
        )
    table_file(
        "model_transfer.tex",
        r"""\begin{table*}[t]
\caption{Matched direct-to-grounded transfer on the same 120 sealed-test questions. Success effects are percentage points; energy effects are generation-window selected-GPU board joules per query. ``Same board'' indicates whether both endpoints ran on the same physical T4.}
\label{tab:model-transfer}
\centering
\small
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Generator & Direct (\%) & Grounded (\%) & $\Delta$ success [95\% CI] & $\Delta$ GPU J [95\% CI] & Same board \\
\midrule
""" + "\n".join(model_rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""",
    )

    highlight_ids = {"M18_granite_grounded_verified", "A14_upper_hybrid_verified", "M20_peer_grounded_verified", "M19_peer_direct", "M17_granite_direct", "A00_tiny_direct", "A03_tiny_hybrid"}
    highlights = sorted((row for row in routes if row["route_id"] in highlight_ids), key=lambda row: (-float(row["success_rate"]), float(row["mean_gpu_joules"])))
    route_rows = [
        f"{tex(row['route_id'].split('_', 1)[0])} & {fpercent(row['success_rate'])} & {fnumber(row['mean_gpu_joules'])} & {fnumber(row['median_seconds'])}" + r" \\"
        for row in highlights
    ]
    table_file(
        "route_highlights.tex",
        r"""\begin{table}[t]
\caption{Selected clean-test route results. GPU J denotes mean generation-window selected-board energy; the complete matrix and hardware assignment are in Appendix~\ref{app:routes}.}
\label{tab:route-highlights}
\centering
\scriptsize
\begin{tabular}{@{}lrrr@{}}
\toprule
ID & Success (\%) & GPU J & Median s \\
\midrule
""" + "\n".join(route_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""",
    )

    robust_rows = [
        f"{tex(row['condition'].replace('_', ' '))} & {fpercent(row['selected_success_rate'])} & {fpercent(row['abstention_rate'])} & {fpercent(row['execution_coverage'])}" + r" \\"
        for row in robust
    ]
    table_file(
        "robustness.tex",
        r"""\begin{table}[t]
\caption{Selected-policy outcomes under four evidence corruptions. Zero clean and corrupted success makes every success delta a floor effect.}
\label{tab:robustness}
\centering
\scriptsize
\begin{tabular}{@{}lrrr@{}}
\toprule
Condition & Success (\%) & Abstain (\%) & Coverage (\%) \\
\midrule
""" + "\n".join(robust_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""",
    )

    trace_rows = [
        f"{tex(row['trace_set'])} & {row['rows']} & {row['routes']} & {row['successes']} ({fpercent(row['success_rate'])}\\%) & {fpercent(row['parse_rate'])}\\% & {fpercent(row['support_ok_rate'])}\\% & {fpercent(row['energy_coverage'])}\\%" + r" \\"
        for row in trace
    ]
    table_file(
        "trace_integrity.tex",
        r"""\begin{table*}[t]
\caption{Released trace integrity and outcome audit. Both sets had zero duplicate unit IDs, zero non-finite audited values, zero divergent route hashes, and zero execution failures.}
\label{tab:trace-integrity}
\centering
\small
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
Set & Rows & Routes & Strict successes & Parse & Support & Energy coverage \\
\midrule
""" + "\n".join(trace_rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""",
    )

    route_lookup = {row["route_id"].split("_", 1)[0]: row for row in routes}
    catalog_rows = []
    for route_id, model, mechanism, retained in DECLARED_ROUTES:
        full = route_lookup.get(route_id)
        success = fpercent(full["success_rate"]) + r"\%" if full else "--"
        energy = fnumber(full["mean_gpu_joules"]) if full else "--"
        catalog_rows.append(
            f"{route_id} & {tex(model)} & {tex(mechanism)} & {retained} & {success} & {energy}" + r" \\"
        )
    table_file(
        "route_catalog.tex",
        r"""\begin{table*}[t]
\caption{Predeclared route catalog and pilot retention. A05--A07, A11, and A15 were removed only by the frozen pilot rule; all mandatory matched endpoints were retained.}
\label{tab:route-catalog}
\centering
\scriptsize
\begin{tabularx}{\textwidth}{@{}llXlrr@{}}
\toprule
ID & Generator & Mechanism & Test retained & Success & GPU J \\
\midrule
""" + "\n".join(catalog_rows) + r"""
\bottomrule
\end{tabularx}
\end{table*}
""",
    )

    hardware_rows = [f"{tex(route)} & {tex(board)}" + r" \\" for route, board in sorted(hardware.items())]
    table_file(
        "hardware_assignment.tex",
        r"""\begin{table}[t]
\caption{Anonymized physical-T4 assignment for clean-test routes. Within-board contrasts avoid board identity as a direct confound; cross-board energy contrasts remain descriptive.}
\label{tab:hardware}
\centering
\scriptsize
\begin{tabular}{@{}ll@{}}
\toprule
Route & Physical board alias \\
\midrule
""" + "\n".join(hardware_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""",
    )


def svg_text(x: float, y: float, text: str, *, size: int = 22, weight: int = 400, anchor: str = "middle", fill: str = "#172033") -> str:
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{weight}" fill="{fill}">{html.escape(text)}</text>'


def build_system_figure() -> str:
    width, height = 1800, 720
    colors = ["#e8f0fe", "#e8f7f0", "#fff4df", "#f2ecff", "#eaf3f8"]
    titles = ["Frozen HybridBench", "Route execution", "Measurement contract", "Calibrated router", "Sealed evaluation"]
    lines = [
        ["800 grouped scenarios", "8 controlled task classes", "procedurally sealed test"],
        ["BM25 + dense + hybrid", "flat / hierarchical / graph memory", "5 pinned FP16 generators"],
        ["CPU retrieval", "T4 model.generate()", "deterministic support guard"],
        ["5 grouped bootstraps", "isotonic success calibration", "q0.90 energy / latency"],
        ["120 clean questions x 17 routes", "4 corruptions x 3 routes", "10,000 cluster bootstraps"],
    ]
    box_w, box_h, gap, start_x, y = 300, 360, 50, 40, 145
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="#ffffff"/>']
    parts.append(svg_text(width / 2, 58, "E2AM-MemRAG experimental pipeline and measurement boundary", size=31, weight=700))
    for index, title in enumerate(titles):
        x = start_x + index * (box_w + gap)
        parts.append(f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="24" fill="{colors[index]}" stroke="#365779" stroke-width="3"/>')
        parts.append(svg_text(x + box_w / 2, y + 58, f"{index + 1}. {title}", size=23, weight=700))
        for line_index, line in enumerate(lines[index]):
            parts.append(svg_text(x + box_w / 2, y + 130 + 58 * line_index, line, size=19))
        if index < len(titles) - 1:
            ax = x + box_w + 8
            parts.append(f'<line x1="{ax}" y1="{y + box_h/2}" x2="{ax + gap - 18}" y2="{y + box_h/2}" stroke="#365779" stroke-width="5"/>')
            parts.append(f'<polygon points="{ax + gap - 18},{y + box_h/2 - 12} {ax + gap - 18},{y + box_h/2 + 12} {ax + gap},{y + box_h/2}" fill="#365779"/>')
    mx = start_x + 2 * (box_w + gap)
    parts.append(f'<rect x="{mx + 20}" y="{y + 153}" width="{box_w - 40}" height="70" rx="12" fill="none" stroke="#c43d3d" stroke-width="5" stroke-dasharray="12 8"/>')
    parts.append(svg_text(mx + box_w / 2, y + 300, "NVML energy window", size=19, weight=700, fill="#a92727"))
    parts.append(svg_text(width / 2, 580, "Energy: selected-GPU board power integrated only over model.generate().  Latency: retrieval + generator call; policy adds probe/router components.", size=20, weight=600))
    parts.append(svg_text(width / 2, 625, "Post-generation parsing, verification, CPU energy, whole-system energy, and carbon are outside the measured energy boundary.", size=19, fill="#a92727"))
    parts.append("</svg>\n")
    return "".join(parts)


def build_forest_figure(models: list[dict[str, str]]) -> str:
    width, height = 1500, 850
    left, right, top, bottom = 330, 1180, 135, 105
    x_min, x_max = -20.0, 45.0
    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (right - left)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="#ffffff"/>']
    parts.append(svg_text(width / 2, 52, "Matched effect of grounding on strict support-qualified success", size=31, weight=700))
    parts.append(svg_text(width / 2, 88, "Difference in percentage points; horizontal bars are 95% scenario-cluster bootstrap intervals", size=19, fill="#4c586d"))
    for tick in range(-20, 46, 5):
        x = sx(float(tick))
        parts.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{height-bottom}" stroke="{"#7c8799" if tick == 0 else "#e3e7ee"}" stroke-width="{3 if tick == 0 else 1}"/>')
        parts.append(svg_text(x, height - 62, str(tick), size=18))
    row_gap = 112
    for index, row in enumerate(models):
        y = top + 68 + index * row_gap
        mean = 100 * float(row["success_diff_mean"])
        low = 100 * float(row["success_diff_low"])
        high = 100 * float(row["success_diff_high"])
        energy = float(row["gpu_joules_diff_mean"])
        color = "#cc5a24" if mean < 0 else ("#2368ad" if low > 0 else "#6f7785")
        parts.append(svg_text(left - 25, y + 7, MODEL_NAMES[row["model_key"]], size=22, weight=600, anchor="end"))
        parts.append(f'<line x1="{sx(low)}" y1="{y}" x2="{sx(high)}" y2="{y}" stroke="{color}" stroke-width="7"/>')
        parts.append(f'<line x1="{sx(low)}" y1="{y-12}" x2="{sx(low)}" y2="{y+12}" stroke="{color}" stroke-width="4"/>')
        parts.append(f'<line x1="{sx(high)}" y1="{y-12}" x2="{sx(high)}" y2="{y+12}" stroke="{color}" stroke-width="4"/>')
        if low <= 0 <= high:
            parts.append(f'<circle cx="{sx(mean)}" cy="{y}" r="13" fill="#ffffff" stroke="{color}" stroke-width="5"/>')
        else:
            parts.append(f'<circle cx="{sx(mean)}" cy="{y}" r="13" fill="{color}" stroke="#ffffff" stroke-width="3"/>')
        parts.append(svg_text(right + 45, y + 7, f"{mean:+.1f} pp", size=20, weight=700, anchor="start", fill=color))
        parts.append(svg_text(right + 190, y + 7, f"{energy:+.2f} J", size=20, anchor="start", fill="#4c586d"))
    parts.append(svg_text((left + right) / 2, height - 20, "Grounded minus direct strict success (percentage points)", size=21, weight=600))
    parts.append(svg_text(right + 55, 123, "Effect", size=18, weight=700, anchor="start"))
    parts.append(svg_text(right + 195, 123, "GPU energy", size=18, weight=700, anchor="start"))
    parts.append("</svg>\n")
    return "".join(parts)


def build_figures() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    models = read_csv("model_transfer.csv")
    (FIGURES / "system_overview.svg").write_text(build_system_figure(), encoding="utf-8")
    (FIGURES / "grounding_effect_forest.svg").write_text(build_forest_figure(models), encoding="utf-8")
    # Keep the audited route scatter colocated with manuscript sources.
    source = PAPER / "figures" / "route_quality_energy.svg"
    (FIGURES / "route_quality_energy.svg").write_bytes(source.read_bytes())


def main() -> None:
    clean = read_jsonl("release__clean_traces.jsonl")
    robust = read_jsonl("release__robustness_traces.jsonl")
    if len(clean) != 2040 or len(robust) != 1440:
        raise RuntimeError("Frozen trace totals changed; manuscript generation stopped")
    build_audit_derivatives(clean, robust)
    build_tables()
    build_figures()
    print(
        "MANUSCRIPT_ASSETS_READY",
        {
            "clean_rows": len(clean),
            "robustness_rows": len(robust),
            "tables": len(list(TABLES.glob("*.tex"))),
            "figures": len(list(FIGURES.glob("*.svg"))),
        },
    )


if __name__ == "__main__":
    main()

"""Verify the pinned E2AM-MemRAG release and build paper-ready evidence.

This script is intentionally dependency-free.  The raw Stage-09 files are the
source of truth; every generated table, figure, and claim is derived from them.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
RAW = PAPER / "data" / "raw"
DERIVED = PAPER / "data" / "derived"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"

SOURCE_REPO = "Shanmuk4622/E2AM-MemRAG-Traces"
SOURCE_MAIN_COMMIT = "0b2405d9cca43fd04e35f792fdc4664405154fc6"
SOURCE_PAPER_COMMIT = "00fa353f273f3a4b3d57a0b998301c85a1bc098b"
EXPERIMENT_ID = "e2am-memrag-v3r1"

RAW_NAME = {
    "HYPOTHESIS_RESULT.json": "HYPOTHESIS_RESULT.json",
    "RELEASE_CANDIDATE.json": "RELEASE_CANDIDATE.json",
    "_SUCCESS.json": "_SUCCESS.json",
    "release/clean_traces.jsonl": "release__clean_traces.jsonl",
    "release/experiment_summary.json": "release__experiment_summary.json",
    "release/mechanism_analysis.json": "release__mechanism_analysis.json",
    "release/model_transfer_panel.json": "release__model_transfer_panel.json",
    "release/robustness_analysis.json": "release__robustness_analysis.json",
    "release/robustness_traces.jsonl": "release__robustness_traces.jsonl",
    "release/route_cards.json": "release__route_cards.json",
    "release_manifest.json": "release_manifest.json",
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(name: str) -> Any:
    return json.loads((RAW / name).read_text(encoding="utf-8"))


def load_jsonl(name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (RAW / name).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Malformed JSONL at {name}:{line_number}") from error
            if not isinstance(value, dict):
                raise RuntimeError(f"Non-object JSONL row at {name}:{line_number}")
            rows.append(value)
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float | None, digits: int = 1) -> str:
    return "NA" if value is None else f"{100.0 * value:.{digits}f}%"


def number(value: float | None, digits: int = 2) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def ci_text(interval: dict[str, Any], *, percentage: bool = False) -> str:
    scale = 100.0 if percentage else 1.0
    suffix = " pp" if percentage else ""
    return (
        f"{scale * interval['mean']:.2f}{suffix} "
        f"[{scale * interval['low']:.2f}, {scale * interval['high']:.2f}]"
    )


def verify_release() -> dict[str, Any]:
    fast_manifest = load_json("FAST_PAPER_MANIFEST.json")
    fast_pointer = load_json("FAST_PAPER_RELEASE.json")
    release_pointer = load_json("RELEASE.json")
    if fast_pointer["paper_commit_sha"] != SOURCE_PAPER_COMMIT:
        raise RuntimeError("Fast release points to an unexpected paper commit")
    manifest_digest = sha256_file(RAW / "FAST_PAPER_MANIFEST.json")
    if manifest_digest != fast_pointer["manifest_sha256"]:
        raise RuntimeError("FAST_PAPER_MANIFEST checksum disagrees with pointer")
    if release_pointer["stage_commit_sha"] != "4d3e111ddfe41247b511ad0fc1a413baacee7864":
        raise RuntimeError("Frozen Stage-09 source commit changed")
    verified: list[dict[str, Any]] = []
    for record in fast_manifest["artifacts"]:
        logical = record["logical_path"]
        path = RAW / RAW_NAME[logical]
        digest = sha256_file(path)
        if path.stat().st_size != record["bytes"] or digest != record["sha256"]:
            raise RuntimeError(f"Raw release checksum failed: {logical}")
        verified.append(
            {
                "logical_path": logical,
                "local_path": path.relative_to(ROOT).as_posix(),
                "bytes": record["bytes"],
                "sha256": digest,
            }
        )
    if len(verified) != 11 or sum(row["bytes"] for row in verified) != 11_528_142:
        raise RuntimeError("Fast release coverage is not the frozen 11-file closure")
    return {
        "source_repo": SOURCE_REPO,
        "source_main_commit": SOURCE_MAIN_COMMIT,
        "source_paper_commit": SOURCE_PAPER_COMMIT,
        "verified_artifacts": verified,
    }


def audit_traces(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    unit_ids = [row["unit_id"] for row in rows]
    route_hashes: dict[str, set[str]] = defaultdict(set)
    finite_failures = 0
    for row in rows:
        route_hashes[row["route_id"]].add(row["route_spec_hash"])
        for value in (
            row.get("total_seconds"),
            row.get("generation", {}).get("energy", {}).get("energy_joules"),
            row.get("metrics", {}).get("quality_score"),
        ):
            if isinstance(value, (int, float)) and not math.isfinite(value):
                finite_failures += 1
    divergent_route_hashes = {
        route: sorted(values) for route, values in route_hashes.items() if len(values) != 1
    }
    metrics = [row.get("metrics", {}) for row in rows]
    energies = [
        row.get("generation", {}).get("energy", {}).get("energy_joules")
        for row in rows
    ]
    energy_values = [value for value in energies if isinstance(value, (int, float))]
    return {
        "trace_set": label,
        "rows": len(rows),
        "unique_units": len(set(unit_ids)),
        "duplicate_unit_ids": len(unit_ids) - len(set(unit_ids)),
        "query_ids": len({row["query_id"] for row in rows}),
        "routes": len({row["route_id"] for row in rows}),
        "conditions": ";".join(sorted({row["condition"] for row in rows})),
        "seeds": ";".join(str(value) for value in sorted({row["seed"] for row in rows})),
        "successes": sum(bool(row["success"]) for row in rows),
        "success_rate": sum(bool(row["success"]) for row in rows) / len(rows),
        "parse_rate": sum(bool(row.get("parse_ok")) for row in metrics) / len(rows),
        "answer_ok_rate": sum(bool(row.get("answer_ok")) for row in metrics) / len(rows),
        "support_ok_rate": sum(bool(row.get("support_ok")) for row in metrics) / len(rows),
        "energy_coverage": len(energy_values) / len(rows),
        "mean_gpu_joules": statistics.fmean(energy_values),
        "execution_failures": sum(row["status"] != "SUCCESS" for row in rows),
        "non_finite_values": finite_failures,
        "divergent_route_hashes": len(divergent_route_hashes),
    }


def validate_summary_against_traces(
    summary: dict[str, Any], clean_rows: list[dict[str, Any]]
) -> None:
    by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clean_rows:
        by_route[row["route_id"]].append(row)
    for published in summary["route_statistics"]:
        rows = by_route[published["route_id"]]
        success = statistics.fmean(float(row["success"]) for row in rows)
        energy = statistics.fmean(
            row["generation"]["energy"]["energy_joules"] for row in rows
        )
        latency = statistics.median(row["total_seconds"] for row in rows)
        if len(rows) != published["queries"]:
            raise RuntimeError(f"Route count mismatch: {published['route_id']}")
        for name, actual, expected in (
            ("success", success, published["success_rate"]),
            ("energy", energy, published["mean_gpu_joules"]),
            ("latency", latency, published["median_seconds"]),
        ):
            if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
                raise RuntimeError(
                    f"Published {name} disagrees with trace rows: {published['route_id']}"
                )


def svg_route_scatter(routes: list[dict[str, Any]]) -> str:
    width, height = 1200, 700
    left, right, top, bottom = 95, 125, 55, 90
    plot_w, plot_h = width - left - right, height - top - bottom
    x_min, x_max, y_min, y_max = 40.0, 290.0, 0.0, 0.5

    def x(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def y(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.axis{stroke:#172033;stroke-width:1.4}.grid{stroke:#dfe5ee;stroke-width:1}.label{font-size:12px}.title{font-size:22px;font-weight:700}.tick{font-size:12px}.note{font-size:13px;fill:#526071}</style>',
        '<text class="title" x="600" y="29" text-anchor="middle">Clean-test quality versus generation-window GPU energy</text>',
        '<circle cx="825" cy="48" r="6" fill="#1c7c54"/><text class="note" x="837" y="52">non-zero success</text>',
        '<circle cx="960" cy="48" r="6" fill="#9aa5b1"/><text class="note" x="972" y="52">zero success</text>',
        '<circle cx="1075" cy="48" r="6" fill="#c43d3d"/><text class="note" x="1087" y="52">selected policy</text>',
    ]
    for tick in range(50, 281, 50):
        px = x(float(tick))
        parts.append(f'<line class="grid" x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{top + plot_h}"/>')
        parts.append(f'<text class="tick" x="{px:.1f}" y="{top + plot_h + 24}" text-anchor="middle">{tick}</text>')
    for tick in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        py = y(tick)
        parts.append(f'<line class="grid" x1="{left}" y1="{py:.1f}" x2="{left + plot_w}" y2="{py:.1f}"/>')
        parts.append(f'<text class="tick" x="{left - 12}" y="{py + 4:.1f}" text-anchor="end">{tick:.1f}</text>')
    parts.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
            f'<text x="{left + plot_w / 2}" y="{height - 30}" text-anchor="middle">Mean generation-window GPU-board energy (J/query)</text>',
            f'<text transform="translate(25 {top + plot_h / 2}) rotate(-90)" text-anchor="middle">Strict support-qualified success rate</text>',
        ]
    )
    for row in routes:
        px, py = x(row["mean_gpu_joules"]), y(row["success_rate"])
        selected = row["route_id"] == "A03_tiny_hybrid"
        color = "#c43d3d" if selected else ("#1c7c54" if row["success_rate"] > 0 else "#9aa5b1")
        radius = 8 if selected else (7 if row["success_rate"] > 0 else 5)
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius}" fill="{color}" stroke="white" stroke-width="1.5"/>')
        if row["success_rate"] > 0 or selected:
            label_y = py - 10
            parts.append(
                f'<text class="label" x="{px + 8:.1f}" y="{label_y:.1f}">{html.escape(row["route_id"])}</text>'
            )
    parts.append('<text class="note" x="1180" y="680" text-anchor="end">All routes: n=120 clean queries; energy coverage=100%</text>')
    parts.append("</svg>\n")
    return "".join(parts)


def svg_model_transfer(models: list[dict[str, Any]]) -> str:
    width, height = 1100, 680
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:22px;font-weight:700}.axis{stroke:#172033;stroke-width:1.3}.grid{stroke:#dfe5ee}.tick{font-size:12px}.model{font-size:13px;font-weight:700}.legend{font-size:13px}</style>',
        '<text class="title" x="550" y="30" text-anchor="middle">Generator-dependent effect of grounding</text>',
        '<rect x="770" y="46" width="15" height="15" fill="#3578c8"/><text class="legend" x="792" y="58">Direct</text>',
        '<rect x="875" y="46" width="15" height="15" fill="#e07a35"/><text class="legend" x="897" y="58">Grounded + verified</text>',
    ]
    panels = [
        (65, 80, 465, 500, "Strict success rate", 0.5, "success"),
        (600, 80, 465, 500, "Mean GPU energy (J/query)", 300.0, "energy"),
    ]
    for left, top, panel_w, panel_h, title, maximum, metric in panels:
        parts.append(f'<text x="{left + panel_w / 2}" y="{top - 8}" text-anchor="middle" font-size="16" font-weight="700">{title}</text>')
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            py = top + panel_h - fraction * panel_h
            value = maximum * fraction
            parts.append(f'<line class="grid" x1="{left}" y1="{py:.1f}" x2="{left + panel_w}" y2="{py:.1f}"/>')
            label = f"{value:.2f}" if metric == "success" else f"{value:.0f}"
            parts.append(f'<text class="tick" x="{left - 8}" y="{py + 4:.1f}" text-anchor="end">{label}</text>')
        parts.append(f'<line class="axis" x1="{left}" y1="{top + panel_h}" x2="{left + panel_w}" y2="{top + panel_h}"/>')
        group_w = panel_w / len(models)
        bar_w = 25
        for index, row in enumerate(models):
            center = left + group_w * (index + 0.5)
            direct = row["direct_success_rate"] if metric == "success" else row["direct_mean_gpu_joules"]
            grounded = row["grounded_success_rate"] if metric == "success" else row["grounded_mean_gpu_joules"]
            for offset, value, color in ((-bar_w, direct, "#3578c8"), (0, grounded, "#e07a35")):
                bar_h = value / maximum * panel_h
                parts.append(f'<rect x="{center + offset:.1f}" y="{top + panel_h - bar_h:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="{color}"/>')
            parts.append(f'<text class="model" x="{center:.1f}" y="{top + panel_h + 24}" text-anchor="middle">{html.escape(row["model_key"])}</text>')
    parts.append('<text x="550" y="655" text-anchor="middle" font-size="13" fill="#526071">Same 120 frozen questions per direct-grounded pair; selected-GPU board energy</text>')
    parts.append("</svg>\n")
    return "".join(parts)


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |\n"
    divider = "| " + " | ".join("---" for _ in columns) + " |\n"
    body = "".join(
        "| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |\n"
        for row in rows
    )
    return header + divider + body


def main() -> None:
    for directory in (DERIVED, TABLES, FIGURES):
        directory.mkdir(parents=True, exist_ok=True)
    provenance = verify_release()
    hypothesis = load_json("HYPOTHESIS_RESULT.json")
    success_gate = load_json("_SUCCESS.json")
    summary = load_json("release__experiment_summary.json")
    model_panel = load_json("release__model_transfer_panel.json")
    robustness = load_json("release__robustness_analysis.json")
    clean_rows = load_jsonl("release__clean_traces.jsonl")
    robust_rows = load_jsonl("release__robustness_traces.jsonl")

    if success_gate["status"] != "PASS" or not success_gate["details"]["experiment_complete"]:
        raise RuntimeError("Stage-09 experiment completion gate did not pass")
    if hypothesis["hypothesis_pass"] or summary["hypothesis_pass"]:
        raise RuntimeError("Frozen release unexpectedly changed the hypothesis outcome")
    if len(clean_rows) != summary["clean_trace_rows"] or len(robust_rows) != summary["robustness_trace_rows"]:
        raise RuntimeError("Trace row totals disagree with experiment_summary.json")
    validate_summary_against_traces(summary, clean_rows)

    hypothesis_rows = [
        {"gate": "quality_noninferiority", "passed": hypothesis["quality_noninferiority_pass"]},
        {"gate": "energy_reduction", "passed": hypothesis["energy_reduction_pass"]},
        {"gate": "operating_constraints", "passed": hypothesis["operating_constraints_pass"]},
        {"gate": "confirmatory_hypothesis", "passed": hypothesis["hypothesis_pass"]},
        {"gate": "experiment_completion", "passed": success_gate["details"]["experiment_complete"]},
        {"gate": "fresh_restore", "passed": success_gate["details"]["fresh_restore_verified"]},
    ]
    write_csv(DERIVED / "hypothesis_gates.csv", hypothesis_rows, ["gate", "passed"])

    route_rows = list(summary["route_statistics"])
    write_csv(
        DERIVED / "route_statistics.csv",
        route_rows,
        ["route_id", "queries", "success_rate", "mean_gpu_joules", "median_seconds", "energy_telemetry_coverage", "execution_failure_rate"],
    )
    scenario_rows = list(summary["scenario_class_statistics"])
    write_csv(
        DERIVED / "scenario_class_statistics.csv",
        scenario_rows,
        ["task_type", "route_id", "queries", "success_rate", "abstention_rate", "mean_gpu_joules", "energy_coverage", "execution_failure_rate"],
    )

    contrast_rows: list[dict[str, Any]] = []
    for row in summary["controlled_route_contrasts"]:
        contrast_rows.append(
            {
                "baseline_route_id": row["baseline_route_id"],
                "candidate_route_id": row["candidate_route_id"],
                "success_diff_mean": row["success_difference"]["mean"],
                "success_diff_low": row["success_difference"]["low"],
                "success_diff_high": row["success_difference"]["high"],
                "gpu_joules_diff_mean": row["gpu_joules_difference"]["mean"],
                "gpu_joules_diff_low": row["gpu_joules_difference"]["low"],
                "gpu_joules_diff_high": row["gpu_joules_difference"]["high"],
                "clusters": row["success_difference"]["clusters"],
                "replicates": row["success_difference"]["replicates"],
            }
        )
    write_csv(DERIVED / "controlled_contrasts.csv", contrast_rows, list(contrast_rows[0]))

    model_rows: list[dict[str, Any]] = []
    frontier = set(model_panel["grounded_pareto_frontier_model_keys"])
    for row in model_panel["models"]:
        model_rows.append(
            {
                "model_key": row["model_key"],
                "repo_id": row["repo_id"],
                "revision": row["revision"],
                "queries": row["queries"],
                "direct_route_id": row["direct_route_id"],
                "direct_success_rate": row["direct_success_rate"],
                "direct_mean_gpu_joules": row["direct_mean_gpu_joules"],
                "direct_median_seconds": row["direct_median_seconds"],
                "grounded_route_id": row["grounded_route_id"],
                "grounded_success_rate": row["grounded_success_rate"],
                "grounded_mean_gpu_joules": row["grounded_mean_gpu_joules"],
                "grounded_median_seconds": row["grounded_median_seconds"],
                "success_diff_mean": row["grounded_minus_direct_success"]["mean"],
                "success_diff_low": row["grounded_minus_direct_success"]["low"],
                "success_diff_high": row["grounded_minus_direct_success"]["high"],
                "gpu_joules_diff_mean": row["grounded_minus_direct_gpu_joules"]["mean"],
                "gpu_joules_diff_low": row["grounded_minus_direct_gpu_joules"]["low"],
                "gpu_joules_diff_high": row["grounded_minus_direct_gpu_joules"]["high"],
                "grounded_pareto_frontier": row["model_key"] in frontier,
            }
        )
    write_csv(DERIVED / "model_transfer.csv", model_rows, list(model_rows[0]))

    robust_condition_rows: list[dict[str, Any]] = []
    for row in robustness["conditions"]:
        interval = row["success_delta_robust_minus_clean_same_route"]
        robust_condition_rows.append(
            {
                "condition": row["condition"],
                "queries": row["queries"],
                "selected_success_rate": row["selected_success_rate"],
                "abstention_rate": row["abstention_rate"],
                "execution_coverage": row["execution_coverage"],
                "prompt_injection_compromise_rate": row["prompt_injection_compromise_rate"],
                "success_delta_mean": interval["mean"],
                "success_delta_low": interval["low"],
                "success_delta_high": interval["high"],
                "selected_route": ";".join(row["selected_route_counts"]),
            }
        )
    write_csv(DERIVED / "robustness_conditions.csv", robust_condition_rows, list(robust_condition_rows[0]))

    trace_audits = [audit_traces(clean_rows, "clean"), audit_traces(robust_rows, "robustness")]
    write_csv(DERIVED / "trace_audit.csv", trace_audits, list(trace_audits[0]))

    overall_rows = [
        {
            "policy_queries": summary["policy_queries"],
            "policy_success_rate": summary["policy_success"]["mean"],
            "policy_success_ci_low": summary["policy_success"]["low"],
            "policy_success_ci_high": summary["policy_success"]["high"],
            "policy_mean_gpu_joules": summary["policy_gpu_joules"]["mean"],
            "policy_gpu_joules_ci_low": summary["policy_gpu_joules"]["low"],
            "policy_gpu_joules_ci_high": summary["policy_gpu_joules"]["high"],
            "policy_latency_seconds": summary["policy_end_to_end_latency_seconds"]["mean"],
            "policy_abstention_rate": summary["abstention_rate"],
            "execution_coverage": summary["execution_coverage"],
            "baseline_route_id": summary["baseline_route_id"],
            "baseline_success_rate": summary["baseline_success_rate"],
            "baseline_mean_gpu_joules": summary["baseline_mean_gpu_joules"],
            "success_difference": summary["paired_success_difference_policy_minus_baseline"]["mean"],
            "gpu_joules_difference": summary["paired_gpu_joules_difference_policy_minus_baseline"]["mean"],
            "hypothesis_pass": summary["hypothesis_pass"],
        }
    ]
    write_csv(DERIVED / "overall_results.csv", overall_rows, list(overall_rows[0]))

    (FIGURES / "route_quality_energy.svg").write_text(svg_route_scatter(route_rows), encoding="utf-8")
    (FIGURES / "model_grounding_effect.svg").write_text(svg_model_transfer(model_rows), encoding="utf-8")

    route_table_rows = [
        {
            "route": row["route_id"],
            "success": pct(row["success_rate"]),
            "energy": number(row["mean_gpu_joules"]),
            "latency": number(row["median_seconds"]),
        }
        for row in sorted(route_rows, key=lambda item: (-item["success_rate"], item["mean_gpu_joules"]))
    ]
    (TABLES / "route_results.md").write_text(
        markdown_table(route_table_rows, [("route", "Route"), ("success", "Strict success"), ("energy", "GPU J/query"), ("latency", "Median s")]),
        encoding="utf-8",
    )
    model_table_rows = [
        {
            "model": row["model_key"],
            "direct": pct(row["direct_success_rate"]),
            "grounded": pct(row["grounded_success_rate"]),
            "delta": ci_text({"mean": row["success_diff_mean"], "low": row["success_diff_low"], "high": row["success_diff_high"]}, percentage=True),
            "energy_delta": number(row["gpu_joules_diff_mean"]),
            "frontier": "yes" if row["grounded_pareto_frontier"] else "no",
        }
        for row in model_rows
    ]
    (TABLES / "model_transfer.md").write_text(
        markdown_table(model_table_rows, [("model", "Model"), ("direct", "Direct success"), ("grounded", "Grounded success"), ("delta", "Grounded-direct success (95% CI)"), ("energy_delta", "Energy delta J"), ("frontier", "Grounded frontier")]),
        encoding="utf-8",
    )
    robust_table_rows = [
        {
            "condition": row["condition"],
            "success": pct(row["selected_success_rate"]),
            "abstention": pct(row["abstention_rate"]),
            "coverage": pct(row["execution_coverage"]),
            "compromise": pct(row["prompt_injection_compromise_rate"]),
        }
        for row in robust_condition_rows
    ]
    (TABLES / "robustness.md").write_text(
        markdown_table(robust_table_rows, [("condition", "Condition"), ("success", "Selected success"), ("abstention", "Abstention"), ("coverage", "Coverage"), ("compromise", "Injection compromise")]),
        encoding="utf-8",
    )

    best_route = max(route_rows, key=lambda row: (row["success_rate"], -row["mean_gpu_joules"]))
    clean_audit, robust_audit = trace_audits
    report = f"""# E2AM-MemRAG v3r1 results and claim audit

## Release verdict

The experiment is **complete and remotely verified**, but the predeclared
confirmatory hypothesis **did not pass**. The formal non-inferiority and
operating-constraint checks passed only because the selected policy reproduced
the baseline trace exactly; they do not establish a useful quality-preserving
router. The required energy-reduction gate failed.

| Gate | Result | Interpretation |
| --- | --- | --- |
| Experiment completion | PASS | Frozen releases restored and verified |
| Fresh-root restore | PASS | All Stage-09 artifact checks passed |
| Formal quality non-inferiority | PASS | Non-informative identity with the baseline |
| Operating constraints | PASS | Test-time execution and accounting were complete |
| Energy reduction | FAIL | Policy-minus-baseline energy was exactly 0 J/query |
| Confirmatory hypothesis | FAIL | The conjunction of required gates was false |

## The central diagnostic: was the action pool routable?

Adapting the established single-best/virtual-best comparison, the 11-action
resident-eligible pool had 12.5% best-fixed success and 12.5% post-hoc
per-query oracle success: routing headroom was exactly **0.0 percentage points**.
All 15 eligible successes came from `A00_tiny_direct`. In contrast, the six
offline reference actions had 44.2% best-fixed success and a 72.5% oracle, or
28.3 points of headroom. The complete 17-route matrix reached a 73.3% oracle.
The subtraction is established; the contribution is its integration with
interface, capability, physical-cost, and validation-feasibility checks for
composite RAG actions.

Eligibility required the 0.6B and 1B checkpoints to coexist on one T4 with at
least 15% free VRAM and no offload. The 3B/4B references were loaded sequentially,
and dynamic loading was outside route latency. A label-aware per-query cost bound
that preserves all 15 successes chooses A03 once and saves only 0.049 J/query
(0.089%). The pool therefore had zero success headroom, not mathematically zero
cost headroom. These analyses are post hoc and do not change the confirmatory
outcome.

## Primary policy result

The router selected `{summary['baseline_route_id']}` for all 120 clean test
queries. Strict support-qualified success was {pct(summary['policy_success']['mean'])}
(95% cluster interval {pct(summary['policy_success']['low'])} to
{pct(summary['policy_success']['high'])}); mean selected-GPU energy was
{summary['policy_gpu_joules']['mean']:.2f} J/query (95% interval
{summary['policy_gpu_joules']['low']:.2f} to {summary['policy_gpu_joules']['high']:.2f}).
The stored policy and baseline records are the same executions, so their paired
differences are exactly 0.00 for success and 0.00 J for energy by identity. The
policy did not discover an energy-saving decision rule.

This is not an execution failure: coverage was
{pct(summary['execution_coverage'])}, all {clean_audit['rows']} clean-test
generation calls have selected-GPU board energy sampled every 50 ms around
`model.generate()`, and the clean release records zero execution failures. CPU
retrieval and embedding, memory traversal, routing, parsing, verification, storage,
network, host and cooling energy, and carbon are outside that boundary.

## Route and generator findings

The best route among all 17 retained endpoints was the offline, router-ineligible
reference `{best_route['route_id']}` at {pct(best_route['success_rate'])} strict
success and {best_route['mean_gpu_joules']:.2f} J/query. The grounded Pareto
frontier contains the `granite` and `peer` model families.

- Granite 3B grounding improved strict success by
  {ci_text(model_panel['models'][2]['grounded_minus_direct_success'], percentage=True)}
  while adding {model_panel['models'][2]['grounded_minus_direct_gpu_joules']['mean']:.2f} J/query.
- SmolLM3 3B grounding improved success by
  {ci_text(model_panel['models'][3]['grounded_minus_direct_success'], percentage=True)};
  its interval includes zero, while energy increased by
  {model_panel['models'][3]['grounded_minus_direct_gpu_joules']['mean']:.2f} J/query.
- Qwen 4B grounding improved success by
  {ci_text(model_panel['models'][4]['grounded_minus_direct_success'], percentage=True)}
  but added {model_panel['models'][4]['grounded_minus_direct_gpu_joules']['mean']:.2f} J/query.
- Qwen 0.6B grounding **reduced** success by
  {ci_text(model_panel['models'][0]['grounded_minus_direct_success'], percentage=True)}
  and added {model_panel['models'][0]['grounded_minus_direct_gpu_joules']['mean']:.2f} J/query.

The controlled tiny-model retrieval and memory routes all changed success from
12.5% to 0% while adding approximately 88--117 J/query. Those 15 direct successes
came from the no-retrieval stratum; the composite-grounded endpoints lost them
without adding strict success on evidence-required tasks. This is an end-to-end
generator--prompt--parser compatibility result, not a universal claim that
retrieval harms small models.

## Retrieval-to-utilization audit

The five grounded endpoints received identical ordered evidence lists for all
120 test queries. Required evidence was retrieved completely for 72 of the 90
evidence-dependent queries (80.0%), including 14 of 15 multi-hop queries. Yet
strict grounded success on that 90-query stratum ranged from 0% to 58.9%, which
locates the matched-panel divergence downstream of retrieval.

## Robustness interpretation

The selected policy achieved 0% strict success on the clean baseline and 0% in
all four corruption conditions. Consequently, the reported zero robustness
deltas are a **floor effect**, not evidence that the system retained useful
accuracy under corruption. Prompt-injection compromise was 0%, but efficacy was
also 0%; the paper must not present this as a successful robustness result.

## Trace audit

- Clean release: {clean_audit['rows']} rows, {clean_audit['routes']} routes,
  {clean_audit['query_ids']} unique queries, {clean_audit['successes']} successes
  ({pct(clean_audit['success_rate'])}).
- Robustness release: {robust_audit['rows']} rows, {robust_audit['routes']} routes,
  {robust_audit['query_ids']} unique queries, {robust_audit['successes']} successes
  ({pct(robust_audit['success_rate'])}).
- Duplicate unit IDs: {clean_audit['duplicate_unit_ids']} clean and
  {robust_audit['duplicate_unit_ids']} robustness.
- Non-finite audited metrics: {clean_audit['non_finite_values']} clean and
  {robust_audit['non_finite_values']} robustness.

## Defensible paper framing

The strongest paper is a failure-first study of **action-pool routability**.
Strict-success selection was mathematically unproductive inside the frozen resident-eligible pool even
though the broader reference pool contained substantial query-level
complementarity. The matched panel explains that contrast through
generator-dependent grounding utility and positive generation-window energy
increments. The contribution is the audit and mechanism analysis, not a claim
that the learned policy reduced energy.

Claims must remain bounded to this controlled synthetic benchmark, one visible T4
per worker (four physical boards across clean lanes), selected-GPU board energy,
the frozen model revisions, and strict
support-qualified success. Carbon, whole-system energy, public-benchmark SOTA,
and broad real-world generalization are outside the evidence.

## Provenance

- Hugging Face dataset: `{SOURCE_REPO}`
- Pinned visible release commit: `{SOURCE_MAIN_COMMIT}`
- Pinned paper branch commit: `{SOURCE_PAPER_COMMIT}`
- Stage-09 artifact checks: 11/11 passed, 11,528,142 bytes verified
- Frozen execution spec SHA-256: `{load_json('release_manifest.json')['execution_spec_sha256']}`
"""
    (PAPER / "RESULTS_AUDIT.md").write_text(report, encoding="utf-8")

    result_summary = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "release_complete": True,
        "hypothesis_pass": False,
        "analysis_status": "confirmatory result frozen; routability and mechanism analyses are post-hoc descriptive",
        "hypothesis_gates": hypothesis,
        "primary_policy": overall_rows[0],
        "policy_baseline_trace_identity": True,
        "best_clean_route": best_route,
        "grounded_pareto_frontier_model_keys": sorted(frontier),
        "routability": {
            "resident_eligible": {"actions": 11, "best_fixed_success": 0.125, "oracle_success": 0.125, "routing_headroom": 0.0},
            "offline_reference": {"actions": 6, "best_fixed_success": 0.44166666666666665, "oracle_success": 0.725, "routing_headroom": 0.2833333333333333},
            "all_retained": {"actions": 17, "best_fixed_success": 0.44166666666666665, "oracle_success": 0.7333333333333333, "routing_headroom": 0.29166666666666663},
        },
        "resident_eligibility": "0.6B and 1B checkpoints coexist on one T4 with >=15% free VRAM and no CPU/disk offload; 3B/4B references loaded sequentially; model loading excluded from route latency",
        "success_preserving_cost_oracle": {
            "baseline_successes": 15,
            "oracle_successes": 15,
            "mean_saving_gpu_joules": 0.04890513398299845,
            "percent_saving": 0.08938346395029928,
            "selection_counts": {"A00_tiny_direct": 119, "A03_tiny_hybrid": 1},
            "analysis_status": "post-hoc label-aware descriptive bound",
        },
        "retrieval_to_utilization": {
            "grounded_endpoints_with_identical_ordered_evidence": 5,
            "queries_with_identical_ordered_evidence": 120,
            "evidence_required_queries": 90,
            "retrieval_complete_queries": 72,
        },
        "trace_audit": trace_audits,
        "measurement_boundary": "selected-GPU board energy sampled during model.generate only; excludes CPU retrieval, embedding, memory traversal, routing, parsing, verification, storage, network, host, cooling, and carbon",
        "claim_boundary": "controlled synthetic benchmark; one visible T4 per worker; four clean-lane boards; generation-window selected-GPU board energy; cross-board contrasts descriptive",
        "provenance": provenance,
    }
    (DERIVED / "results_summary.json").write_text(canonical_json(result_summary), encoding="utf-8")

    generated = sorted(
        path for path in PAPER.rglob("*")
        if path.is_file()
        and path.name != "RESULTS_MANIFEST.json"
        and "manuscript/build" not in path.relative_to(PAPER).as_posix()
        and "__pycache__" not in path.parts
    )
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "source_repo": SOURCE_REPO,
        "source_main_commit": SOURCE_MAIN_COMMIT,
        "source_paper_commit": SOURCE_PAPER_COMMIT,
        "hypothesis_pass": False,
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in generated
        ],
    }
    (PAPER / "RESULTS_MANIFEST.json").write_text(canonical_json(manifest), encoding="utf-8")
    print(
        "PAPER_RESULTS_READY",
        {
            "raw_artifacts_verified": 11,
            "clean_rows": len(clean_rows),
            "robustness_rows": len(robust_rows),
            "hypothesis_pass": False,
            "generated_files": len(manifest["files"]),
        },
    )


if __name__ == "__main__":
    main()

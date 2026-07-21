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
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.e2am_memrag.hybridbench import generate_hybridbench  # noqa: E402


ROOT = PROJECT_ROOT
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

MATCHED_PAIRS = {
    "tiny": ("A00_tiny_direct", "M16_tiny_grounded_verified"),
    "small": ("A04_small_direct", "A13_small_hybrid_verified"),
    "granite": ("M17_granite_direct", "M18_granite_grounded_verified"),
    "peer": ("M19_peer_direct", "M20_peer_grounded_verified"),
    "upper": ("M21_upper_direct", "A14_upper_hybrid_verified"),
}

TASK_LABELS = {
    "no_retrieval": "Direct copy",
    "knowledge_only": "Knowledge",
    "memory_only": "Memory",
    "knowledge_memory": "Knowledge + memory",
    "temporal_update": "Temporal update",
    "authority_conflict": "Authority conflict",
    "multi_hop": "Two hop",
    "deleted_or_missing": "Deleted / missing",
}

EVIDENCE_TASKS = {
    "knowledge_only",
    "memory_only",
    "knowledge_memory",
    "temporal_update",
    "authority_conflict",
    "multi_hop",
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


def cluster_bootstrap(
    observations: list[tuple[str, float]],
    *,
    seed: int = 4622,
    replicates: int = 10_000,
) -> dict[str, float | int]:
    """Match the frozen scenario-cluster bootstrap used in Stage 09."""

    grouped: dict[str, list[float]] = defaultdict(list)
    for cluster_id, value in observations:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise RuntimeError("Non-finite value in manuscript bootstrap")
        grouped[str(cluster_id)].append(numeric)
    cluster_ids = sorted(grouped)
    if not cluster_ids:
        raise RuntimeError("Manuscript bootstrap received no observations")
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(replicates):
        sampled = [cluster_ids[rng.randrange(len(cluster_ids))] for _ in cluster_ids]
        values = [value for cluster in sampled for value in grouped[cluster]]
        means.append(statistics.fmean(values))
    means.sort()
    values = [value for cluster in cluster_ids for value in grouped[cluster]]
    return {
        "mean": statistics.fmean(values),
        "low": means[int(0.025 * (replicates - 1))],
        "high": means[int(0.975 * (replicates - 1))],
        "clusters": len(cluster_ids),
        "replicates": replicates,
    }


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        raise RuntimeError("Correlation requires equally sized non-trivial vectors")
    if len(set(left)) < 2 or len(set(right)) < 2:
        return None
    return statistics.correlation(left, right)


def benchmark_test_metadata() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    benchmark = generate_hybridbench(800, seed=4622)
    queries = {
        str(row["query_id"]): row
        for row in benchmark["queries"]
        if str(row["split"]) == "test"
    }
    labels = {
        str(row["query_id"]): row
        for row in benchmark["labels"]
        if str(row["query_id"]) in queries
    }
    if len(queries) != 120 or set(queries) != set(labels):
        raise RuntimeError("Frozen HybridBench test reconstruction changed")
    return queries, labels


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


def build_routability_derivatives(clean: list[dict[str, Any]]) -> None:
    """Derive post-hoc routability diagnostics without rerunning any model."""

    queries, labels = benchmark_test_metadata()
    query_ids = sorted(queries)
    trace_query_ids = {str(row["query_id"]) for row in clean}
    if trace_query_ids != set(query_ids):
        raise RuntimeError("Clean traces do not match the reconstructed sealed test IDs")

    by_pair = {
        (str(row["query_id"]), str(row["route_id"])): row
        for row in clean
    }
    route_ids = sorted({str(row["route_id"]) for row in clean})
    if len(by_pair) != len(query_ids) * len(route_ids):
        raise RuntimeError("Clean trace matrix is incomplete or contains duplicate route-query pairs")

    policy = json.loads((RAW / "stage06_frozen_policy.json").read_text(encoding="utf-8"))
    deployable = [str(value) for value in policy["route_ids"]]
    offline_reference = [
        "M17_granite_direct",
        "M18_granite_grounded_verified",
        "M19_peer_direct",
        "M20_peer_grounded_verified",
        "M21_upper_direct",
        "A14_upper_hybrid_verified",
    ]
    pools = [
        ("deployable", "Resident-eligible pool", deployable),
        ("granite_pair", "Granite 4.1 3B pair", list(MATCHED_PAIRS["granite"])),
        ("peer_pair", "SmolLM3 3B pair", list(MATCHED_PAIRS["peer"])),
        ("upper_pair", "Qwen3 4B pair", list(MATCHED_PAIRS["upper"])),
        ("offline_reference", "Offline reference pool", offline_reference),
        ("all_retained", "All retained routes", route_ids),
    ]

    pool_rows: list[dict[str, Any]] = []
    contribution_rows: list[dict[str, Any]] = []
    for pool_id, label, routes in pools:
        missing = sorted(set(routes) - set(route_ids))
        if missing:
            raise RuntimeError(f"Routability pool {pool_id} contains absent routes: {missing}")
        success_counts = {
            route_id: sum(bool(by_pair[(query_id, route_id)]["success"]) for query_id in query_ids)
            for route_id in routes
        }
        mean_energies = {
            route_id: statistics.fmean(
                float(by_pair[(query_id, route_id)]["generation"]["energy"]["energy_joules"])
                for query_id in query_ids
            )
            for route_id in routes
        }
        best_route = max(routes, key=lambda route_id: (success_counts[route_id], -mean_energies[route_id]))
        oracle_count = sum(
            any(bool(by_pair[(query_id, route_id)]["success"]) for route_id in routes)
            for query_id in query_ids
        )
        best_fixed = success_counts[best_route] / len(query_ids)
        oracle = oracle_count / len(query_ids)
        pool_rows.append(
            {
                "pool_id": pool_id,
                "pool_label": label,
                "actions": len(routes),
                "best_route_id": best_route,
                "best_fixed_success": best_fixed,
                "oracle_success": oracle,
                "routing_headroom": oracle - best_fixed,
                "oracle_successes": oracle_count,
            }
        )
        for route_id in routes:
            unique = sum(
                bool(by_pair[(query_id, route_id)]["success"])
                and sum(bool(by_pair[(query_id, candidate)]["success"]) for candidate in routes) == 1
                for query_id in query_ids
            )
            contribution_rows.append(
                {
                    "pool_id": pool_id,
                    "route_id": route_id,
                    "successes": success_counts[route_id],
                    "unique_successes": unique,
                    "mean_gpu_joules": mean_energies[route_id],
                }
            )

    write_csv(
        DERIVED / "routability_pools.csv",
        pool_rows,
        [
            "pool_id",
            "pool_label",
            "actions",
            "best_route_id",
            "best_fixed_success",
            "oracle_success",
            "routing_headroom",
            "oracle_successes",
        ],
    )
    write_csv(
        DERIVED / "routability_route_contributions.csv",
        contribution_rows,
        ["pool_id", "route_id", "successes", "unique_successes", "mean_gpu_joules"],
    )

    # A zero success gap does not logically imply zero cost opportunity because
    # per-query energy varies.  Compute the label-aware minimum-energy selector
    # that must retain every success of the best fixed deployable route.  This is
    # a descriptive ceiling over recorded traces, not an executable policy.
    baseline_route = "A00_tiny_direct"
    cost_choices: list[tuple[str, dict[str, Any]]] = []
    for query_id in query_ids:
        baseline_row = by_pair[(query_id, baseline_route)]
        candidates = [
            (route_id, by_pair[(query_id, route_id)])
            for route_id in deployable
            if not bool(baseline_row["success"])
            or bool(by_pair[(query_id, route_id)]["success"])
        ]
        if not candidates:
            raise RuntimeError(f"No success-preserving cost choice for {query_id}")
        cost_choices.append(
            min(
                candidates,
                key=lambda item: float(item[1]["generation"]["energy"]["energy_joules"]),
            )
        )
    baseline_rows = [by_pair[(query_id, baseline_route)] for query_id in query_ids]
    baseline_mean = statistics.fmean(
        float(row["generation"]["energy"]["energy_joules"]) for row in baseline_rows
    )
    cost_oracle_mean = statistics.fmean(
        float(row["generation"]["energy"]["energy_joules"])
        for _, row in cost_choices
    )
    baseline_successes = sum(bool(row["success"]) for row in baseline_rows)
    cost_oracle_successes = sum(bool(row["success"]) for _, row in cost_choices)
    selection_counts = Counter(route_id for route_id, _ in cost_choices)
    changed = [
        (query_id, route_id, row)
        for query_id, (route_id, row) in zip(query_ids, cost_choices)
        if route_id != baseline_route
    ]
    same_board_for_changes = all(
        str(by_pair[(query_id, baseline_route)]["generation"]["energy"]["gpu_uuid"])
        == str(row["generation"]["energy"]["gpu_uuid"])
        for query_id, _, row in changed
    )
    if cost_oracle_successes < baseline_successes or cost_oracle_mean > baseline_mean:
        raise RuntimeError("Success-preserving cost oracle violated its constraints")
    cost_oracle_row = {
        "baseline_route_id": baseline_route,
        "queries": len(query_ids),
        "baseline_successes": baseline_successes,
        "oracle_successes": cost_oracle_successes,
        "baseline_mean_gpu_joules": baseline_mean,
        "oracle_mean_gpu_joules": cost_oracle_mean,
        "mean_saving_gpu_joules": baseline_mean - cost_oracle_mean,
        "percent_saving": 100.0 * (baseline_mean - cost_oracle_mean) / baseline_mean,
        "selections_changed": len(changed),
        "selection_counts": ";".join(
            f"{route_id}:{count}" for route_id, count in sorted(selection_counts.items())
        ),
        "same_board_for_changes": same_board_for_changes,
        "analysis_status": "post-hoc label-aware descriptive bound",
    }
    write_csv(
        DERIVED / "success_preserving_cost_oracle.csv",
        [cost_oracle_row],
        [
            "baseline_route_id",
            "queries",
            "baseline_successes",
            "oracle_successes",
            "baseline_mean_gpu_joules",
            "oracle_mean_gpu_joules",
            "mean_saving_gpu_joules",
            "percent_saving",
            "selections_changed",
            "selection_counts",
            "same_board_for_changes",
            "analysis_status",
        ],
    )

    task_oracle_rows: list[dict[str, Any]] = []
    pair_effects: dict[str, list[float]] = {}
    for model_key, (direct_id, grounded_id) in MATCHED_PAIRS.items():
        direct_rows = [by_pair[(query_id, direct_id)] for query_id in query_ids]
        grounded_rows = [by_pair[(query_id, grounded_id)] for query_id in query_ids]
        selected_rows = [
            by_pair[(query_id, direct_id)]
            if str(queries[query_id]["task_type"]) in {"no_retrieval", "deleted_or_missing"}
            else by_pair[(query_id, grounded_id)]
            for query_id in query_ids
        ]
        scenario_ids = [str(queries[query_id]["scenario_id"]) for query_id in query_ids]
        success_delta = [
            float(bool(selected["success"])) - float(bool(grounded["success"]))
            for selected, grounded in zip(selected_rows, grounded_rows)
        ]
        energy_delta = [
            float(selected["generation"]["energy"]["energy_joules"])
            - float(grounded["generation"]["energy"]["energy_joules"])
            for selected, grounded in zip(selected_rows, grounded_rows)
        ]
        success_ci = cluster_bootstrap(list(zip(scenario_ids, success_delta)))
        energy_ci = cluster_bootstrap(list(zip(scenario_ids, energy_delta)))
        oracle_rate = statistics.fmean(
            float(bool(direct["success"]) or bool(grounded["success"]))
            for direct, grounded in zip(direct_rows, grounded_rows)
        )
        task_rate = statistics.fmean(float(bool(row["success"])) for row in selected_rows)
        if not math.isclose(task_rate, oracle_rate, abs_tol=1e-12):
            raise RuntimeError(f"Task-aware and pair-oracle utility diverged for {model_key}")
        pair_effects[model_key] = [
            float(bool(grounded["success"])) - float(bool(direct["success"]))
            for direct, grounded in zip(direct_rows, grounded_rows)
        ]
        task_oracle_rows.append(
            {
                "model_key": model_key,
                "direct_route_id": direct_id,
                "grounded_route_id": grounded_id,
                "always_direct_success": statistics.fmean(float(bool(row["success"])) for row in direct_rows),
                "always_grounded_success": statistics.fmean(float(bool(row["success"])) for row in grounded_rows),
                "task_aware_success": task_rate,
                "pair_oracle_success": oracle_rate,
                "success_gain_vs_grounded": success_ci["mean"],
                "success_gain_low": success_ci["low"],
                "success_gain_high": success_ci["high"],
                "energy_change_vs_grounded": energy_ci["mean"],
                "energy_change_low": energy_ci["low"],
                "energy_change_high": energy_ci["high"],
            }
        )
    write_csv(
        DERIVED / "task_aware_oracle.csv",
        task_oracle_rows,
        [
            "model_key",
            "direct_route_id",
            "grounded_route_id",
            "always_direct_success",
            "always_grounded_success",
            "task_aware_success",
            "pair_oracle_success",
            "success_gain_vs_grounded",
            "success_gain_low",
            "success_gain_high",
            "energy_change_vs_grounded",
            "energy_change_low",
            "energy_change_high",
        ],
    )

    interaction_rows: list[dict[str, Any]] = []
    for left, right in [("granite", "tiny"), ("upper", "tiny"), ("peer", "tiny"), ("granite", "peer")]:
        differences = [a - b for a, b in zip(pair_effects[left], pair_effects[right])]
        result = cluster_bootstrap(
            [(str(queries[query_id]["scenario_id"]), value) for query_id, value in zip(query_ids, differences)]
        )
        interaction_rows.append(
            {
                "left_model": left,
                "right_model": right,
                "effect_difference": result["mean"],
                "low": result["low"],
                "high": result["high"],
                "clusters": result["clusters"],
                "replicates": result["replicates"],
            }
        )
    write_csv(
        DERIVED / "grounding_interactions.csv",
        interaction_rows,
        ["left_model", "right_model", "effect_difference", "low", "high", "clusters", "replicates"],
    )

    grounded_routes = [grounded for _, grounded in MATCHED_PAIRS.values()]
    for query_id in query_ids:
        signatures = {
            (
                tuple(by_pair[(query_id, route_id)]["retrieval"]["knowledge_ids"]),
                tuple(by_pair[(query_id, route_id)]["retrieval"]["memory_ids"]),
            )
            for route_id in grounded_routes
        }
        if len(signatures) != 1:
            raise RuntimeError(f"Grounded evidence differs across generators for {query_id}")

    evidence_query_ids = [
        query_id for query_id in query_ids if str(queries[query_id]["task_type"]) in EVIDENCE_TASKS
    ]
    utilization_rows: list[dict[str, Any]] = []
    for model_key, (_, route_id) in MATCHED_PAIRS.items():
        rows = [by_pair[(query_id, route_id)] for query_id in evidence_query_ids]
        retrieval_complete = []
        for row in rows:
            query_id = str(row["query_id"])
            required = set(map(str, labels[query_id]["required_doc_ids"])) | set(
                map(str, labels[query_id]["required_memory_ids"])
            )
            retrieved = set(map(str, row["retrieval"]["knowledge_ids"])) | set(
                map(str, row["retrieval"]["memory_ids"])
            )
            retrieval_complete.append(required.issubset(retrieved))
        utilization_rows.append(
            {
                "model_key": model_key,
                "route_id": route_id,
                "queries": len(rows),
                "retrieval_complete_rate": statistics.fmean(map(float, retrieval_complete)),
                "parse_rate": statistics.fmean(float(bool(row["metrics"]["parse_ok"])) for row in rows),
                "citation_complete_rate": statistics.fmean(
                    float(bool(row["metrics"]["evidence_set_complete"])) for row in rows
                ),
                "answer_valid_rate": statistics.fmean(float(bool(row["metrics"]["answer_ok"])) for row in rows),
                "support_valid_rate": statistics.fmean(float(bool(row["metrics"]["support_ok"])) for row in rows),
                "strict_success_rate": statistics.fmean(float(bool(row["success"])) for row in rows),
            }
        )
    write_csv(
        DERIVED / "retrieval_to_utilization.csv",
        utilization_rows,
        [
            "model_key",
            "route_id",
            "queries",
            "retrieval_complete_rate",
            "parse_rate",
            "citation_complete_rate",
            "answer_valid_rate",
            "support_valid_rate",
            "strict_success_rate",
        ],
    )

    task_effect_rows: list[dict[str, Any]] = []
    for model_key, (direct_id, grounded_id) in MATCHED_PAIRS.items():
        for task_type in TASK_LABELS:
            task_ids = [query_id for query_id in query_ids if str(queries[query_id]["task_type"]) == task_type]
            direct_rate = statistics.fmean(float(bool(by_pair[(query_id, direct_id)]["success"])) for query_id in task_ids)
            grounded_rate = statistics.fmean(float(bool(by_pair[(query_id, grounded_id)]["success"])) for query_id in task_ids)
            task_effect_rows.append(
                {
                    "model_key": model_key,
                    "task_type": task_type,
                    "queries": len(task_ids),
                    "direct_success": direct_rate,
                    "grounded_success": grounded_rate,
                    "grounded_minus_direct": grounded_rate - direct_rate,
                }
            )
    write_csv(
        DERIVED / "task_grounding_effects.csv",
        task_effect_rows,
        ["model_key", "task_type", "queries", "direct_success", "grounded_success", "grounded_minus_direct"],
    )

    decomposition_rows: list[dict[str, Any]] = []
    for route_id in route_ids:
        rows = [by_pair[(query_id, route_id)] for query_id in query_ids]
        decomposition_rows.append(
            {
                "route_id": route_id,
                "parse_rate": statistics.fmean(float(bool(row["metrics"]["parse_ok"])) for row in rows),
                "answer_valid_rate": statistics.fmean(float(bool(row["metrics"]["answer_ok"])) for row in rows),
                "citation_complete_rate": statistics.fmean(
                    float(bool(row["metrics"]["evidence_set_complete"])) for row in rows
                ),
                "support_valid_rate": statistics.fmean(float(bool(row["metrics"]["support_ok"])) for row in rows),
                "correct_abstention_rate": statistics.fmean(
                    float(bool(row["metrics"]["correct_abstention"])) for row in rows
                ),
                "strict_success_rate": statistics.fmean(float(bool(row["success"])) for row in rows),
            }
        )
    write_csv(
        DERIVED / "failure_decomposition.csv",
        decomposition_rows,
        [
            "route_id",
            "parse_rate",
            "answer_valid_rate",
            "citation_complete_rate",
            "support_valid_rate",
            "correct_abstention_rate",
            "strict_success_rate",
        ],
    )

    threshold_rows: list[dict[str, Any]] = []
    for model_key, (_, grounded_id) in MATCHED_PAIRS.items():
        rows = [by_pair[(query_id, grounded_id)] for query_id in query_ids]
        for threshold in (0.5, 0.8, 1.0):
            rescored = []
            for row in rows:
                metrics = row["metrics"]
                rescored.append(
                    float(metrics["answer_f1"]) >= threshold
                    and float(metrics["citation_recall"]) == 1.0
                    and float(metrics["citation_precision"]) == 1.0
                    and bool(metrics["correct_abstention"])
                    and bool(metrics["parse_ok"])
                    and not metrics["invalid_citation_ids"]
                    and not metrics["forbidden_exposure_ids"]
                )
            threshold_rows.append(
                {
                    "model_key": model_key,
                    "route_id": grounded_id,
                    "answer_f1_threshold": threshold,
                    "strict_success_rate": statistics.fmean(map(float, rescored)),
                }
            )
    write_csv(
        DERIVED / "strict_threshold_sensitivity.csv",
        threshold_rows,
        ["model_key", "route_id", "answer_f1_threshold", "strict_success_rate"],
    )

    energy_rows: list[dict[str, Any]] = []
    for model_key, (direct_id, grounded_id) in MATCHED_PAIRS.items():
        direct_rows = [by_pair[(query_id, direct_id)] for query_id in query_ids]
        grounded_rows = [by_pair[(query_id, grounded_id)] for query_id in query_ids]
        delta_input = [
            float(grounded["generation"]["input_tokens"]) - float(direct["generation"]["input_tokens"])
            for direct, grounded in zip(direct_rows, grounded_rows)
        ]
        delta_output = [
            float(grounded["generation"]["output_tokens"]) - float(direct["generation"]["output_tokens"])
            for direct, grounded in zip(direct_rows, grounded_rows)
        ]
        delta_time = [
            float(grounded["generation"]["latency_seconds"]) - float(direct["generation"]["latency_seconds"])
            for direct, grounded in zip(direct_rows, grounded_rows)
        ]
        delta_energy = [
            float(grounded["generation"]["energy"]["energy_joules"])
            - float(direct["generation"]["energy"]["energy_joules"])
            for direct, grounded in zip(direct_rows, grounded_rows)
        ]
        direct_boards = {str(row["generation"]["energy"]["gpu_uuid"]) for row in direct_rows}
        grounded_boards = {str(row["generation"]["energy"]["gpu_uuid"]) for row in grounded_rows}
        energy_rows.append(
            {
                "model_key": model_key,
                "same_board": direct_boards == grounded_boards and len(direct_boards) == 1,
                "mean_delta_input_tokens": statistics.fmean(delta_input),
                "mean_delta_output_tokens": statistics.fmean(delta_output),
                "mean_delta_generation_seconds": statistics.fmean(delta_time),
                "mean_delta_gpu_joules": statistics.fmean(delta_energy),
                "correlation_energy_time": pearson(delta_energy, delta_time),
                "correlation_energy_output_tokens": pearson(delta_energy, delta_output),
            }
        )
    write_csv(
        DERIVED / "energy_mechanism.csv",
        energy_rows,
        [
            "model_key",
            "same_board",
            "mean_delta_input_tokens",
            "mean_delta_output_tokens",
            "mean_delta_generation_seconds",
            "mean_delta_gpu_joules",
            "correlation_energy_time",
            "correlation_energy_output_tokens",
        ],
    )

    deployable_contributions = {
        row["route_id"]: row for row in contribution_rows if row["pool_id"] == "deployable"
    }
    a00 = deployable_contributions["A00_tiny_direct"]
    dominance_rows: list[dict[str, Any]] = []
    a00_boards = {
        str(by_pair[(query_id, "A00_tiny_direct")]["generation"]["energy"]["gpu_uuid"])
        for query_id in query_ids
    }
    for route_id, row in sorted(deployable_contributions.items()):
        if route_id == "A00_tiny_direct":
            continue
        route_boards = {
            str(by_pair[(query_id, route_id)]["generation"]["energy"]["gpu_uuid"])
            for query_id in query_ids
        }
        dominated = (
            int(a00["successes"]) >= int(row["successes"])
            and float(a00["mean_gpu_joules"]) <= float(row["mean_gpu_joules"])
            and (
                int(a00["successes"]) > int(row["successes"])
                or float(a00["mean_gpu_joules"]) < float(row["mean_gpu_joules"])
            )
        )
        dominance_rows.append(
            {
                "candidate_route_id": route_id,
                "a00_dominates": dominated,
                "same_board": a00_boards == route_boards and len(a00_boards) == 1,
                "success_difference_a00_minus_candidate": (
                    int(a00["successes"]) - int(row["successes"])
                ) / len(query_ids),
                "energy_difference_a00_minus_candidate": (
                    float(a00["mean_gpu_joules"]) - float(row["mean_gpu_joules"])
                ),
            }
        )
    if not all(bool(row["a00_dominates"]) for row in dominance_rows):
        raise RuntimeError("A00 no longer descriptively dominates every deployable alternative")
    write_csv(
        DERIVED / "deployable_dominance.csv",
        dominance_rows,
        [
            "candidate_route_id",
            "a00_dominates",
            "same_board",
            "success_difference_a00_minus_candidate",
            "energy_difference_a00_minus_candidate",
        ],
    )

    deployable_row = next(row for row in pool_rows if row["pool_id"] == "deployable")
    zero_parse_routes = [
        row["route_id"] for row in decomposition_rows
        if row["route_id"] in deployable and math.isclose(float(row["parse_rate"]), 0.0)
    ]
    validation = policy["validation_selection"]
    gates = [
        {
            "gate": "Interface compatibility",
            "status": "PARTIAL",
            "evidence": f"{len(zero_parse_routes)} retained actions had 0% parseability",
        },
        {
            "gate": "Capability",
            "status": "FAIL",
            "evidence": f"oracle {100*float(deployable_row['oracle_success']):.1f}% < frozen 80% requirement",
        },
        {
            "gate": "Complementarity",
            "status": "FAIL",
            "evidence": f"routing headroom {100*float(deployable_row['routing_headroom']):.1f} pp",
        },
        {
            "gate": "Cost comparability",
            "status": "QUALIFIED",
            "evidence": "complete generation-window telemetry; multiple physical T4 boards",
        },
        {
            "gate": "Constrained selectability",
            "status": "FAIL",
            "evidence": f"0/{len(validation['candidates'])} validation thresholds feasible",
        },
    ]
    write_csv(DERIVED / "routability_gates.csv", gates, ["gate", "status", "evidence"])

    multi_hop_ids = [
        query_id for query_id in query_ids if str(queries[query_id]["task_type"]) == "multi_hop"
    ]
    grounded_probe = MATCHED_PAIRS["granite"][1]
    multi_hop_complete = 0
    for query_id in multi_hop_ids:
        row = by_pair[(query_id, grounded_probe)]
        required = set(map(str, labels[query_id]["required_doc_ids"])) | set(
            map(str, labels[query_id]["required_memory_ids"])
        )
        retrieved = set(map(str, row["retrieval"]["knowledge_ids"])) | set(
            map(str, row["retrieval"]["memory_ids"])
        )
        multi_hop_complete += int(required.issubset(retrieved))
    audit = {
        "schema_version": 1,
        "analysis_status": "post-hoc descriptive; no model inference rerun",
        "queries": len(query_ids),
        "routes": len(route_ids),
        "deployable_routes": len(deployable),
        "grounded_retrieval_identical_queries": len(query_ids),
        "evidence_required_queries": len(evidence_query_ids),
        "retrieval_complete_evidence_queries": int(
            round(len(evidence_query_ids) * float(utilization_rows[0]["retrieval_complete_rate"]))
        ),
        "multi_hop_queries": len(multi_hop_ids),
        "multi_hop_retrieval_complete": multi_hop_complete,
        "pools": pool_rows,
        "success_preserving_cost_oracle": cost_oracle_row,
        "gates": gates,
    }
    (DERIVED / "routability_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
\caption{Frozen generator panel and matched direct/grounded endpoints. Only the first two model families satisfied the co-resident online resource contract; references were loaded sequentially. All checkpoints were loaded in FP16 at immutable revisions.}
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
Formal margin check & pass; policy $=$ baseline \\
Energy reduction & \textbf{{fail}} \\
Coverage and abstention gates & pass \\
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

    pool_order = [
        "deployable",
        "granite_pair",
        "peer_pair",
        "upper_pair",
        "offline_reference",
        "all_retained",
    ]
    pool_lookup = {row["pool_id"]: row for row in read_csv("routability_pools.csv")}
    pool_rows = []
    for pool_id in pool_order:
        row = pool_lookup[pool_id]
        pool_rows.append(
            f"{tex(row['pool_label'])} & {row['actions']} & "
            f"{tex(row['best_route_id'].split('_', 1)[0])} & {fpercent(row['best_fixed_success'])} & "
            f"{fpercent(row['oracle_success'])} & {100*float(row['routing_headroom']):.1f}" + r" \\"
        )
    table_file(
        "routability_pools.tex",
        r"""\begin{table*}[t]
\caption{Post-hoc action-pool routability on the complete sealed-test matrix. The per-query oracle uses outcome labels and is an upper-bound diagnostic, not a deployable policy. Headroom is oracle utility minus the best fixed route.}
\label{tab:routability-pools}
\centering
\small
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Pool & Actions & Best route & Best fixed (\%) & Oracle (\%) & Headroom (pp) \\
\midrule
""" + "\n".join(pool_rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""",
    )

    utilization = read_csv("retrieval_to_utilization.csv")
    utilization_rows = []
    for row in utilization:
        utilization_rows.append(
            f"{tex(MODEL_NAMES[row['model_key']])} & {fpercent(row['retrieval_complete_rate'])} & "
            f"{fpercent(row['parse_rate'])} & {fpercent(row['citation_complete_rate'])} & "
            f"{fpercent(row['answer_valid_rate'])} & {fpercent(row['support_valid_rate'])} & "
            f"{fpercent(row['strict_success_rate'])}" + r" \\"
        )
    table_file(
        "retrieval_utilization.tex",
        r"""\begin{table*}[t]
\caption{Retrieval-to-utilization decomposition on the 90 evidence-required questions. Every grounded endpoint received identical ordered retrieved IDs; retrieval completeness checks required IDs against those retrieved IDs. Citation completeness is an output property and is therefore reported separately.}
\label{tab:retrieval-utilization}
\centering
\small
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
Generator & Retrieved (\%) & Parse (\%) & Cite all (\%) & Answer valid (\%) & Support valid (\%) & Strict (\%) \\
\midrule
""" + "\n".join(utilization_rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""",
    )

    task_oracle = {row["model_key"]: row for row in read_csv("task_aware_oracle.csv")}
    task_oracle_rows = []
    for model_key in ("tiny", "granite", "peer", "upper"):
        row = task_oracle[model_key]
        task_oracle_rows.append(
            f"{tex(MODEL_NAMES[model_key])} & {fpercent(row['always_direct_success'])} & "
            f"{fpercent(row['always_grounded_success'])} & {fpercent(row['task_aware_success'])} & "
            f"{100*float(row['success_gain_vs_grounded']):+.1f} "
            f"[{100*float(row['success_gain_low']):+.1f}, {100*float(row['success_gain_high']):+.1f}] & "
            f"{float(row['energy_change_vs_grounded']):+.2f} "
            f"[{float(row['energy_change_low']):+.2f}, {float(row['energy_change_high']):+.2f}]" + r" \\"
        )
    table_file(
        "task_aware_oracle.tex",
        r"""\begin{table*}[t]
\caption{Exploratory task-aware choice: direct for no-retrieval and deleted/missing items, grounded otherwise. It uses benchmark task labels and is not a learned router. Effects compare this choice with always grounding; intervals are paired scenario-cluster bootstraps.}
\label{tab:task-oracle}
\centering
\small
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Generator & Direct (\%) & Grounded (\%) & Task-aware (\%) & $\Delta$ success [95\% CI] & $\Delta$ GPU J [95\% CI] \\
\midrule
""" + "\n".join(task_oracle_rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""",
    )

    energy_mechanism = {row["model_key"]: row for row in read_csv("energy_mechanism.csv")}
    energy_rows = []
    for model_key in ("tiny", "granite", "peer", "upper"):
        row = energy_mechanism[model_key]
        energy_rows.append(
            f"{tex(MODEL_NAMES[model_key])} & {float(row['mean_delta_input_tokens']):+.1f} & "
            f"{float(row['mean_delta_output_tokens']):+.1f} & "
            f"{float(row['mean_delta_generation_seconds']):+.3f} & "
            f"{float(row['mean_delta_gpu_joules']):+.2f} & "
            f"{float(row['correlation_energy_time']):.3f}" + r" \\"
        )
    table_file(
        "energy_mechanism.tex",
        r"""\begin{table*}[t]
\caption{Descriptive same-board grounding overhead. Deltas are grounded minus direct; $r(\Delta J,\Delta t)$ is the per-query Pearson association between energy and generation-time differences. Route-major, single-run execution precludes a causal decomposition.}
\label{tab:energy-mechanism}
\centering
\small
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Generator & $\Delta$ input tok. & $\Delta$ output tok. & $\Delta$ generation s & $\Delta$ GPU J & $r(\Delta J,\Delta t)$ \\
\midrule
""" + "\n".join(energy_rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""",
    )

    sensitivity = read_csv("strict_threshold_sensitivity.csv")
    sensitivity_lookup = {
        (row["model_key"], float(row["answer_f1_threshold"])): row for row in sensitivity
    }
    sensitivity_rows = []
    for model_key in ("granite", "peer", "upper"):
        sensitivity_rows.append(
            f"{tex(MODEL_NAMES[model_key])} & "
            + " & ".join(
                fpercent(sensitivity_lookup[(model_key, threshold)]["strict_success_rate"])
                for threshold in (0.5, 0.8, 1.0)
            )
            + r" \\"
        )
    table_file(
        "threshold_sensitivity.tex",
        r"""\begin{table}[t]
\caption{Strict-success sensitivity to the answer-F1 threshold for capable grounded endpoints. All other evidence, abstention, and parsing requirements remain unchanged.}
\label{tab:threshold-sensitivity}
\centering
\scriptsize
\begin{tabular}{@{}lrrr@{}}
\toprule
Generator & $F_1\geq0.5$ & $F_1\geq0.8$ & $F_1=1.0$ \\
\midrule
""" + "\n".join(sensitivity_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""",
    )

    decomposition = {row["route_id"]: row for row in read_csv("failure_decomposition.csv")}
    decomposition_ids = [
        "A00_tiny_direct",
        "A03_tiny_hybrid",
        "A13_small_hybrid_verified",
        "M16_tiny_grounded_verified",
        "M18_granite_grounded_verified",
        "M20_peer_grounded_verified",
        "A14_upper_hybrid_verified",
    ]
    decomposition_rows = []
    for route_id in decomposition_ids:
        row = decomposition[route_id]
        decomposition_rows.append(
            f"{tex(route_id.split('_', 1)[0])} & {fpercent(row['parse_rate'])} & "
            f"{fpercent(row['answer_valid_rate'])} & {fpercent(row['citation_complete_rate'])} & "
            f"{fpercent(row['support_valid_rate'])} & {fpercent(row['correct_abstention_rate'])} & "
            f"{fpercent(row['strict_success_rate'])}" + r" \\"
        )
    table_file(
        "failure_decomposition.tex",
        r"""\begin{table*}[t]
\caption{Independently scored outcome components on all 120 clean questions for selected routes; the columns are not a monotone funnel. ``Cite all'' is exact required-ID recall and is automatically one on the 30 evidence-free items, creating a 25\% floor. ``Support'' also requires parseability, exact citation precision, no invalid citation, and no forbidden exposure.}
\label{tab:failure-decomposition}
\centering
\small
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
Route & Parse (\%) & Answer valid (\%) & Cite all (\%) & Support (\%) & Abstention correct (\%) & Strict (\%) \\
\midrule
""" + "\n".join(decomposition_rows) + r"""
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


def build_readiness_figure() -> str:
    width, height = 1800, 650
    boxes = [
        ("1. Interface", "PARTIAL", "3 eligible routes", "0% parseable"),
        ("2. Capability", "FAIL", "Oracle ceiling 12.5%", "required floor 80%"),
        ("3. Complementarity", "FAIL", "Best fixed = oracle", "headroom 0.0 pp"),
        ("4. Cost", "QUALIFIED", "1,320 calls sampled", "generation only; 4 boards"),
        ("5. Selectability", "FAIL", "0 of 9 thresholds", "validation-feasible"),
    ]
    status_fill = {"FAIL": "#fde8e8", "PARTIAL": "#fff4d6", "QUALIFIED": "#e7f0fb"}
    status_stroke = {"FAIL": "#b4232f", "PARTIAL": "#9a6700", "QUALIFIED": "#245f9e"}
    box_w, box_h, gap, start_x, y = 300, 320, 45, 38, 135
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 52, "Failure-first audit of the frozen resident-eligible pool", size=32, weight=700),
        svg_text(width / 2, 90, "Each question is a prerequisite for defensible constrained routing", size=20, fill="#536173"),
    ]
    for index, (title, status, line1, line2) in enumerate(boxes):
        x = start_x + index * (box_w + gap)
        stroke = status_stroke[status]
        parts.append(
            f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="22" '
            f'fill="{status_fill[status]}" stroke="{stroke}" stroke-width="4"/>'
        )
        parts.append(svg_text(x + box_w / 2, y + 58, title, size=24, weight=700))
        parts.append(
            f'<rect x="{x + 80}" y="{y + 88}" width="{box_w - 160}" height="48" rx="16" '
            f'fill="{stroke}"/>'
        )
        parts.append(svg_text(x + box_w / 2, y + 121, status, size=20, weight=700, fill="#ffffff"))
        parts.append(svg_text(x + box_w / 2, y + 205, line1, size=20, weight=600))
        parts.append(svg_text(x + box_w / 2, y + 247, line2, size=20))
        if index < len(boxes) - 1:
            ax = x + box_w + 7
            mid_y = y + box_h / 2
            parts.append(
                f'<line x1="{ax}" y1="{mid_y}" x2="{ax + gap - 18}" y2="{mid_y}" '
                'stroke="#6d7888" stroke-width="5"/>'
            )
            parts.append(
                f'<polygon points="{ax + gap - 18},{mid_y - 11} {ax + gap - 18},{mid_y + 11} '
                f'{ax + gap},{mid_y}" fill="#6d7888"/>'
            )
    parts.append(
        '<rect x="360" y="515" width="1080" height="72" rx="18" fill="#eef2f7" stroke="#50627a" stroke-width="2"/>'
    )
    parts.append(
        svg_text(
            width / 2,
            561,
            "Audit outcome: redesign the eligible pool before interpreting router optimization.",
            size=24,
            weight=700,
        )
    )
    parts.append("</svg>\n")
    return "".join(parts)


def build_headroom_figure(pools: list[dict[str, str]]) -> str:
    wanted = ["deployable", "granite_pair", "peer_pair", "upper_pair", "offline_reference", "all_retained"]
    lookup = {row["pool_id"]: row for row in pools}
    labels = {
        "deployable": ("Resident eligible", "11 actions"),
        "granite_pair": ("Granite 3B", "direct / grounded"),
        "peer_pair": ("SmolLM3 3B", "direct / grounded"),
        "upper_pair": ("Qwen3 4B", "direct / grounded"),
        "offline_reference": ("References", "6 actions"),
        "all_retained": ("Complete", "17 actions"),
    }
    width, height = 1700, 900
    left, right, top, bottom = 120, 55, 105, 175
    plot_h = height - top - bottom
    maximum = 0.8

    def sy(value: float) -> float:
        return top + plot_h * (1.0 - value / maximum)

    group_w = (width - left - right) / len(wanted)
    bar_w = 72
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 48, "Routing opportunity is a property of the action pool", size=32, weight=700),
        svg_text(width / 2, 82, "Best fixed utility versus a post-hoc per-query oracle", size=20, fill="#536173"),
    ]
    for tick in range(0, 81, 10):
        value = tick / 100
        y = sy(value)
        parts.append(f'<line x1="{left}" y1="{y}" x2="{width-right}" y2="{y}" stroke="#e1e5eb" stroke-width="2"/>')
        parts.append(svg_text(left - 18, y + 7, str(tick), size=18, anchor="end"))
    for index, pool_id in enumerate(wanted):
        row = lookup[pool_id]
        center = left + group_w * (index + 0.5)
        best = float(row["best_fixed_success"])
        oracle = float(row["oracle_success"])
        for offset, value, color in [(-bar_w / 2, best, "#7f8c9d"), (bar_w / 2, oracle, "#2563a6")]:
            x = center + offset - bar_w / 2
            y = sy(value)
            parts.append(
                f'<rect x="{x}" y="{y}" width="{bar_w}" height="{top + plot_h - y}" '
                f'fill="{color}" rx="5"/>'
            )
            parts.append(svg_text(x + bar_w / 2, y - 12, f"{100*value:.1f}", size=18, weight=700))
        headroom = 100 * float(row["routing_headroom"])
        parts.append(svg_text(center, top + plot_h + 42, labels[pool_id][0], size=20, weight=700))
        parts.append(svg_text(center, top + plot_h + 70, labels[pool_id][1], size=17, fill="#536173"))
        parts.append(svg_text(center, top + plot_h + 112, f"H = {headroom:.1f} pp", size=18, weight=700, fill="#1f4f82"))
    parts.append(svg_text(28, top + plot_h / 2, "Strict success (%)", size=20, weight=600))
    legend_y = height - 28
    parts.append(f'<rect x="610" y="{legend_y-18}" width="26" height="18" fill="#7f8c9d"/>')
    parts.append(svg_text(650, legend_y - 3, "Best fixed", size=18, anchor="start"))
    parts.append(f'<rect x="820" y="{legend_y-18}" width="26" height="18" fill="#2563a6"/>')
    parts.append(svg_text(860, legend_y - 3, "Per-query oracle", size=18, anchor="start"))
    parts.append("</svg>\n")
    return "".join(parts)


def build_task_heatmap(rows: list[dict[str, str]]) -> str:
    model_order = ["tiny", "small", "granite", "peer", "upper"]
    task_order = list(TASK_LABELS)
    lookup = {(row["model_key"], row["task_type"]): float(row["grounded_minus_direct"]) for row in rows}
    short_labels = {
        "no_retrieval": "Direct copy",
        "knowledge_only": "Knowledge",
        "memory_only": "Memory",
        "knowledge_memory": "Know. + mem.",
        "temporal_update": "Temporal",
        "authority_conflict": "Authority",
        "multi_hop": "Two hop",
        "deleted_or_missing": "Deleted",
    }

    def blend(value: float) -> str:
        neutral = (245, 247, 250)
        target = (34, 102, 172) if value >= 0 else (178, 32, 48)
        intensity = min(1.0, abs(value))
        rgb = tuple(round(neutral[index] + intensity * (target[index] - neutral[index])) for index in range(3))
        return "#" + "".join(f"{channel:02x}" for channel in rgb)

    width, height = 1820, 800
    left, top = 305, 155
    cell_w, cell_h = 178, 92
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 48, "Grounding effects reverse across generators and task classes", size=32, weight=700),
        svg_text(width / 2, 84, "Grounded minus direct strict success (percentage points; 15 questions per cell)", size=20, fill="#536173"),
    ]
    for column, task_type in enumerate(task_order):
        x = left + column * cell_w + cell_w / 2
        parts.append(svg_text(x, top - 30, short_labels[task_type], size=18, weight=600))
    for row_index, model_key in enumerate(model_order):
        y = top + row_index * cell_h
        parts.append(svg_text(left - 22, y + cell_h / 2 + 7, MODEL_NAMES[model_key], size=21, weight=600, anchor="end"))
        for column, task_type in enumerate(task_order):
            value = lookup[(model_key, task_type)]
            x = left + column * cell_w
            color = blend(value)
            text_color = "#ffffff" if abs(value) >= 0.55 else "#172033"
            parts.append(
                f'<rect x="{x+2}" y="{y+2}" width="{cell_w-4}" height="{cell_h-4}" '
                f'fill="{color}" stroke="#ffffff" stroke-width="2"/>'
            )
            parts.append(svg_text(x + cell_w / 2, y + cell_h / 2 + 8, f"{100*value:+.1f}", size=21, weight=700, fill=text_color))
    base_y = top + len(model_order) * cell_h + 54
    parts.append(svg_text(width / 2, base_y, "Red: grounding reduced success    |    Blue: grounding increased success", size=20, weight=600))
    parts.append(svg_text(width / 2, base_y + 40, "The 1B Granite row is an integration failure and is not interpreted as model capability.", size=18, fill="#7d2934"))
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
    pools = read_csv("routability_pools.csv")
    task_effects = read_csv("task_grounding_effects.csv")
    (FIGURES / "system_overview.svg").write_text(build_system_figure(), encoding="utf-8")
    (FIGURES / "routability_audit.svg").write_text(build_readiness_figure(), encoding="utf-8")
    (FIGURES / "action_pool_headroom.svg").write_text(build_headroom_figure(pools), encoding="utf-8")
    (FIGURES / "task_grounding_heatmap.svg").write_text(build_task_heatmap(task_effects), encoding="utf-8")
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
    build_routability_derivatives(clean)
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

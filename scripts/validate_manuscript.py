"""Fail-closed validation for the E2AM-MemRAG manuscript."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
MANUSCRIPT = PAPER / "manuscript"
DERIVED = PAPER / "data" / "derived"
RAW = PAPER / "data" / "raw"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tex_files() -> list[Path]:
    return sorted(MANUSCRIPT.rglob("*.tex"))


def all_tex() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in tex_files())


def bibliography_keys() -> list[str]:
    text = (MANUSCRIPT / "references.bib").read_text(encoding="utf-8")
    return re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", text)


def citation_keys(text: str | None = None) -> set[str]:
    text = text if text is not None else all_tex()
    keys: set[str] = set()
    for group in re.findall(r"\\cite(?:\[[^\]]*\]){0,2}\{([^}]+)\}", text):
        keys.update(key.strip() for key in group.split(",") if key.strip())
    return keys


def label_audit(text: str | None = None) -> dict[str, Any]:
    text = text if text is not None else all_tex()
    labels = re.findall(r"\\label\{([^}]+)\}", text)
    references = re.findall(r"\\(?:ref|eqref|pageref)\{([^}]+)\}", text)
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    unresolved = sorted(set(references) - set(labels))
    return {"labels": len(labels), "references": len(references), "duplicates": duplicates, "unresolved": unresolved}


def input_audit() -> list[str]:
    missing: list[str] = []
    text = (MANUSCRIPT / "main.tex").read_text(encoding="utf-8")
    for name in re.findall(r"\\input\{([^}]+)\}", text):
        candidate = MANUSCRIPT / (name if name.endswith(".tex") else f"{name}.tex")
        if not candidate.is_file():
            missing.append(name)
    # Inputs in section and appendix files are relative to the manuscript root in TeX.
    for path in tex_files():
        for name in re.findall(r"\\input\{([^}]+)\}", path.read_text(encoding="utf-8")):
            candidate = MANUSCRIPT / (name if name.endswith(".tex") else f"{name}.tex")
            if not candidate.is_file():
                missing.append(f"{path.relative_to(MANUSCRIPT)} -> {name}")
    return sorted(set(missing))


def figure_audit(text: str | None = None) -> list[str]:
    text = text if text is not None else all_tex()
    missing: list[str] = []
    for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
        candidates = [MANUSCRIPT / "figures" / name, PAPER / "figures" / name]
        if not Path(name).suffix:
            candidates = [candidate.with_suffix(ext) for candidate in candidates for ext in (".pdf", ".png", ".jpg")]
        if not any(candidate.is_file() for candidate in candidates):
            missing.append(name)
    return sorted(set(missing))


def read_csv(name: str) -> list[dict[str, str]]:
    with (DERIVED / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def evidence_audit(text: str) -> dict[str, Any]:
    overall = read_csv("overall_results.csv")[0]
    routes = {row["route_id"]: row for row in read_csv("route_statistics.csv")}
    models = {row["model_key"]: row for row in read_csv("model_transfer.csv")}
    policy = json.loads((RAW / "stage06_frozen_policy.json").read_text(encoding="utf-8"))

    exact_checks = {
        "policy_success": math.isclose(float(overall["policy_success_rate"]), 0.0),
        "policy_energy": math.isclose(float(overall["policy_mean_gpu_joules"]), 143.1556540192412, abs_tol=1e-12),
        "best_route_success": math.isclose(float(routes["M18_granite_grounded_verified"]["success_rate"]), 53 / 120, abs_tol=1e-12),
        "tiny_effect": math.isclose(float(models["tiny"]["success_diff_mean"]), -15 / 120, abs_tol=1e-12),
        "granite_effect": math.isclose(float(models["granite"]["success_diff_mean"]), 28 / 120, abs_tol=1e-12),
        "upper_effect": math.isclose(float(models["upper"]["success_diff_mean"]), 35 / 120, abs_tol=1e-12),
        "policy_tau": math.isclose(float(policy["tau"]), 1.0),
        "validation_infeasible": policy["validation_selection"]["feasible"] is False,
        "test_sealed_at_freeze": policy["test_accessed"] is False and policy["validation_selection"]["test_accessed"] is False,
        "amendment_forbids_claim": policy["validation_selection"]["protocol_amendment"]["hypothesis_claim_allowed_when_infeasible"] is False,
        "stage06_hash": sha256(RAW / "stage06_frozen_policy.json") == "fda2bda0a715227162ceead0ff032f2228999f6dc9f4bb22890ec23a4acdab14",
    }
    if not all(exact_checks.values()):
        raise RuntimeError(f"Frozen evidence audit failed: {exact_checks}")

    required_statements = {
        "generation_window_boundary": "generation-window selected-GPU board energy" in text,
        "cpu_excluded": "CPU" in text and "outside" in text,
        "carbon_excluded": "carbon" in text and "not" in text,
        "validation_infeasibility": "validation infeasibility" in text or "validation constraints were infeasible" in text,
        "pre_test_amendment": "before accessing test labels" in text or "before test-label access" in text,
        "hypothesis_failed": "confirmatory hypothesis failed" in text or "joint hypothesis therefore failed" in text,
        "robustness_floor": "floor effect" in text,
        "granite_compatibility": "compatibility failure" in text or "incompatibility" in text,
        "board_confounding": "cross-board" in text,
    }
    if not all(required_statements.values()):
        raise RuntimeError(f"Required disclosure missing: {required_statements}")

    forbidden = [
        "end-to-end selected-GPU board-energy accounting",
        "the router reduced energy",
        "the router was robust",
        "whole-system energy was measured",
        "carbon emissions were measured",
        "the verifier proves factuality",
    ]
    present = [phrase for phrase in forbidden if phrase.lower() in text.lower()]
    if present:
        raise RuntimeError(f"Forbidden overclaim found: {present}")

    return {"exact_checks": exact_checks, "required_statements": required_statements, "forbidden_present": present}


def unicode_audit() -> list[dict[str, Any]]:
    findings = []
    for path in [*tex_files(), MANUSCRIPT / "references.bib", MANUSCRIPT / "BUILD.md"]:
        chars = sorted({char for char in path.read_text(encoding="utf-8") if ord(char) > 127})
        if chars:
            findings.append({"path": path.relative_to(ROOT).as_posix(), "codepoints": [f"U+{ord(char):04X}" for char in chars]})
    return findings


def validate() -> dict[str, Any]:
    text = all_tex()
    bib = bibliography_keys()
    citations = citation_keys(text)
    duplicates = sorted({key for key in bib if bib.count(key) > 1})
    unresolved_citations = sorted(citations - set(bib))
    unused_bibliography = sorted(set(bib) - citations)
    labels = label_audit(text)
    missing_inputs = input_audit()
    missing_figures = figure_audit(text)
    unicode_findings = unicode_audit()
    evidence = evidence_audit(text)
    errors = {
        "duplicate_bib_keys": duplicates,
        "unresolved_citations": unresolved_citations,
        "unused_bibliography": unused_bibliography,
        "duplicate_labels": labels["duplicates"],
        "unresolved_references": labels["unresolved"],
        "missing_inputs": missing_inputs,
        "missing_figures": missing_figures,
        "unicode_findings": unicode_findings,
    }
    if any(errors.values()):
        raise RuntimeError(f"MANUSCRIPT_VALIDATION_FAILED: {json.dumps(errors, sort_keys=True)}")
    report = {
        "status": "PASS",
        "tex_files": len(tex_files()),
        "bibliography_entries": len(bib),
        "citations_used": len(citations),
        "labels": labels["labels"],
        "references": labels["references"],
        "figures": len(re.findall(r"\\includegraphics", text)),
        "tables": len(re.findall(r"\\begin\{table\*?\}", text)),
        "evidence": evidence,
    }
    (MANUSCRIPT / "VALIDATION_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    report = validate()
    print("MANUSCRIPT_VALIDATION_PASS", {key: report[key] for key in ("tex_files", "bibliography_entries", "citations_used", "labels", "figures", "tables")})


if __name__ == "__main__":
    main()

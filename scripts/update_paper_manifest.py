"""Finalize the checksum inventory after every paper artifact is generated."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
MANIFEST = PAPER / "RESULTS_MANIFEST.json"

METADATA = {
    "schema_version": 1,
    "experiment_id": "e2am-memrag-v3r1",
    "source_repo": "Shanmuk4622/E2AM-MemRAG-Traces",
    "source_main_commit": "0b2405d9cca43fd04e35f792fdc4664405154fc6",
    "source_paper_commit": "00fa353f273f3a4b3d57a0b998301c85a1bc098b",
    "hypothesis_pass": False,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def included_files() -> list[Path]:
    files: list[Path] = []
    for path in PAPER.rglob("*"):
        if not path.is_file() or path == MANIFEST:
            continue
        relative = path.relative_to(PAPER)
        if "__pycache__" in relative.parts:
            continue
        if relative.parts[:2] == ("manuscript", "build"):
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def main() -> None:
    records = []
    for path in included_files():
        records.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    if len(records) < 30:
        raise RuntimeError(f"Paper inventory is unexpectedly small: {len(records)} files")
    manifest = {**METADATA, "files": records}
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(
        "PAPER_MANIFEST_FINALIZED",
        {"files": len(records), "manifest": MANIFEST.relative_to(ROOT).as_posix()},
    )


if __name__ == "__main__":
    main()

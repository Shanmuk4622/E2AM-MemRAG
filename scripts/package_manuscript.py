"""Create deterministic PDF and Overleaf deliverables for E2AM-MemRAG."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "paper" / "manuscript"
BUILD_PDF = MANUSCRIPT / "build" / "main.pdf"
OUTPUT = ROOT / "output"
PDF_OUTPUT = OUTPUT / "pdf" / "E2AM_MemRAG_Paper.pdf"
ZIP_OUTPUT = OUTPUT / "latex" / "E2AM_MemRAG_Overleaf.zip"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def source_files() -> list[Path]:
    allowed_suffixes = {".tex", ".bib", ".png", ".svg", ".md", ".json"}
    return sorted(
        path
        for path in MANUSCRIPT.rglob("*")
        if path.is_file()
        and "build" not in path.relative_to(MANUSCRIPT).parts
        and path.suffix.lower() in allowed_suffixes
    )


def package() -> dict[str, object]:
    validation_path = MANUSCRIPT / "VALIDATION_REPORT.json"
    if not validation_path.is_file():
        raise RuntimeError("Run scripts/validate_manuscript.py before packaging")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "PASS":
        raise RuntimeError("The manuscript validation report is not PASS")
    if not BUILD_PDF.is_file() or BUILD_PDF.stat().st_size < 100_000:
        raise RuntimeError("Compile paper/manuscript/main.tex before packaging")
    if not BUILD_PDF.read_bytes().startswith(b"%PDF-"):
        raise RuntimeError("The compiled manuscript is not a PDF")

    PDF_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ZIP_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(BUILD_PDF, PDF_OUTPUT)

    files = source_files()
    inventory = []
    for path in files:
        payload = path.read_bytes()
        inventory.append(
            {
                "path": path.relative_to(MANUSCRIPT).as_posix(),
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
        )
    manifest = {
        "schema_version": 1,
        "paper": "E2AM-MemRAG",
        "entrypoint": "main.tex",
        "files": inventory,
    }
    manifest_payload = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")

    temporary = ZIP_OUTPUT.with_suffix(".zip.tmp")
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(MANUSCRIPT).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
        info = zipfile.ZipInfo("PACKAGE_MANIFEST.json", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, manifest_payload)
    temporary.replace(ZIP_OUTPUT)

    result = {
        "pdf": PDF_OUTPUT.relative_to(ROOT).as_posix(),
        "pdf_bytes": PDF_OUTPUT.stat().st_size,
        "pdf_sha256": sha256_bytes(PDF_OUTPUT.read_bytes()),
        "overleaf_zip": ZIP_OUTPUT.relative_to(ROOT).as_posix(),
        "zip_bytes": ZIP_OUTPUT.stat().st_size,
        "zip_sha256": sha256_bytes(ZIP_OUTPUT.read_bytes()),
        "source_files": len(files),
    }
    (OUTPUT / "PACKAGE_REPORT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print("MANUSCRIPT_PACKAGE_READY", package())

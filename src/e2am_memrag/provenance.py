from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .identity import make_config_hash
from .utils import canonical_json, sha256_file


_SECRET_KEYS = {
    "token",
    "hf_token",
    "access_token",
    "api_token",
    "password",
    "secret",
    "api_key",
    "apikey",
    "credential",
    "credentials",
}


def _looks_secret_key(normalized: str) -> bool:
    if normalized in _SECRET_KEYS:
        return True
    return normalized.endswith(
        ("_token", "_password", "_secret", "_api_key", "_credential", "_credentials")
    ) or normalized.startswith(("secret_", "password_", "credential_"))


def assert_secret_free(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if _looks_secret_key(normalized):
                raise ValueError(f"Secret-like key is forbidden in provenance: {path}.{key}")
            assert_secret_free(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_secret_free(child, f"{path}[{index}]")


def _run_git(root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def source_fingerprint(
    root: str | Path,
    include_roots: Iterable[str] = ("src", "configs", "notebooks", "scripts", "tests"),
) -> dict[str, Any]:
    project = Path(root).resolve()
    files: list[Path] = []
    for relative_root in include_roots:
        candidate = project / relative_root
        if candidate.is_file():
            files.append(candidate)
        elif candidate.is_dir():
            files.extend(
                path
                for path in candidate.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and ".ipynb_checkpoints" not in path.parts
            )
    for name in ("pyproject.toml", "requirements-kaggle.txt"):
        candidate = project / name
        if candidate.is_file():
            files.append(candidate)
    entries = [
        {
            "path": path.relative_to(project).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(set(files))
    ]
    tree_hash = hashlib.sha256(canonical_json(entries).encode("utf-8")).hexdigest()
    status = _run_git(project, "status", "--porcelain=v1", "--untracked-files=all")
    return {
        "git_commit": _run_git(project, "rev-parse", "HEAD"),
        "git_status_sha256": hashlib.sha256((status or "").encode("utf-8")).hexdigest(),
        "tree_sha256": tree_hash,
        "file_count": len(entries),
        "files": entries,
    }


@dataclass(frozen=True)
class ExperimentSpec:
    payload: dict[str, Any]
    spec_hash: str

    @classmethod
    def create(cls, **sections: Any) -> "ExperimentSpec":
        payload = {"schema_version": 1, **sections}
        assert_secret_free(payload)
        return cls(payload=payload, spec_hash=make_config_hash(payload, length=20))

    def as_dict(self) -> dict[str, Any]:
        return {"spec_hash": self.spec_hash, "spec": self.payload}

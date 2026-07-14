from __future__ import annotations

import hashlib
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar

from .utils import canonical_json


_Result = TypeVar("_Result")


def _http_status(error: BaseException) -> int | None:
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _retry_after(error: BaseException) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {}) or {}
    value = headers.get("Retry-After") or headers.get("retry-after")
    try:
        return max(0.0, float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _hub_call(operation: Callable[[], _Result], *, attempts: int = 5) -> _Result:
    """Retry transient Hub failures without retrying authentication errors."""
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as error:
            status = _http_status(error)
            retryable = status == 429 or (status is not None and 500 <= status < 600)
            if not retryable or attempt + 1 == attempts:
                raise
            delay = _retry_after(error)
            if delay is None:
                delay = min(60.0, 2.0 ** attempt)
            time.sleep(min(60.0, delay))
    raise RuntimeError("Unreachable Hub retry state")


@dataclass(frozen=True)
class HubSourcePaths:
    archive: str
    manifest: str
    environment_pin: str
    pointer: str = "source-bundles/LATEST.json"


def hub_source_paths(
    *,
    archive_sha256: str,
    experiment_id: str,
    source_tree_sha256: str,
    environment_sha256: str,
) -> HubSourcePaths:
    """Return content-addressed Hub paths without embedding credentials."""
    for label, value in (
        ("archive_sha256", archive_sha256),
        ("source_tree_sha256", source_tree_sha256),
        ("environment_sha256", environment_sha256),
    ):
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"{label} must be a lowercase SHA-256")
    if not experiment_id or "/" in experiment_id or ".." in experiment_id:
        raise ValueError("experiment_id is unsafe for a Hub path")
    prefix = f"source-bundles/{archive_sha256}"
    return HubSourcePaths(
        archive=f"{prefix}/e2am-memrag-runtime.zip",
        manifest=f"{prefix}/manifest.json",
        environment_pin=(
            f"preflight-pins/{experiment_id}/{source_tree_sha256}/"
            f"{environment_sha256}.json"
        ),
    )


def _download_exact(
    *,
    repo_id: str,
    revision: str,
    remote_path: str,
    cache_root: Path,
    token: str,
) -> bytes:
    from huggingface_hub import hf_hub_download

    local_path = Path(
        _hub_call(
            lambda: hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                filename=remote_path,
                token=token,
                local_dir=cache_root,
            )
        )
    )
    return local_path.read_bytes()


def verify_hub_source_snapshot(
    *,
    repo_id: str,
    revision: str,
    token: str,
    paths: HubSourcePaths,
    archive_bytes: bytes,
    manifest_bytes: bytes,
    environment_pin_bytes: bytes,
    cache_root: str | Path,
) -> dict[str, Any]:
    """Download one pinned revision and compare every byte with the local contract."""
    cache = Path(cache_root)
    cache.mkdir(parents=True, exist_ok=True)
    downloaded_archive = _download_exact(
        repo_id=repo_id,
        revision=revision,
        remote_path=paths.archive,
        cache_root=cache,
        token=token,
    )
    expected_archive_sha = hashlib.sha256(archive_bytes).hexdigest()
    if hashlib.sha256(downloaded_archive).hexdigest() != expected_archive_sha:
        raise RuntimeError("Hub source archive hash differs from the embedded archive")
    if _download_exact(
        repo_id=repo_id,
        revision=revision,
        remote_path=paths.manifest,
        cache_root=cache,
        token=token,
    ) != manifest_bytes:
        raise RuntimeError("Hub source manifest differs from the local source contract")
    if _download_exact(
        repo_id=repo_id,
        revision=revision,
        remote_path=paths.pointer,
        cache_root=cache,
        token=token,
    ) != manifest_bytes:
        raise RuntimeError("Hub source pointer differs from the local source contract")
    if _download_exact(
        repo_id=repo_id,
        revision=revision,
        remote_path=paths.environment_pin,
        cache_root=cache,
        token=token,
    ) != environment_pin_bytes:
        raise RuntimeError("Hub environment pin differs from the local runtime contract")
    return {
        "status": "HUB_SOURCE_VERIFIED",
        "repo_id": repo_id,
        "revision_sha": revision,
        "archive_path": paths.archive,
        "archive_sha256": expected_archive_sha,
        "pointer_path": paths.pointer,
        "environment_pin_path": paths.environment_pin,
    }


def publish_hub_source_snapshot(
    *,
    repo_id: str,
    base_revision: str,
    token: str,
    paths: HubSourcePaths,
    archive_bytes: bytes,
    manifest: Mapping[str, Any],
    environment_pin: Mapping[str, Any],
    cache_root: str | Path,
) -> dict[str, Any]:
    """Create at most one commit for missing immutable source objects, then verify."""
    from huggingface_hub import CommitOperationAdd, HfApi

    manifest_bytes = (canonical_json(dict(manifest)) + "\n").encode("utf-8")
    environment_pin_bytes = (canonical_json(dict(environment_pin)) + "\n").encode(
        "utf-8"
    )
    api = HfApi(token=token)
    _hub_call(
        lambda: api.create_repo(
            repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True
        )
    )
    remote_files = set(
        _hub_call(
            lambda: api.list_repo_files(
                repo_id=repo_id, repo_type="dataset", revision=base_revision
            )
        )
    )
    payloads = {
        paths.archive: archive_bytes,
        paths.manifest: manifest_bytes,
        paths.pointer: manifest_bytes,
        paths.environment_pin: environment_pin_bytes,
    }
    pointer_needs_update = paths.pointer not in remote_files
    if not pointer_needs_update:
        pointer_needs_update = _download_exact(
            repo_id=repo_id,
            revision=base_revision,
            remote_path=paths.pointer,
            cache_root=Path(cache_root),
            token=token,
        ) != manifest_bytes
    operations = [
        CommitOperationAdd(path_in_repo=remote_path, path_or_fileobj=io.BytesIO(payload))
        for remote_path, payload in payloads.items()
        if remote_path not in remote_files
        or (remote_path == paths.pointer and pointer_needs_update)
    ]
    if operations:
        _hub_call(
            lambda: api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                revision=base_revision,
                operations=operations,
                commit_message=(
                    f"bootstrap source {hashlib.sha256(archive_bytes).hexdigest()[:12]} "
                    f"and environment {environment_pin['environment_sha256'][:12]}"
                ),
            )
        )
    revision_sha = _hub_call(
        lambda: api.repo_info(
            repo_id=repo_id,
            repo_type="dataset",
            revision=base_revision,
        )
    ).sha
    return verify_hub_source_snapshot(
        repo_id=repo_id,
        revision=revision_sha,
        token=token,
        paths=paths,
        archive_bytes=archive_bytes,
        manifest_bytes=manifest_bytes,
        environment_pin_bytes=environment_pin_bytes,
        cache_root=cache_root,
    )


def verify_published_source_snapshot(
    *,
    repo_id: str,
    base_revision: str,
    token: str,
    paths: HubSourcePaths,
    archive_bytes: bytes,
    manifest: Mapping[str, Any],
    environment_pin: Mapping[str, Any],
    cache_root: str | Path,
) -> dict[str, Any]:
    """Resolve the current base HEAD, then verify all source objects at that SHA."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    revision_sha = _hub_call(
        lambda: api.repo_info(
            repo_id=repo_id,
            repo_type="dataset",
            revision=base_revision,
        )
    ).sha
    return verify_hub_source_snapshot(
        repo_id=repo_id,
        revision=revision_sha,
        token=token,
        paths=paths,
        archive_bytes=archive_bytes,
        manifest_bytes=(canonical_json(dict(manifest)) + "\n").encode("utf-8"),
        environment_pin_bytes=(canonical_json(dict(environment_pin)) + "\n").encode(
            "utf-8"
        ),
        cache_root=cache_root,
    )

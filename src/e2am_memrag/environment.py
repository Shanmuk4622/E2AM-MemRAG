from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from .utils import utc_now


def select_single_gpu(physical_gpu_index: int = 0) -> None:
    """Call before importing torch, transformers, or any CUDA-aware library."""
    if "torch" in sys.modules:
        raise RuntimeError(
            "torch is already imported. Restart the kernel and run the GPU-selection cell first."
        )
    if physical_gpu_index < 0:
        raise ValueError("physical_gpu_index must be non-negative")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_index)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONHASHSEED", "4622")


def _nvidia_smi() -> str | None:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    result = subprocess.run(
        [
            executable,
            "--query-gpu=index,name,uuid,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _nvidia_smi_contract() -> str | None:
    """Return hardware/driver properties without machine-specific GPU UUIDs."""
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    result = subprocess.run(
        [
            executable,
            "--id=0",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def environment_contract() -> dict[str, Any]:
    """Build the deterministic environment section used in an experiment spec.

    Volatile values (timestamps, paths, process IDs and GPU UUIDs) are excluded so
    a restart on an equivalent Kaggle T4 produces the same spec hash. Exact package
    versions are retained to prevent silently mixing incompatible environments.
    """
    packages: dict[str, str | None] = {}
    for name in (
        "e2am-memrag",
        "huggingface-hub",
        "nvidia-ml-py",
        "PyYAML",
        "torch",
        "transformers",
        "accelerate",
        "tokenizers",
        "safetensors",
        "hf-xet",
        "numpy",
        "scipy",
        "scikit-learn",
        "joblib",
        "bitsandbytes",
        "faiss-cpu",
    ):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    contract: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "cuda_visible_device_count_required": 1,
        "nvidia_smi": _nvidia_smi_contract(),
        "packages": packages,
    }
    try:
        import torch

        contract["torch_runtime"] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "visible_device_count": torch.cuda.device_count(),
        }
    except Exception as error:
        contract["torch_runtime_error_type"] = type(error).__name__
    return contract


def collect_environment(include_pip_freeze: bool = False) -> dict[str, Any]:
    info: dict[str, Any] = {
        "captured_at": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": _nvidia_smi(),
    }
    try:
        import torch

        info["torch"] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "visible_device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as error:
        info["torch_error"] = f"{type(error).__name__}: {error}"
    if include_pip_freeze:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        info["pip_freeze"] = result.stdout.splitlines() if result.returncode == 0 else []
    return info


def preflight(
    work_root: str | Path,
    minimum_free_gib: float = 5.0,
    *,
    projected_download_bytes: int = 0,
    maximum_used_fraction: float = 0.75,
    emergency_reserve_gib: float = 2.0,
) -> dict[str, Any]:
    if projected_download_bytes < 0:
        raise ValueError("projected_download_bytes cannot be negative")
    if not 0 < maximum_used_fraction < 1:
        raise ValueError("maximum_used_fraction must be between zero and one")
    if emergency_reserve_gib < 0:
        raise ValueError("emergency_reserve_gib cannot be negative")
    root = Path(work_root)
    root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(root)
    free_gib = usage.free / (1024**3)
    projected_used = usage.used + projected_download_bytes
    projected_free = usage.free - projected_download_bytes
    reserve_bytes = int(emergency_reserve_gib * (1024**3))
    storage_ok = (
        projected_free >= reserve_bytes
        and projected_used / usage.total <= maximum_used_fraction
    )
    checks = {
        "work_root": str(root.resolve()),
        "total_gib": usage.total / (1024**3),
        "used_gib": usage.used / (1024**3),
        "free_gib": free_gib,
        "minimum_free_gib": minimum_free_gib,
        "projected_download_gib": projected_download_bytes / (1024**3),
        "projected_used_fraction": projected_used / usage.total,
        "maximum_used_fraction": maximum_used_fraction,
        "emergency_reserve_gib": emergency_reserve_gib,
        "storage_ok": storage_ok,
        "disk_ok": free_gib >= minimum_free_gib and storage_ok,
        "hf_token_present": bool(os.environ.get("HF_TOKEN")),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        checks["cuda_available"] = torch.cuda.is_available()
        checks["visible_gpu_count"] = torch.cuda.device_count()
        checks["single_gpu_mask_ok"] = torch.cuda.device_count() == 1
        checks["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as error:
        checks["torch_error"] = f"{type(error).__name__}: {error}"
        checks["single_gpu_mask_ok"] = False
    return checks

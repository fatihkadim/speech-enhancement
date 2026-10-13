"""Tekrar üretilebilirlik: tohum ayarlama ve çalıştırma bilgisi (git commit, sürümler)."""
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """random, numpy ve torch (CPU + CUDA) tohumlarını ayarlar."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(worker_id: int) -> None:
    """DataLoader worker'larına torch'un verdiği tohumu numpy ve random'a da aktarır."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None  # git yoksa veya klasör bir depo değilse bilgi yazılmaz


def run_info(root: Path) -> dict:
    """Deney klasörüne yazılacak ortam bilgisi."""
    status = _git(root, "status", "--porcelain")
    return {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

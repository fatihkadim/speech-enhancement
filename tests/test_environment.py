"""M0 duman testleri: bağımlılıklar yüklü, paket içe aktarılabilir, config okunabilir."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_core_imports() -> None:
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import scipy  # noqa: F401
    import soundfile  # noqa: F401
    import torch
    import torchaudio  # noqa: F401
    import pystoi  # noqa: F401

    assert torch.__version__


def test_src_package_importable() -> None:
    import src  # noqa: F401
    import src.data  # noqa: F401
    import src.models  # noqa: F401
    import src.utils  # noqa: F401


def test_baseline_config_loads() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "baseline.yaml").read_text(encoding="utf-8"))
    for key in ("seed", "data", "stft", "model", "loss", "train", "output_dir"):
        assert key in cfg
    assert cfg["data"]["sr"] == 16000
    assert cfg["stft"]["n_fft"] == 512
    assert cfg["model"]["out_mode"] in ("mask", "direct")
    assert cfg["loss"]["name"] in ("l1", "mse", "si_snr")

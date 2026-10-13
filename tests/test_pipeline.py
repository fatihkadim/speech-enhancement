"""train.py, evaluate.py ve infer.py için sahte mini veri setinde uçtan uca testler (CPU, minik model)."""
import json
import math
from pathlib import Path

import pandas as pd
import pytest
import soundfile as sf
import torch

import src.evaluate as evaluate_mod
from src.data.make_dataset import build_dataset
from src.evaluate import evaluate
from src.infer import enhance_file
from src.train import train
from tests.test_make_dataset import SR, make_cfg, make_fake_corpus


def pipeline_cfg(output_dir: str = "experiments/test_run", loss: str = "l1") -> dict:
    cfg = make_cfg()
    cfg |= {
        "device": "cpu",
        "model": {"name": "unet", "base_ch": 4, "depth": 2, "skip": True, "out_mode": "mask"},
        "loss": {"name": loss},
        "train": {"batch_size": 4, "epochs": 2, "lr": 1e-3, "amp": True, "grad_clip": 5.0, "num_workers": 0,
                  "early_stop_patience": 8, "lr_factor": 0.5, "lr_patience": 3},
        "eval": {"segment_seconds": None},
        "output_dir": output_dir,
    }
    return cfg


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("pipeline")
    make_fake_corpus(root)
    build_dataset(pipeline_cfg(), root)
    return root, train(pipeline_cfg(), root)


def test_train_writes_outputs(trained: tuple[Path, Path]) -> None:
    _, out = trained
    for name in ("config.yaml", "run_info.json", "log.csv", "best.pt", "last.pt"):
        assert (out / name).exists(), name
    log = pd.read_csv(out / "log.csv")
    assert list(log["epoch"]) == [1, 2]
    assert log[["train_loss", "val_loss", "val_si_snr"]].notna().all().all()
    assert "git_commit" in json.loads((out / "run_info.json").read_text(encoding="utf-8"))


def test_train_refuses_existing_output(trained: tuple[Path, Path]) -> None:
    root, _ = trained
    with pytest.raises(FileExistsError):
        train(pipeline_cfg(), root)


def test_train_with_si_snr_loss(trained: tuple[Path, Path]) -> None:
    root, _ = trained
    out = train(pipeline_cfg("experiments/test_si_snr", loss="si_snr"), root)
    assert (out / "best.pt").exists()


def test_evaluate_model_and_e0(trained: tuple[Path, Path]) -> None:
    root, out = trained
    summary = evaluate(pipeline_cfg(), root, "test", out / "eval_test", checkpoint=out / "best.pt")
    df = pd.read_csv(out / "eval_test" / "metrics_per_sample.csv")
    assert len(df) == summary["n"] == 6
    for prefix in ("noisy", "enhanced", "delta"):
        assert summary["metrics"][f"{prefix}_si_snr"]["n"] == 6
        assert summary["metrics"][f"{prefix}_stoi"] is not None
    assert set(summary["by_snr"]) and set(summary["by_noise_class"])

    e0 = evaluate(pipeline_cfg(), root, "test_unseen_noise", root / "experiments/E0/test_unseen_noise")
    assert "enhanced_si_snr" not in e0["metrics"] and e0["metrics"]["noisy_si_snr"]["n"] == 6


def test_evaluate_without_pesq_does_not_crash(trained: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    root, _ = trained
    monkeypatch.setattr(evaluate_mod, "_pesq", None)
    summary = evaluate(pipeline_cfg(), root, "test", root / "experiments/E0_nopesq")
    assert summary["pesq_available"] is False and summary["metrics"]["noisy_pesq"] is None


def test_evaluate_rejects_train_split_and_overwrite(trained: tuple[Path, Path]) -> None:
    root, out = trained
    with pytest.raises(ValueError):
        evaluate(pipeline_cfg(), root, "train", root / "x")
    evaluate(pipeline_cfg(), root, "test", root / "experiments/E0_once")
    with pytest.raises(FileExistsError):
        evaluate(pipeline_cfg(), root, "test", root / "experiments/E0_once")


def test_infer_writes_same_length(trained: tuple[Path, Path], tmp_path: Path) -> None:
    root, out = trained
    noisy = next((root / "data/processed/test/noisy").glob("*.wav"))
    result = enhance_file(out / "best.pt", noisy, tmp_path / "temiz.wav")
    est, sr = sf.read(result)
    assert sr == SR and len(est) == sf.info(noisy).frames and not math.isnan(est.sum())
    with pytest.raises(ValueError):
        enhance_file(out / "best.pt", noisy, noisy)

"""src/collect_results.py testleri: eşleştirilmiş fark, tohum istatistiği ve markdown üretimi."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.collect_results import by_snr_table, experiment_table, load_per_sample, to_markdown, unseen_table

MATRIX = {"output_root": "exp", "seeds": [1, 2], "experiments": {"baseline": {}, "better": {}}}


def _write(root: Path, name: str, seed: int, split: str, offset: float) -> None:
    rng = np.random.default_rng(seed)
    n = 40
    noisy = rng.normal(5, 3, n)
    enhanced = noisy + 8 + offset + rng.normal(0, 0.1, n)
    df = pd.DataFrame({"id": [f"u{i}" for i in range(n)], "snr_db": np.tile([-5.0, 0.0, 5.0, 10.0], n // 4),
                       "noise_class": "dog"})
    for m, base in (("si_snr", 0.0), ("pesq", 1.5), ("stoi", 0.8)):
        df[f"noisy_{m}"] = noisy if m == "si_snr" else base
        df[f"enhanced_{m}"] = enhanced if m == "si_snr" else base + 0.1 + offset / 100
        df[f"delta_{m}"] = df[f"enhanced_{m}"] - df[f"noisy_{m}"]
    path = root / "exp" / name / f"seed{seed}" / f"eval_{split}" / "metrics_per_sample.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


@pytest.fixture
def results(tmp_path: Path) -> pd.DataFrame:
    for name, offset in (("baseline", 0.0), ("better", 1.0)):
        for seed in MATRIX["seeds"]:
            for split in ("test", "test_unseen_noise"):
                _write(tmp_path, name, seed, split, offset)
    return load_per_sample(MATRIX, tmp_path)


def test_paired_difference_detects_real_gap(results: pd.DataFrame) -> None:
    t = experiment_table(results, "test").set_index("experiment")
    better = t.loc["better"]
    assert better["si_snr_vs_base"] == pytest.approx(1.0, abs=0.05)
    assert better["si_snr_vs_base_lo"] > 0  # 1 dB'lik gerçek fark tesadüften ayrılmalı
    assert t.loc["baseline", "seeds"] == 2
    assert t.loc["E0 (gürültülü girdi)", "si_snr_mean"] == pytest.approx(results.query("split=='test' and experiment=='baseline' and seed==1")["noisy_si_snr"].mean())


def test_tables_and_markdown(results: pd.DataFrame) -> None:
    snr = by_snr_table(results)
    assert set(snr[0]["snr_db"]) == {-5.0, 0.0, 5.0, 10.0}
    unseen = unseen_table(results)
    assert {"test_mean", "test_unseen_noise_mean"} <= set(unseen.columns)
    md = to_markdown({s: experiment_table(results, s) for s in ("test", "test_unseen_noise")}, snr, unseen)
    assert "| better |" in md and " *" in md


def test_missing_result_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_per_sample(MATRIX, tmp_path)

"""src/run_experiments.py testleri: config birleştirme, plan ve kaldığı yerden devam."""
from pathlib import Path

import pytest
import yaml

from src.data.make_dataset import build_dataset
from src.run_experiments import TRAIN_DONE, deep_merge, plan_runs, run_one
from tests.test_make_dataset import make_fake_corpus
from tests.test_pipeline import pipeline_cfg

ROOT = Path(__file__).resolve().parents[1]


def test_deep_merge_changes_only_given_keys() -> None:
    base = {"model": {"depth": 4, "skip": True}, "loss": {"name": "l1"}}
    merged = deep_merge(base, {"model": {"skip": False}})
    assert merged == {"model": {"depth": 4, "skip": False}, "loss": {"name": "l1"}}
    assert base["model"]["skip"] is True  # orijinal değişmez


def test_project_matrix_changes_one_setting_per_experiment() -> None:
    """Gerçek matristeki her deney baseline'dan tam olarak bir yaprak ayar değiştirmeli."""
    base = yaml.safe_load((ROOT / "configs/baseline.yaml").read_text(encoding="utf-8"))
    matrix = yaml.safe_load((ROOT / "configs/experiments.yaml").read_text(encoding="utf-8"))

    def leaves(d: dict, prefix: str = "") -> dict:
        out = {}
        for k, v in d.items():
            out |= leaves(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v}
        return out

    base_leaves = leaves(base)
    for name, override in matrix["experiments"].items():
        changed = {k for k, v in leaves(deep_merge(base, override or {})).items() if base_leaves.get(k) != v}
        assert len(changed) == (0 if name == "baseline" else 1), (name, changed)


def test_plan_orders_by_seed_and_sets_output() -> None:
    matrix = {"output_root": "exp", "seeds": [1, 2], "experiments": {"a": {}, "b": {"loss": {"name": "mse"}}}}
    runs = plan_runs({"loss": {"name": "l1"}}, matrix)
    assert [(n, s) for n, s, _ in runs] == [("a", 1), ("b", 1), ("a", 2), ("b", 2)]
    assert runs[1][2]["output_dir"] == "exp/b/seed1" and runs[1][2]["loss"]["name"] == "mse"
    with pytest.raises(ValueError):
        plan_runs({}, matrix, only={"yok"})


def test_run_one_resumes(tmp_path: Path) -> None:
    make_fake_corpus(tmp_path)
    build_dataset(pipeline_cfg(), tmp_path)
    cfg = pipeline_cfg("experiments/x/seed1") | {"train": pipeline_cfg()["train"] | {"epochs": 1}}
    out = tmp_path / cfg["output_dir"]

    out.mkdir(parents=True)
    (out / "log.csv").write_text("yarım")  # yarıda kalmış çalıştırma taklidi
    run_one(cfg, tmp_path)
    assert (out / TRAIN_DONE).exists()
    assert all((out / f"eval_{s}" / "summary.json").exists() for s in ("test", "test_unseen_noise"))
    assert len(list(out.parent.glob("seed1_yarim_*"))) == 1  # silinmedi, kenara alındı

    best_mtime = (out / "best.pt").stat().st_mtime
    run_one(cfg, tmp_path)  # ikinci çağrı hiçbir şeyi yeniden yapmamalı
    assert (out / "best.pt").stat().st_mtime == best_mtime

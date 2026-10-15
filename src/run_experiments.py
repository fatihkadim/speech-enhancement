"""Deney matrisini sırayla eğitir ve iki test bölmesinde değerlendirir; yarıda kalırsa kaldığı yerden devam eder.

Kullanım:
    uv run python -m src.run_experiments --base configs/baseline.yaml --matrix configs/experiments.yaml
    ... --only baseline,e1_mse     # yalnızca bu deneyler
    ... --dry-run                  # çalıştırmadan planı göster

Sıra: önce tüm deneylerin ilk tohumu, sonra ikinci tohum...; böylece yarıda kesilse bile her deneyin sonucu olur.
Her çalıştırma `<output_root>/<deney>/seed<tohum>/` altına yazılır. Eğitim bitince `train_done.json` yazılır;
bu dosya olmadan dolu bulunan klasör yarım kalmış sayılır, silinmez, `_yarim_<zaman>` ekiyle kenara alınır.
"""
import argparse
import copy
import json
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path

import yaml

from src.evaluate import EVAL_SPLITS, evaluate
from src.train import train

log = logging.getLogger(__name__)
TRAIN_DONE = "train_done.json"


def deep_merge(base: dict, override: dict) -> dict:
    """`base`'in kopyasına `override`'ı iç içe sözlükleri koruyarak uygular."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def plan_runs(base_cfg: dict, matrix: dict, only: set[str] | None = None) -> list[tuple[str, int, dict]]:
    """(deney adı, tohum, tam config) listesi; tohum dış döngüde."""
    unknown = (only or set()) - set(matrix["experiments"])
    if unknown:
        raise ValueError(f"Matriste olmayan deneyler: {sorted(unknown)}")
    runs = []
    for seed in matrix["seeds"]:
        for name, override in matrix["experiments"].items():
            if only and name not in only:
                continue
            cfg = deep_merge(base_cfg, override or {})
            cfg["seed"] = seed
            cfg["output_dir"] = f"{matrix['output_root']}/{name}/seed{seed}"
            runs.append((name, seed, cfg))
    return runs


def run_one(cfg: dict, root: Path) -> None:
    """Gerekiyorsa eğitir, eksik değerlendirmeleri yapar."""
    out = root / cfg["output_dir"]
    if (out / TRAIN_DONE).exists():
        log.info("eğitim zaten bitmiş, atlanıyor: %s", out)
    else:
        if out.exists() and any(out.iterdir()):
            aside = out.with_name(f"{out.name}_yarim_{datetime.now():%Y%m%d_%H%M%S}")
            out.rename(aside)
            log.warning("yarım kalmış çalıştırma kenara alındı: %s", aside)
        start = time.perf_counter()
        train(cfg, root)
        (out / TRAIN_DONE).write_text(json.dumps({"minutes": round((time.perf_counter() - start) / 60, 1)}),
                                      encoding="utf-8")
    for split in EVAL_SPLITS:
        if not (out / f"eval_{split}" / "summary.json").exists():
            evaluate(cfg, root, split, out / f"eval_{split}", checkpoint=out / "best.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deney matrisini çalıştır")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--only", help="virgülle ayrılmış deney adları")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    base_cfg = yaml.safe_load(args.base.read_text(encoding="utf-8"))
    matrix = yaml.safe_load(args.matrix.read_text(encoding="utf-8"))
    only = set(args.only.split(",")) if args.only else None
    runs = plan_runs(base_cfg, matrix, only)

    log_dir = root / matrix["output_root"]
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%m-%d %H:%M:%S",
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / "run_experiments.log", encoding="utf-8")])
    log.info("%d çalıştırma planlandı", len(runs))
    if args.dry_run:
        for name, seed, cfg in runs:
            log.info("  %-12s seed %d -> %s", name, seed, cfg["output_dir"])
        return

    failed = []
    for i, (name, seed, cfg) in enumerate(runs, 1):
        log.info("=== [%d/%d] %s, seed %d ===", i, len(runs), name, seed)
        try:
            run_one(cfg, root)
        except Exception:  # bir çalıştırmanın hatası diğerlerini durdurmasın; tam iz loga yazılır
            log.error("BAŞARISIZ: %s seed %d\n%s", name, seed, traceback.format_exc())
            failed.append(f"{name}/seed{seed}")
    if failed:
        log.error("%d çalıştırma başarısız: %s", len(failed), ", ".join(failed))
        raise SystemExit(1)
    log.info("Tüm çalıştırmalar tamamlandı")


if __name__ == "__main__":
    main()

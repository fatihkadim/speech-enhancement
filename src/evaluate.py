"""Test bölmelerinde SI-SNR, PESQ ve STOI hesaplar; gürültülü girdinin (E0) metriklerini de raporlar.

Kullanım:
    uv run python -m src.evaluate --checkpoint experiments/baseline/best.pt --split test
    uv run python -m src.evaluate --config configs/baseline.yaml --split test      # yalnızca E0

Çıktı: <out_dir>/metrics_per_sample.csv ve <out_dir>/summary.json.
Varsayılan out_dir: checkpoint klasörü/eval_<split>, E0 için experiments/E0/<split>.
"""
import argparse
import json
import logging
import math
from pathlib import Path

import pandas as pd
import torch
import yaml
from pystoi import stoi
from tqdm import tqdm

from src.data.dataset import NoisySpeechDataset
from src.losses import si_snr
from src.models.enhancer import load_model
from src.train import resolve_device

try:
    from pesq import pesq as _pesq
except ImportError:  # Windows'ta derlenemeyebilir; Colab'da kurulur
    _pesq = None

log = logging.getLogger(__name__)
EVAL_SPLITS = ("test", "test_unseen_noise")
METRICS = ("si_snr", "pesq", "stoi")


def compute_metrics(est: torch.Tensor, clean: torch.Tensor, sr: int) -> dict[str, float]:
    """(T,) tahmin ve referans için SI-SNR (dB), PESQ (wb) ve STOI. PESQ yoksa nan."""
    est_np, clean_np = est.numpy(), clean.numpy()
    return {
        "si_snr": float(si_snr(est[None], clean[None])[0]),
        "pesq": float(_pesq(sr, clean_np, est_np, "wb")) if _pesq is not None else math.nan,
        "stoi": float(stoi(clean_np, est_np, sr, extended=False)),
    }


def _stats(series: pd.Series) -> dict:
    s = series.dropna()
    return {"mean": float(s.mean()), "std": float(s.std(ddof=0)), "n": int(s.size)} if len(s) else None


def summarize(df: pd.DataFrame, split: str, checkpoint: Path | None) -> dict:
    """Ortalama ± std; SNR'a (E5) ve gürültü sınıfına (E6) göre ortalamalar."""
    cols = [c for c in df.columns if c.split("_", 1)[0] in ("noisy", "enhanced", "delta")]
    by = lambda key: {str(k): {c: (None if pd.isna(v) else float(v)) for c, v in g[cols].mean().items()}
                      for k, g in df.groupby(key)}
    return {
        "split": split,
        "checkpoint": None if checkpoint is None else checkpoint.as_posix(),
        "n": len(df),
        "pesq_available": _pesq is not None,
        "metrics": {c: _stats(df[c]) for c in cols},
        "by_snr": by("snr_db"),
        "by_noise_class": by("noise_class"),
    }


def evaluate(cfg: dict, root: Path, split: str, out_dir: Path, checkpoint: Path | None = None,
             overwrite: bool = False) -> dict:
    """Bir test bölmesini değerlendirir; checkpoint yoksa yalnızca gürültülü girdi (E0) ölçülür."""
    if split not in EVAL_SPLITS:
        raise ValueError(f"Değerlendirme yalnızca {EVAL_SPLITS} bölmelerinde yapılır, verilen: {split!r}")
    if (out_dir / "summary.json").exists() and not overwrite:
        raise FileExistsError(f"{out_dir / 'summary.json'} zaten var; yeniden hesaplamak için --overwrite kullanın")
    if _pesq is None:
        log.warning("pesq kurulu değil; PESQ sütunları boş kalacak (Colab'da: uv sync --extra pesq)")

    device = resolve_device(cfg)
    model = None
    if checkpoint is not None:
        model, _ = load_model(checkpoint, device)
    d = cfg["data"]
    sr = d["sr"]
    ds = NoisySpeechDataset(root / d["manifest"], split, cfg["eval"]["segment_seconds"], sr)

    rows = []
    for i in tqdm(range(len(ds)), desc=f"değerlendirme ({split})"):
        item = ds[i]
        clean, noisy = item["clean"], item["noisy"]
        row = {"id": item["id"], "snr_db": item["snr_db"], "noise_class": ds.rows.loc[i, "noise_class"]}
        noisy_m = compute_metrics(noisy, clean, sr)
        row |= {f"noisy_{k}": v for k, v in noisy_m.items()}
        if model is not None:
            with torch.no_grad():
                est = model(noisy[None].to(device))["est_wave"][0].float().cpu()
            est_m = compute_metrics(est, clean, sr)
            row |= {f"enhanced_{k}": v for k, v in est_m.items()}
            row |= {f"delta_{k}": est_m[k] - noisy_m[k] for k in METRICS}
        rows.append(row)

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "metrics_per_sample.csv", index=False)
    summary = summarize(df, split, checkpoint)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("%s: %d örnek -> %s", split, len(df), out_dir)
    for name, s in summary["metrics"].items():
        if s:
            log.info("  %-18s %8.3f ± %.3f", name, s["mean"], s["std"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="SesTemiz değerlendirmesi")
    parser.add_argument("--checkpoint", type=Path, help="yoksa yalnızca gürültülü girdi (E0) ölçülür")
    parser.add_argument("--config", type=Path, help="checkpoint verilmediğinde gerekli")
    parser.add_argument("--split", choices=EVAL_SPLITS, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    root = Path.cwd()
    if args.checkpoint is not None:
        cfg = torch.load(args.checkpoint, map_location="cpu")["config"]
        out_dir = args.out_dir or args.checkpoint.parent / f"eval_{args.split}"
    elif args.config is not None:
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        out_dir = args.out_dir or root / "experiments" / "E0" / args.split
    else:
        parser.error("--checkpoint veya --config verilmeli")
    evaluate(cfg, root, args.split, out_dir, args.checkpoint, args.overwrite)


if __name__ == "__main__":
    main()

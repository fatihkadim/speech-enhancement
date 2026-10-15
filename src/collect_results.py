"""Deney matrisinin sonuçlarını toplar: deney tablosu (E1–E4), SNR'a göre (E5) ve görülmemiş gürültü (E6).

Kullanım:
    uv run python -m src.collect_results --matrix configs/experiments.yaml --markdown docs/results.md

Çıktı: <output_root>/results/ altında results.md ve CSV tabloları.
- Tohum değişkenliği: her tohumun ortalaması alınır, tohumlar arasında ortalama ± std raporlanır.
- Baseline'a göre fark: aynı tohum ve aynı test kaydı eşleştirilerek fark alınır, tohumlar üzerinden ortalanır;
  %95 güven aralığı test kayıtları üzerinde bootstrap ile hesaplanır.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.evaluate import EVAL_SPLITS, METRICS

log = logging.getLogger(__name__)
BASELINE = "baseline"
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 0


def load_per_sample(matrix: dict, root: Path) -> pd.DataFrame:
    """Tüm deney/tohum/bölme metrics_per_sample.csv dosyalarını tek tabloda birleştirir."""
    frames = []
    for name in matrix["experiments"]:
        for seed in matrix["seeds"]:
            for split in EVAL_SPLITS:
                path = root / matrix["output_root"] / name / f"seed{seed}" / f"eval_{split}" / "metrics_per_sample.csv"
                if not path.exists():
                    raise FileNotFoundError(f"Eksik sonuç: {path}")
                frames.append(pd.read_csv(path).assign(experiment=name, seed=seed, split=split))
    return pd.concat(frames, ignore_index=True)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Ortalamanın %95 bootstrap güven aralığı."""
    idx = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def experiment_table(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Deney başına: tohumlar arası ortalama ± std ve baseline'a göre eşleştirilmiş fark (%95 GA)."""
    d = df[df["split"] == split]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    per_seed = d.groupby(["experiment", "seed"])[[f"enhanced_{m}" for m in METRICS]].mean()
    base = d[d["experiment"] == BASELINE].set_index(["seed", "id"])
    rows = []
    for name in d["experiment"].unique():
        row = {"experiment": name, "seeds": per_seed.loc[name].shape[0]}
        exp = d[d["experiment"] == name].set_index(["seed", "id"])
        for m in METRICS:
            seed_means = per_seed.loc[name, f"enhanced_{m}"]
            row[f"{m}_mean"], row[f"{m}_std"] = seed_means.mean(), seed_means.std(ddof=1)
            if name != BASELINE:
                diff = (exp[f"enhanced_{m}"] - base[f"enhanced_{m}"]).groupby(level="id").mean().to_numpy()
                row[f"{m}_vs_base"] = diff.mean()
                row[f"{m}_vs_base_lo"], row[f"{m}_vs_base_hi"] = bootstrap_ci(diff, rng)
        rows.append(row)
    noisy = d[(d["experiment"] == BASELINE) & (d["seed"] == d["seed"].min())]
    rows.insert(0, {"experiment": "E0 (gürültülü girdi)", "seeds": 0,
                    **{f"{m}_mean": noisy[f"noisy_{m}"].mean() for m in METRICS}})
    return pd.DataFrame(rows)


def by_snr_table(df: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """E5: her deney için SNR değerine göre SI-SNR iyileşmesi ve PESQ (tohumlar üzerinden ortalama)."""
    d = df[df["split"] == split]
    out = d.groupby(["experiment", "snr_db"])[["delta_si_snr", "enhanced_pesq", "delta_stoi"]].mean().reset_index()
    noisy = d[d["experiment"] == BASELINE].groupby("snr_db")[["noisy_si_snr", "noisy_pesq", "noisy_stoi"]].mean()
    return out, noisy.reset_index()


def unseen_table(df: pd.DataFrame) -> pd.DataFrame:
    """E6: test ve test_unseen_noise'da SI-SNR iyileşmesi (tohumlar arası ortalama ± std)."""
    per_seed = df.groupby(["experiment", "split", "seed"])["delta_si_snr"].mean()
    agg = per_seed.groupby(["experiment", "split"]).agg(["mean", "std"]).unstack("split")
    agg.columns = [f"{split}_{stat}" for stat, split in agg.columns]
    return agg.reset_index()


def _pm(mean: float, std: float, digits: int) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}" if pd.notna(std) else f"{mean:.{digits}f}"


def _vs(row: pd.Series, m: str, digits: int) -> str:
    if pd.isna(row.get(f"{m}_vs_base")):
        return "–"
    lo, hi = row[f"{m}_vs_base_lo"], row[f"{m}_vs_base_hi"]
    mark = "" if lo <= 0 <= hi else " *"
    return f"{row[f'{m}_vs_base']:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]{mark}"


def to_markdown(tables: dict[str, pd.DataFrame], snr: tuple[pd.DataFrame, pd.DataFrame], unseen: pd.DataFrame) -> str:
    digits = {"si_snr": 2, "pesq": 3, "stoi": 3}
    lines = ["# Deney sonuçları", "",
             "Ortalama ± std: 3 tohumun ortalamalarının tohumlar arası değişimi. "
             "Baseline'a fark: aynı tohum ve test kaydı eşleştirilerek, [%95 güven aralığı]; "
             "`*` aralığın 0'ı içermediğini (farkın tesadüf olmadığını) gösterir.", ""]
    for split, t in tables.items():
        lines += [f"## {split}", "", "| Deney | SI-SNR (dB) | PESQ | STOI | Δ SI-SNR vs baseline | Δ PESQ vs baseline | Δ STOI vs baseline |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in t.iterrows():
            cells = [_pm(r[f"{m}_mean"], r.get(f"{m}_std"), digits[m]) for m in METRICS]
            cells += [_vs(r, m, digits[m]) for m in METRICS]
            lines.append(f"| {r['experiment']} | " + " | ".join(cells) + " |")
        lines.append("")
    by_snr, noisy = snr
    lines += ["## E5: SNR'a göre SI-SNR iyileşmesi (test, dB)", ""]
    pivot = by_snr.pivot(index="experiment", columns="snr_db", values="delta_si_snr")
    lines += ["| Deney | " + " | ".join(f"{c:+.0f} dB" for c in pivot.columns) + " |",
              "|---|" + "---|" * len(pivot.columns)]
    lines += [f"| girdi SI-SNR (E0) | " + " | ".join(f"{v:.2f}" for v in noisy["noisy_si_snr"]) + " |"]
    for name, r in pivot.iterrows():
        lines.append(f"| {name} | " + " | ".join(f"{v:+.2f}" for v in r) + " |")
    lines += ["", "## E6: görülmüş / görülmemiş gürültüde SI-SNR iyileşmesi (dB)", "",
              "| Deney | test | test_unseen_noise |", "|---|---|---|"]
    for _, r in unseen.iterrows():
        lines.append(f"| {r['experiment']} | {_pm(r['test_mean'], r['test_std'], 2)} | "
                     f"{_pm(r['test_unseen_noise_mean'], r['test_unseen_noise_std'], 2)} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Deney sonuçlarını topla")
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, help="results.md'nin bir kopyası da buraya yazılır (ör. docs/results.md)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    root = Path.cwd()
    matrix = yaml.safe_load(args.matrix.read_text(encoding="utf-8"))
    df = load_per_sample(matrix, root)
    tables = {split: experiment_table(df, split) for split in EVAL_SPLITS}
    snr = by_snr_table(df)
    unseen = unseen_table(df)

    out = root / matrix["output_root"] / "results"
    out.mkdir(parents=True, exist_ok=True)
    for split, t in tables.items():
        t.to_csv(out / f"experiments_{split}.csv", index=False)
    snr[0].to_csv(out / "by_snr_test.csv", index=False)
    unseen.to_csv(out / "unseen_noise.csv", index=False)
    markdown = to_markdown(tables, snr, unseen)
    (out / "results.md").write_text(markdown, encoding="utf-8")
    if args.markdown:
        args.markdown.write_text(markdown, encoding="utf-8")
    log.info("Sonuçlar yazıldı -> %s", out)


if __name__ == "__main__":
    main()

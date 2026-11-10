"""E7: gerçek kayıtları config'teki modellerle temizler.

Kullanım:
    uv run python -m src.real_recordings --config configs/e7_real.yaml

Her kayıt için `output_dir` altına `<ad>_gurultulu.wav`, `<ad>_<model>.wav` ve dört panelli
`<ad>.png` (gürültülü / ilk modelin maskesi / iki modelin çıktısı) yazılır. Temiz referans olmadığından
SI-SNR/PESQ/STOI hesaplanamaz; `ozet.csv` referanssız bir enerji özeti içerir: gürültü tabanı
(en sessiz çerçeveler) ve konuşma seviyesi (en yüksek çerçeveler) model çıktısında kaç dB düştü.
İdeal sonuç: taban çok düşer, konuşma seviyesi neredeyse hiç düşmez.
Var olan, boş olmayan bir çıktı klasörünün üzerine yazılmaz.
"""
import argparse
import csv
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.models.enhancer import SpectralEnhancer, load_model  # noqa: E402
from src.train import resolve_device  # noqa: E402
from src.utils.audio import load_audio  # noqa: E402
from src.utils.plotting import plot_spectrogram, spectrogram_db  # noqa: E402

log = logging.getLogger(__name__)
EPS = 1e-8


def find_recordings(input_dir: Path, extensions: list[str]) -> list[Path]:
    """Klasördeki (alt klasörler hariç) desteklenen ses dosyalarını ada göre sıralı döndürür."""
    exts = {e.lower() for e in extensions}
    files = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)
    if not files:
        raise FileNotFoundError(f"{input_dir} altında {sorted(exts)} uzantılı kayıt yok")
    return files


def levels_db(wave: np.ndarray, sr: int, frame_seconds: float, noise_pct: float,
              speech_pct: float) -> tuple[float, float]:
    """Çerçeve enerjilerinden (gürültü tabanı, konuşma seviyesi) dB değerlerini döndürür."""
    frame = int(frame_seconds * sr)
    n = len(wave) // frame
    if n == 0:
        raise ValueError(f"Kayıt bir çerçeveden ({frame_seconds} sn) kısa")
    energy = 10 * np.log10((wave[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-10)
    return float(np.percentile(energy, noise_pct)), float(np.percentile(energy, speech_pct))


def run_model(model: SpectralEnhancer, wave: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(T,) dalga -> (temizlenmiş (T,), etkin maske (F, T_frames) [0, 1])."""
    device = next(model.parameters()).device
    x = wave[None].to(device)
    with torch.no_grad():
        out = model(x)
        mask = (out["est_mag"] / (model.magnitude(x) + EPS)).clamp(0, 1)
    return out["est_wave"][0].float().cpu(), mask[0].float().cpu()


def save_figure(noisy: torch.Tensor, outputs: dict[str, tuple[torch.Tensor, torch.Tensor]], sr: int,
                n_fft: int, hop: int, path: Path, title: str) -> None:
    """Gürültülü / ilk modelin maskesi / her modelin çıktısı panellerini aynı dB ölçeğiyle kaydeder."""
    vmax = float(spectrogram_db(noisy, n_fft, hop).max())
    names = list(outputs)
    fig, axes = plt.subplots(2 + len(names), 1, figsize=(9, 2.5 * (2 + len(names))), sharex=True,
                             constrained_layout=True)
    plot_spectrogram(axes[0], noisy, sr, n_fft, hop, "gürültülü (gerçek kayıt)", vmax)
    im = axes[1].imshow(outputs[names[0]][1].numpy(), origin="lower", aspect="auto", cmap="viridis",
                        vmin=0, vmax=1, extent=(0, noisy.numel() / sr, 0, sr / 2 / 1000))
    axes[1].set(title=f"{names[0]} maskesi (sarı = koru, mor = sil)", ylabel="frekans (kHz)")
    fig.colorbar(im, ax=axes[1], fraction=0.02)
    for ax, name in zip(axes[2:], names):
        plot_spectrogram(ax, outputs[name][0], sr, n_fft, hop, f"temizlenmiş: {name}", vmax)
    fig.suptitle(title)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def _write_wav(path: Path, wave: torch.Tensor, sr: int) -> None:
    """float32 wav yazar; tepe 1'i aşarsa kliplememek için ölçekler."""
    peak = float(wave.abs().max())
    sf.write(path, (wave / max(1.0, peak)).numpy(), sr, subtype="FLOAT")


def process(cfg: dict, root: Path) -> list[dict]:
    """Tüm kayıtları temizler, çıktıları ve `ozet.csv`'yi yazar; özet satırlarını döndürür."""
    out_dir = root / cfg["output_dir"]
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} boş değil; üzerine yazılmaz")
    files = find_recordings(root / cfg["input_dir"], cfg["extensions"])
    device = resolve_device(cfg)
    models = {name: load_model(root / ckpt, device) for name, ckpt in cfg["models"].items()}
    model_cfgs = [c for _, c in models.values()]
    sr, n_fft, hop = model_cfgs[0]["data"]["sr"], model_cfgs[0]["stft"]["n_fft"], model_cfgs[0]["stft"]["hop"]
    if any((c["data"]["sr"], c["stft"]["n_fft"], c["stft"]["hop"]) != (sr, n_fft, hop) for c in model_cfgs):
        raise ValueError("Modellerin örnekleme hızı / STFT ayarları aynı olmalı")
    lv = cfg["levels"]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                                         encoding="utf-8")
    rows = []
    for path in tqdm(files, desc="E7"):
        noisy = load_audio(path, sr)
        outputs = {name: run_model(model, noisy) for name, (model, _) in models.items()}
        _write_wav(out_dir / f"{path.stem}_gurultulu.wav", noisy, sr)
        noise0, speech0 = levels_db(noisy.numpy(), sr, lv["frame_seconds"], lv["noise_percentile"],
                                    lv["speech_percentile"])
        row = {"dosya": path.name, "sure_s": round(noisy.numel() / sr, 2),
               "taban_db": round(noise0, 1), "konusma_db": round(speech0, 1)}
        for name, (est, _) in outputs.items():
            _write_wav(out_dir / f"{path.stem}_{name}.wav", est, sr)
            noise, speech = levels_db(est.numpy(), sr, lv["frame_seconds"], lv["noise_percentile"],
                                      lv["speech_percentile"])
            row[f"{name}_taban_dususu_db"] = round(noise0 - noise, 1)
            row[f"{name}_konusma_dususu_db"] = round(speech0 - speech, 1)
        rows.append(row)
        save_figure(noisy, outputs, sr, n_fft, hop, out_dir / f"{path.stem}.png", path.stem)

    with open(out_dir / "ozet.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    log.info("%d kayıt -> %s", len(rows), out_dir)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="E7: gerçek kayıtları temizle")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = Path.cwd()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for row in process(cfg, root):
        log.info("%s", row)


if __name__ == "__main__":
    main()

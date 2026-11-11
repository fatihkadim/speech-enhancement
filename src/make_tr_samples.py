"""Türkçe denemeler: temiz Türkçe kayıtları veri setimizin gürültüleriyle karıştırıp modelleri değerlendirir.

Gürültüler `samples/` altındaki örneklerden alınır (gürültülü − temiz = ESC-50 gürültüsü). Modeller yalnızca
İngilizce (LibriSpeech) konuşmayla eğitildiği için bu deney dil genellemesini ölçer.

Kullanım:
    uv run python -m src.make_tr_samples --speech-dir data/raw/turkce --out samples/tr
`--speech-dir` içindeki her ses dosyası (wav/mp3/flac/ogg/m4a) bir örnek olur. İsteğe bağlı `metinler.txt`
dosyasında `dosya_adı|metin` satırlarıyla yazıya dökümler verilebilir.
"""
import argparse
import json
import logging
import re
from pathlib import Path

import soundfile as sf
import torch

from src.evaluate import compute_metrics
from src.infer import enhance
from src.models.enhancer import load_model
from src.utils.audio import load_audio, mix_at_snr

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
SR = 16000
SNRS = [0, 5, -5, 10, 5, 0, 15, -5]
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
NOISE_RE = re.compile(r"snr[+-]\d+_(.+)_noisy\.wav")


def noise_pool(samples_dir: Path) -> list[tuple[str, torch.Tensor]]:
    """samples/ altındaki her gürültülü/temiz çiftinden saf gürültüyü çıkarır (sınıf başına bir tane)."""
    pool = {}
    for noisy in sorted(samples_dir.glob("snr*_noisy.wav")):
        m = NOISE_RE.fullmatch(noisy.name)
        clean = noisy.with_name(noisy.name.replace("_noisy", "_clean"))
        if m and clean.exists() and m.group(1) not in pool:
            pool[m.group(1)] = load_audio(noisy, SR) - load_audio(clean, SR)
    if not pool:
        raise FileNotFoundError(f"{samples_dir} altında gürültü çıkarılacak örnek yok")
    return list(pool.items())


def fit(noise: torch.Tensor, n: int) -> torch.Tensor:
    """Gürültüyü n örneğe tamamlar: döngüyle tekrarlar, eklerde 20 ms çapraz geçiş yapar."""
    out, fade = noise.clone(), int(0.02 * SR)
    ramp = torch.linspace(0, 1, fade)
    while len(out) < n:
        nxt = noise.clone()
        nxt[:fade] = out[-fade:] * (1 - ramp) + nxt[:fade] * ramp
        out = torch.cat([out[:-fade], nxt])
    return out[:n]


def read_texts(speech_dir: Path) -> dict[str, str]:
    f = speech_dir / "metinler.txt"
    if not f.exists():
        return {}
    pairs = (line.split("|", 1) for line in f.read_text(encoding="utf-8").splitlines() if "|" in line)
    return {Path(k.strip()).stem: v.strip() for k, v in pairs}


def main() -> None:
    parser = argparse.ArgumentParser(description="Türkçe örnekler ve denemeler")
    parser.add_argument("--speech-dir", type=Path, default=ROOT / "data/raw/turkce")
    parser.add_argument("--samples-dir", type=Path, default=ROOT / "samples")
    parser.add_argument("--out", type=Path, default=ROOT / "samples/tr")
    parser.add_argument("--models", nargs="+", default=["e3_depth5", "e1_si_snr"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-seconds", type=float, default=6.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from app.server import local_checkpoint  # models/ veya experiments/ altındaki checkpoint
    models = {}
    for name in args.models:
        ckpt = local_checkpoint(name)
        if ckpt is None:
            raise FileNotFoundError(f"{name} checkpoint'i yok; önce arayüzden bir kez çalıştırın ya da indirin")
        models[name] = load_model(ckpt, args.device)[0]

    files = sorted(p for p in args.speech_dir.iterdir() if p.suffix.lower() in AUDIO_EXT)
    if not files:
        raise FileNotFoundError(f"{args.speech_dir} içinde ses dosyası yok")
    noises, texts = noise_pool(args.samples_dir), read_texts(args.speech_dir)
    args.out.mkdir(parents=True, exist_ok=True)

    items = []
    for i, path in enumerate(files):
        clean = load_audio(path, SR)[: int(args.max_seconds * SR)]
        clean = clean / clean.abs().max() * 0.5
        noise_name, noise = noises[i % len(noises)]
        snr = SNRS[i % len(SNRS)]
        noisy = mix_at_snr(clean, fit(noise, len(clean)), snr)
        peak = max(1.0, float(noisy.abs().max()) / 0.95)
        clean, noisy = clean / peak, noisy / peak
        sid = f"snr{snr:+d}_{noise_name}_{re.sub(r'[^0-9A-Za-z]+', '_', path.stem).strip('_').lower()}"
        sf.write(args.out / f"{sid}_clean.wav", clean.numpy(), SR, subtype="PCM_16")
        sf.write(args.out / f"{sid}_noisy.wav", noisy.numpy(), SR, subtype="PCM_16")

        # değerlendirme kaydedilen (16-bit) dosyalar üzerinden: arayüzdekiyle aynı girdi
        clean, noisy = load_audio(args.out / f"{sid}_clean.wav", SR), load_audio(args.out / f"{sid}_noisy.wav", SR)
        metrics = {"noisy": compute_metrics(noisy, clean, SR)}
        for name, model in models.items():
            metrics[name] = compute_metrics(enhance(model, noisy), clean, SR)
        items.append({"id": sid, "snr": snr, "noise": noise_name, "text": texts.get(path.stem, ""),
                      "seconds": round(len(clean) / SR, 2), "metrics": metrics})
        log.info("%s: SI-SNR %.2f -> %s", sid, metrics["noisy"]["si_snr"],
                 ", ".join(f"{n} {metrics[n]['si_snr']:.2f}" for n in models))

    nan_to_none = lambda o: {k: nan_to_none(v) for k, v in o.items()} if isinstance(o, dict) else (
        [nan_to_none(v) for v in o] if isinstance(o, list) else (None if isinstance(o, float) and o != o else o))
    summary = {k: {m: sum(it["metrics"][k][m] for it in items) / len(items) for m in ("si_snr", "stoi")}
               for k in ["noisy", *models]}
    manifest = {"models": list(models), "items": items, "summary": summary}
    (args.out / "denemeler.json").write_text(json.dumps(nan_to_none(manifest), ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("yazıldı: %s (%d örnek)", args.out / "denemeler.json", len(items))


if __name__ == "__main__":
    main()

"""Tek bir .wav dosyasını temizler.

Kullanım:
    uv run python -m src.infer --checkpoint experiments/baseline/best.pt --input kayit.wav --output temiz.wav
"""
import argparse
import logging
from pathlib import Path

import soundfile as sf
import torch

from src.models.enhancer import SpectralEnhancer, load_model
from src.utils.audio import load_audio

log = logging.getLogger(__name__)


def enhance(model: SpectralEnhancer, wave: torch.Tensor) -> torch.Tensor:
    """(T,) gürültülü dalga -> (T,) temizlenmiş dalga."""
    device = next(model.parameters()).device
    with torch.no_grad():
        return model(wave[None].to(device))["est_wave"][0].float().cpu()


def enhance_file(checkpoint: Path, input_path: Path, output_path: Path, device: str = "cpu") -> Path:
    """Dosyayı modelin örnekleme hızında okuyup temizler ve float32 wav olarak yazar."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Çıkış dosyası girişle aynı olamaz")
    model, cfg = load_model(checkpoint, device)
    sr = cfg["data"]["sr"]
    est = enhance(model, load_audio(input_path, sr))
    peak = est.abs().max()
    if peak > 1:
        log.warning("Çıkış 1'i aşıyor (tepe %.2f); kliplememek için ölçeklendi", peak)
        est = est / peak
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, est.numpy(), sr, subtype="FLOAT")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Tek dosyayı temizle")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log.info("yazıldı: %s", enhance_file(args.checkpoint, args.input, args.output, args.device))


if __name__ == "__main__":
    main()

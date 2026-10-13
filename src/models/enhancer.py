"""Dalga formu ↔ spektrogram dönüşümünü U-Net'le birleştiren uçtan uca model.

noisy → STFT → |X| (Nyquist bandı atılır) → log1p → U-Net → tahmini genlik
      → Nyquist bandına 0 eklenir → gürültülü fazla birleştirilir → iSTFT → est_wave
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.unet import UNet
from src.utils.audio import istft, stft


class SpectralEnhancer(nn.Module):
    """`net` (B,1,F,T) log1p genlik alıp `out_mode`'a göre maske veya log1p temiz genlik döndürür."""

    def __init__(self, net: nn.Module, n_fft: int, hop: int, out_mode: str = "mask") -> None:
        super().__init__()
        self.net, self.n_fft, self.hop, self.out_mode = net, n_fft, hop, out_mode

    def magnitude(self, wave: torch.Tensor) -> torch.Tensor:
        """(B, T) -> (B, F-1, T_frames) genlik; Nyquist bandı atılmış (kayıp hesabında hedef olarak da kullanılır)."""
        return stft(wave, self.n_fft, self.hop)[:, :-1].abs()

    def forward(self, noisy: torch.Tensor) -> dict[str, torch.Tensor]:
        """(B, T) gürültülü dalga -> {'est_mag': (B, F-1, T_frames), 'est_wave': (B, T)}."""
        spec = stft(noisy, self.n_fft, self.hop)                # (B, F, T_frames) karmaşık
        mag = spec[:, :-1].abs()                                 # Nyquist bandı atıldı
        out = self.net(torch.log1p(mag).unsqueeze(1)).squeeze(1)
        if self.out_mode == "mask":
            est_mag = out * mag
        else:
            est_mag = torch.expm1(out.clamp(min=0))
        phase = torch.polar(torch.ones_like(spec.real), spec.angle())  # e^{j∠X}: gürültülü faz
        est_spec = F.pad(est_mag, (0, 0, 0, 1)) * phase                # Nyquist bandına 0 eklendi
        est_wave = istft(est_spec, self.n_fft, self.hop, length=noisy.shape[-1])
        return {"est_mag": est_mag, "est_wave": est_wave}


def load_model(checkpoint: Path, device: torch.device | str = "cpu") -> tuple[SpectralEnhancer, dict]:
    """train.py'nin kaydettiği checkpoint'ten modeli (eval modunda) ve config'i yükler."""
    state = torch.load(checkpoint, map_location=device)
    model = build_model(state["config"]).to(device)
    model.load_state_dict(state["model"])
    return model.eval(), state["config"]


def build_model(cfg: dict) -> SpectralEnhancer:
    """Config'den modeli kurar."""
    m = cfg["model"]
    if m["name"] != "unet":
        raise ValueError(f"Bilinmeyen model: {m['name']}")
    net = UNet(in_ch=1, base_ch=m["base_ch"], depth=m["depth"], skip=m["skip"], out_mode=m["out_mode"])
    return SpectralEnhancer(net, cfg["stft"]["n_fft"], cfg["stft"]["hop"], out_mode=m["out_mode"])

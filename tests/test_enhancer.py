"""src/models/enhancer.py testleri: dönüşüm zinciri, gradyan akışı ve tek batch'e overfit (M3)."""
import pytest
import torch
import torch.nn as nn

from src.losses import l1_loss, si_snr_loss
from src.models.enhancer import SpectralEnhancer, build_model
from src.models.unet import UNet
from src.utils.audio import istft, stft

SR, N_FFT, HOP = 16000, 512, 128


class ConstantNet(nn.Module):
    """Her hücreye sabit değer döndüren sahte ağ."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, self.value)


def _toy_batch(batch: int = 2, seconds: float = 0.5, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Harmonik "konuşma" + beyaz gürültü (0 dB civarı)."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(int(SR * seconds)) / SR
    f0 = 120 + 40 * torch.rand(batch, 1, generator=g)
    clean = sum(torch.sin(2 * torch.pi * h * f0 * t) / h for h in range(1, 15)) * 0.3
    noisy = clean + 0.3 * torch.randn(batch, t.numel(), generator=g)
    return clean, noisy


def test_identity_mask_reconstructs_input() -> None:
    """Maske = 1 ise çıkış, Nyquist bandı sıfırlanmış girişe eşit olmalı.

    Hann penceresinin sızıntısı yüzünden saf sinüslerde bile Nyquist bandı tam sıfır değildir (~-49 dB);
    bu bandı atmak şartnamenin bilinçli tercihi ve orijinale göre yalnızca çok küçük bir fark bırakır.
    """
    t = torch.arange(SR // 2) / SR
    wave = (torch.sin(2 * torch.pi * 300 * t) + 0.5 * torch.sin(2 * torch.pi * 3000 * t)).unsqueeze(0)
    out = SpectralEnhancer(ConstantNet(1.0), N_FFT, HOP)(wave)
    spec = stft(wave, N_FFT, HOP)
    spec[:, -1] = 0
    expected = istft(spec, N_FFT, HOP, length=wave.shape[-1])
    assert out["est_wave"].shape == wave.shape
    assert (out["est_wave"] - expected).abs().max() < 1e-5   # zincir doğru
    assert (out["est_wave"] - wave).abs().max() < 1e-3       # Nyquist'i atmanın bedeli küçük


def test_zero_mask_gives_silence() -> None:
    _, noisy = _toy_batch()
    out = SpectralEnhancer(ConstantNet(0.0), N_FFT, HOP)(noisy)
    assert out["est_wave"].abs().max() < 1e-6


@pytest.mark.parametrize("out_mode", ["mask", "direct"])
def test_shapes_and_gradient_flow(out_mode: str) -> None:
    cfg = {"model": {"name": "unet", "base_ch": 4, "depth": 3, "skip": True, "out_mode": out_mode},
           "stft": {"n_fft": N_FFT, "hop": HOP}}
    model = build_model(cfg)
    clean, noisy = _toy_batch()
    out = model(noisy)
    assert out["est_wave"].shape == noisy.shape
    assert out["est_mag"].shape == model.magnitude(clean).shape == (2, N_FFT // 2, noisy.shape[-1] // HOP + 1)
    (l1_loss(out["est_mag"], model.magnitude(clean)) + si_snr_loss(out["est_wave"], clean)).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not missing, f"gradyan ulaşmayan parametreler: {missing}"


def test_overfit_single_batch() -> None:
    """Tek bir batch'i ezberleyebilmeli: kayıp başlangıcın %10'unun altına inmeli."""
    torch.manual_seed(0)
    model = SpectralEnhancer(UNet(base_ch=8, depth=3, out_mode="mask"), N_FFT, HOP)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    clean, noisy = _toy_batch()
    target = model.magnitude(clean)
    losses = []
    for _ in range(300):
        opt.zero_grad()
        loss = l1_loss(model(noisy)["est_mag"], target)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.1 * losses[0], f"ilk {losses[0]:.4f} -> son {losses[-1]:.4f}"

"""src/losses.py testleri."""
import math

import pytest
import torch

from src.losses import l1_loss, mse_loss, si_snr, si_snr_loss


def _randn(*shape: int, seed: int = 0) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def test_magnitude_losses_zero_on_equal_and_positive_otherwise() -> None:
    mag = _randn(2, 256, 50).abs()
    assert l1_loss(mag, mag) == 0 and mse_loss(mag, mag) == 0
    other = mag + 0.5
    assert l1_loss(other, mag) > 0 and mse_loss(other, mag) > 0


def test_l1_mse_known_value() -> None:
    est, ref = torch.full((4,), math.e - 1), torch.zeros(4)  # log1p farkı tam 1
    assert torch.isclose(l1_loss(est, ref), torch.tensor(1.0))
    assert torch.isclose(mse_loss(est, ref), torch.tensor(1.0))


@pytest.mark.parametrize("scale", [0.1, 3.0, 50.0])
def test_si_snr_scale_invariant(scale: float) -> None:
    clean, est = _randn(2, 16000), _randn(2, 16000, seed=1) * 0.3 + _randn(2, 16000)
    assert torch.allclose(si_snr_loss(est, clean), si_snr_loss(scale * est, clean), atol=1e-4)


def test_si_snr_offset_invariant() -> None:
    clean, est = _randn(1, 8000), _randn(1, 8000, seed=2)
    assert torch.allclose(si_snr(est, clean), si_snr(est + 0.7, clean), atol=1e-4)


def test_si_snr_known_value() -> None:
    """ŝ = s + e, e ⟂ s ve ‖e‖² = ‖s‖²/10 ise SI-SNR = 10 dB."""
    s = _randn(1, 16000)
    s = s - s.mean()
    e = _randn(1, 16000, seed=3)
    e = e - e.mean()
    e = e - (e * s).sum() / (s * s).sum() * s          # s'ye dik yap
    e = e * torch.sqrt((s * s).sum() / (e * e).sum() / 10)
    assert torch.allclose(si_snr(s + e, s), torch.tensor([10.0]), atol=1e-3)


def test_si_snr_perfect_estimate_is_high_and_loss_negative() -> None:
    s = _randn(2, 8000)
    assert (si_snr(s, s) > 60).all()
    assert si_snr_loss(s, s) < 0


def test_si_snr_silent_input_is_finite() -> None:
    zeros = torch.zeros(2, 4000)
    assert torch.isfinite(si_snr_loss(zeros, zeros))
    assert torch.isfinite(si_snr_loss(_randn(2, 4000), zeros))

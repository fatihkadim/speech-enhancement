"""Kayıp fonksiyonları: log1p genlik uzayında L1/MSE ve dalga formu üzerinde SI-SNR."""
import torch


def l1_loss(est_mag: torch.Tensor, clean_mag: torch.Tensor) -> torch.Tensor:
    """log1p genlik uzayında ortalama mutlak hata."""
    return (torch.log1p(est_mag) - torch.log1p(clean_mag)).abs().mean()


def mse_loss(est_mag: torch.Tensor, clean_mag: torch.Tensor) -> torch.Tensor:
    """log1p genlik uzayında ortalama karesel hata."""
    return (torch.log1p(est_mag) - torch.log1p(clean_mag)).pow(2).mean()


def si_snr(est_wave: torch.Tensor, clean_wave: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """(B, T) -> (B,) örnek başına SI-SNR (dB). Önce ikisinin de ortalaması çıkarılır."""
    est = est_wave - est_wave.mean(dim=-1, keepdim=True)
    ref = clean_wave - clean_wave.mean(dim=-1, keepdim=True)
    dot = (est * ref).sum(dim=-1, keepdim=True)
    s_target = dot / (ref.pow(2).sum(dim=-1, keepdim=True) + eps) * ref
    e_noise = est - s_target
    return 10 * torch.log10((s_target.pow(2).sum(dim=-1) + eps) / (e_noise.pow(2).sum(dim=-1) + eps))


def si_snr_loss(est_wave: torch.Tensor, clean_wave: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Negatif SI-SNR (dB), batch ortalaması."""
    return -si_snr(est_wave, clean_wave, eps).mean()


LOSSES = {"l1": l1_loss, "mse": mse_loss, "si_snr": si_snr_loss}
WAVE_LOSSES = {"si_snr"}  # dalga formu üzerinde çalışanlar; diğerleri genlik üzerinde

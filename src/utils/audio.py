"""Ses yardımcıları: dosya okuma, STFT/iSTFT ve hedef SNR'da karıştırma."""
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as AF


def load_audio(path: Path, sr: int = 16000) -> torch.Tensor:
    """Mono, `sr` Hz, float32 dalga formu (T,). Gerekirse yeniden örnekler."""
    data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)  # (T, C)
    wave = torch.from_numpy(data).mean(dim=1)  # kanalların ortalaması -> mono (T,)
    if file_sr != sr:
        wave = AF.resample(wave, orig_freq=file_sr, new_freq=sr)
    return wave.contiguous()


def _hann(n_fft: int, wave: torch.Tensor) -> torch.Tensor:
    """Girdiyle aynı cihaz ve veri tipinde Hann penceresi."""
    dtype = wave.real.dtype if wave.is_complex() else wave.dtype
    return torch.hann_window(n_fft, device=wave.device, dtype=dtype)


def stft(wave: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    """(B, T) -> karmaşık (B, F, T_frames). Hann penceresi, center=True."""
    return torch.stft(
        wave,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=_hann(n_fft, wave),
        center=True,
        return_complex=True,
    )


def istft(spec: torch.Tensor, n_fft: int, hop: int, length: int) -> torch.Tensor:
    """(B, F, T_frames) karmaşık -> (B, length)."""
    return torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=_hann(n_fft, spec),
        center=True,
        length=length,
    )


def mix_at_snr(clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    """clean + scale * noise; ölçek, hedef SNR'ı sağlayacak şekilde hesaplanır."""
    if clean.shape != noise.shape:
        raise ValueError(f"clean {tuple(clean.shape)} ve noise {tuple(noise.shape)} aynı boyutta olmalı")
    clean_power = clean.pow(2).mean()
    noise_power = noise.pow(2).mean()
    if noise_power == 0:
        raise ValueError("Gürültünün enerjisi sıfır; hedef SNR sağlanamaz")
    scale = torch.sqrt(clean_power / (noise_power * 10 ** (snr_db / 10)))
    return clean + scale * noise

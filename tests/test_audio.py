"""src/utils/audio.py testleri (M1 kabul kriterleri)."""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from src.utils.audio import istft, load_audio, mix_at_snr, stft

N_FFT = 512
HOP = 128
SR = 16000


def _randn(*shape: int, seed: int = 0) -> torch.Tensor:
    """Tekrar üretilebilir rastgele tensör."""
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def _snr_db(clean: torch.Tensor, noise: torch.Tensor) -> float:
    return float(10 * torch.log10(clean.pow(2).mean() / noise.pow(2).mean()))


# --- STFT / iSTFT ---------------------------------------------------------

def test_stft_shape() -> None:
    wave = _randn(2, 3 * SR)  # 3 sn'lik iki örnek
    spec = stft(wave, N_FFT, HOP)
    assert spec.is_complex()
    # center=True: kare sayısı = T // hop + 1
    assert spec.shape == (2, N_FFT // 2 + 1, 3 * SR // HOP + 1)


@pytest.mark.parametrize("length", [3 * SR, 3 * SR + 37])  # hop'un katı olan ve olmayan uzunluk
def test_stft_istft_roundtrip(length: int) -> None:
    wave = _randn(2, length, seed=1)
    rec = istft(stft(wave, N_FFT, HOP), N_FFT, HOP, length=length)
    assert rec.shape == wave.shape
    assert (rec - wave).abs().max() < 1e-4


# --- mix_at_snr -----------------------------------------------------------

@pytest.mark.parametrize("snr_db", [-5.0, 0.0, 5.0, 10.0, 15.0])
def test_mix_at_snr_hits_target(snr_db: float) -> None:
    clean = _randn(3 * SR, seed=2)
    noise = 0.3 * _randn(3 * SR, seed=3)
    mix = mix_at_snr(clean, noise, snr_db)
    measured = _snr_db(clean, mix - clean)  # karışımdaki gerçek gürültü = mix - clean
    assert abs(measured - snr_db) < 0.1


def test_mix_at_snr_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        mix_at_snr(_randn(100), _randn(90), 0.0)


def test_mix_at_snr_rejects_silent_noise() -> None:
    with pytest.raises(ValueError):
        mix_at_snr(_randn(100), torch.zeros(100), 0.0)


# --- load_audio -----------------------------------------------------------

def test_load_audio_mono_passthrough(tmp_path: Path) -> None:
    data = 0.5 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR).astype(np.float32)
    path = tmp_path / "mono.wav"
    sf.write(path, data, SR, subtype="FLOAT")
    wave = load_audio(path, sr=SR)
    assert wave.dtype == torch.float32 and wave.shape == (SR,)
    assert torch.allclose(wave, torch.from_numpy(data))


def test_load_audio_stereo_resampled(tmp_path: Path) -> None:
    file_sr = 44100
    t = np.arange(file_sr) / file_sr  # 1 sn
    left = 0.4 * np.sin(2 * np.pi * 440 * t)
    stereo = np.stack([left, left * 0.5], axis=1).astype(np.float32)
    path = tmp_path / "stereo.wav"
    sf.write(path, stereo, file_sr, subtype="PCM_16")
    wave = load_audio(path, sr=SR)
    assert wave.ndim == 1 and wave.dtype == torch.float32
    assert wave.shape[0] == SR  # 1 sn -> 16 000 örnek
    assert wave.abs().max() <= 1.0

"""Spektrogram çizimleri (insan kontrolü ve rapor görselleri için)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # ekransız ortamlarda (sunucu, test) çizim için
import matplotlib.pyplot as plt
import torch

from src.utils.audio import stft

DYNAMIC_RANGE_DB = 80  # renk ölçeği: en güçlü hücrenin 80 dB altı siyah
EPS = 1e-8             # log(0)'ı önlemek için


def spectrogram_db(wave: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    """(T,) dalga formu -> (F, T_frames) dB cinsinden genlik spektrogramı."""
    mag = stft(wave.unsqueeze(0), n_fft, hop)[0].abs()
    return 20 * torch.log10(mag + EPS)


def plot_spectrogram(ax: plt.Axes, wave: torch.Tensor, sr: int, n_fft: int, hop: int, title: str,
                     vmax: float | None = None) -> None:
    """Verilen eksene spektrogram çizer; y ekseni kHz, x ekseni saniye."""
    db = spectrogram_db(wave, n_fft, hop)
    vmax = float(db.max()) if vmax is None else vmax
    ax.imshow(db.numpy(), origin="lower", aspect="auto", cmap="magma",
              extent=(0, wave.numel() / sr, 0, sr / 2 / 1000), vmin=vmax - DYNAMIC_RANGE_DB, vmax=vmax)
    ax.set(title=title, xlabel="zaman (sn)", ylabel="frekans (kHz)")


def save_enhancement_figure(clean: torch.Tensor, noisy: torch.Tensor, enhanced: torch.Tensor, mask: torch.Tensor,
                            sr: int, n_fft: int, hop: int, path: Path, title: str) -> None:
    """Temiz / gürültülü / model maskesi / temizlenmiş: dört paneli aynı renk ölçeğiyle kaydeder.

    mask: (F, T_frames), [0, 1]; 1 = korunan, 0 = silinen hücre.
    """
    vmax = float(max(spectrogram_db(w, n_fft, hop).max() for w in (clean, noisy, enhanced)))
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True, constrained_layout=True)
    plot_spectrogram(axes[0], clean, sr, n_fft, hop, "temiz (hedef)", vmax)
    plot_spectrogram(axes[1], noisy, sr, n_fft, hop, "gürültülü (girdi)", vmax)
    im = axes[2].imshow(mask.numpy(), origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1,
                        extent=(0, noisy.numel() / sr, 0, sr / 2 / 1000))
    axes[2].set(title="modelin maskesi (sarı = koru, mor = sil)", ylabel="frekans (kHz)")
    fig.colorbar(im, ax=axes[2], fraction=0.02)
    plot_spectrogram(axes[3], enhanced, sr, n_fft, hop, "temizlenmiş (çıktı)", vmax)
    fig.suptitle(title)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def save_example_figure(clean: torch.Tensor, noisy: torch.Tensor, sr: int, n_fft: int, hop: int,
                        path: Path, title: str) -> None:
    """Temiz ve gürültülü spektrogramları aynı renk ölçeğiyle alt alta kaydeder."""
    vmax = float(torch.maximum(spectrogram_db(clean, n_fft, hop).max(), spectrogram_db(noisy, n_fft, hop).max()))
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True, constrained_layout=True)
    plot_spectrogram(axes[0], clean, sr, n_fft, hop, "temiz", vmax)
    plot_spectrogram(axes[1], noisy, sr, n_fft, hop, "gürültülü", vmax)
    fig.suptitle(title)
    fig.savefig(path, dpi=110)
    plt.close(fig)

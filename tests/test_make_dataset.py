"""src/data/make_dataset.py testleri: bölme, gürültü hazırlama, karışım ve sızıntı kontrolü."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from src.data.make_dataset import (
    UNSEEN_SPLIT,
    build_dataset,
    check_leakage,
    fit_noise_length,
    mix_pair,
    split_speakers,
    trim_silence,
)
from src.utils.audio import load_audio

SR = 16000
NOISE_SR = 44100
CLASSES = ["dog", "rain", "siren", "rooster"]
UNSEEN = ["siren"]
SPEAKERS = {f"{100 + i}": ("M" if i % 2 else "F") for i in range(8)}


def make_fake_corpus(root: Path) -> None:
    """Gerçek klasör yapısını taklit eden minik LibriSpeech + ESC-50."""
    rng = np.random.default_rng(0)
    libri = root / "LibriSpeech"
    lines = [";ID |SEX| SUBSET | MINUTES | NAME"]
    for spk, sex in SPEAKERS.items():
        lines.append(f"{spk} | {sex} | dev-clean | 1.0 | Kişi {spk}")
        for utt in range(3):
            path = libri / "dev-clean" / spk / "1" / f"{spk}-1-{utt:04d}.flac"
            path.parent.mkdir(parents=True, exist_ok=True)
            t = np.arange(int(SR * rng.uniform(1.0, 2.0))) / SR
            sf.write(path, (0.5 * np.sin(2 * np.pi * (150 + 10 * utt) * t)).astype(np.float32), SR)
    (libri / "SPEAKERS.TXT").write_text("\n".join(lines), encoding="utf-8")

    esc = root / "ESC-50-master"
    (esc / "audio").mkdir(parents=True)
    (esc / "meta").mkdir()
    meta = []
    for target, cls in enumerate(CLASSES):
        for fold in range(1, 6):
            name = f"{fold}-{target}-A-{target}.wav"
            clip = np.zeros(NOISE_SR, dtype=np.float32)  # 1 sn; ilk 0,4 sn sessiz
            clip[int(0.4 * NOISE_SR):] = 0.3 * rng.standard_normal(int(0.6 * NOISE_SR))
            sf.write(esc / "audio" / name, clip, NOISE_SR)
            meta.append({"filename": name, "fold": fold, "target": target, "category": cls})
    pd.DataFrame(meta).to_csv(esc / "meta" / "esc50.csv", index=False)


def make_cfg(processed: str = "data/processed") -> dict:
    return {
        "seed": 42,
        "stft": {"n_fft": 512, "hop": 128},
        "data": {
            "clean_dir": "LibriSpeech/dev-clean",
            "noise_dir": "ESC-50-master",
            "manifest": f"{processed}/manifest.csv",
            "sr": SR,
            "segment_seconds": 0.5,
            "snr_list": [-5, 0, 5, 10, 15],
            "split_ratios": {"train": 0.5, "val": 0.25, "test": 0.25},
            "noise_folds": {"train": [1, 2, 3], "val": [4], "test": [5]},
            "unseen_noise_classes": UNSEEN,
            "max_utterances_per_speaker": None,
            "noise_silence_db": -50,
            "samples_dir": "samples",
            "num_samples": 5,
        },
    }


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    make_fake_corpus(tmp_path)
    return tmp_path


# --- Birim testleri ---------------------------------------------------------

def test_split_speakers_disjoint_balanced_deterministic() -> None:
    ratios = {"train": 0.5, "val": 0.25, "test": 0.25}
    a = split_speakers(list(SPEAKERS), ratios, seed=42, genders=SPEAKERS)
    b = split_speakers(list(SPEAKERS), ratios, seed=42, genders=SPEAKERS)
    assert a == b
    assert set(a) == set(SPEAKERS)
    for split, n in {"train": 4, "val": 2, "test": 2}.items():
        members = [s for s, v in a.items() if v == split]
        assert len(members) == n
        if split != "train":  # küçük bölmeler cinsiyet dengeli
            assert sorted(SPEAKERS[s] for s in members) == ["F", "M"]


def test_split_speakers_rejects_too_few() -> None:
    with pytest.raises(ValueError):
        split_speakers(["a", "b"], {"train": 0.8, "val": 0.1, "test": 0.1}, seed=0)


def test_trim_silence_and_all_silent() -> None:
    wave = torch.cat([torch.zeros(100), torch.ones(50), torch.zeros(30)])
    assert trim_silence(wave, -50).numel() == 50
    with pytest.raises(ValueError):
        trim_silence(torch.zeros(10), -50)


@pytest.mark.parametrize("noise_len", [30, 1000])  # tekrar gereken ve kesit alınan durum
def test_fit_noise_length(noise_len: int) -> None:
    out = fit_noise_length(torch.arange(noise_len, dtype=torch.float32) + 1, 200, np.random.default_rng(0))
    assert out.numel() == 200


def test_mix_pair_prevents_clipping_and_keeps_snr() -> None:
    clean = 0.9 * torch.sin(torch.linspace(0, 200, 8000))
    noise = torch.randn(8000, generator=torch.Generator().manual_seed(0))
    clean_out, noisy = mix_pair(clean, noise, snr_db=-5.0)
    assert noisy.abs().max() <= 1.0 + 1e-6
    measured = 10 * torch.log10(clean_out.pow(2).mean() / (noisy - clean_out).pow(2).mean())
    assert abs(float(measured) + 5.0) < 0.1


def _row(split: str, spk: str, noise: str, cls: str) -> dict:
    return {"split": split, "speaker_id": spk, "noise_path": noise, "noise_class": cls}


def test_check_leakage_passes_on_clean_split() -> None:
    df = pd.DataFrame([_row("train", "1", "n1", "dog"), _row("val", "2", "n2", "dog"),
                       _row("test", "3", "n3", "rain"), _row(UNSEEN_SPLIT, "3", "n4", "siren")])
    check_leakage(df, UNSEEN)


@pytest.mark.parametrize("bad_row", [
    _row("test", "1", "n9", "rain"),        # train konuşmacısı test'te
    _row("val", "2", "n1", "dog"),          # aynı gürültü klibi train ve val'de
    _row("train", "1", "n8", "siren"),      # görülmemiş sınıf eğitimde
    _row(UNSEEN_SPLIT, "3", "n7", "dog"),   # görülmüş sınıf unseen bölmesinde
])
def test_check_leakage_detects(bad_row: dict) -> None:
    df = pd.DataFrame([_row("train", "1", "n1", "dog"), _row("val", "2", "n2", "dog"),
                       _row("test", "3", "n3", "rain"), bad_row])
    with pytest.raises(ValueError, match="sızıntı"):
        check_leakage(df, UNSEEN)


# --- Uçtan uca --------------------------------------------------------------

def test_build_dataset_end_to_end(corpus: Path) -> None:
    df = build_dataset(make_cfg(), corpus)
    manifest = corpus / "data/processed/manifest.csv"
    assert manifest.exists()
    assert list(df.columns) == ["id", "clean_path", "noisy_path", "noise_path", "noise_class", "snr_db", "speaker_id", "split"]
    assert df["id"].is_unique
    counts = df["split"].value_counts().to_dict()
    # 8 konuşmacı × 3 kayıt: 4/2/2 konuşmacı; test kayıtları unseen gürültüyle bir kez daha
    assert counts == {"train": 12, "val": 6, "test": 6, UNSEEN_SPLIT: 6}
    assert set(df.loc[df["split"] == UNSEEN_SPLIT, "noise_class"]) == set(UNSEEN)

    for _, row in df.iterrows():
        clean = load_audio(manifest.parent / row["clean_path"], SR)
        noisy = load_audio(manifest.parent / row["noisy_path"], SR)
        assert clean.shape == noisy.shape
        assert noisy.abs().max() <= 1.0
        measured = 10 * torch.log10(clean.pow(2).mean() / (noisy - clean).pow(2).mean())
        assert abs(float(measured) - row["snr_db"]) < 0.1


def test_build_dataset_is_reproducible(corpus: Path) -> None:
    a = build_dataset(make_cfg("out_a"), corpus)
    b = build_dataset(make_cfg("out_b"), corpus)
    pd.testing.assert_frame_equal(a, b)


def test_build_dataset_refuses_overwrite(corpus: Path) -> None:
    build_dataset(make_cfg(), corpus)
    with pytest.raises(FileExistsError):
        build_dataset(make_cfg(), corpus)
    build_dataset(make_cfg(), corpus, overwrite=True)

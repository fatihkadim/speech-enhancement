"""E7 gerçek kayıt betiği: sentetik kayıtlar ve eğitilmemiş küçük modellerle."""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from src.models.enhancer import build_model
from src.real_recordings import find_recordings, levels_db, process

SR = 16000


def _checkpoint(path: Path, out_mode: str) -> Path:
    """Eğitilmemiş küçük bir U-Net'i train.py biçiminde ({model, config}) kaydeder."""
    cfg = {"data": {"sr": SR}, "stft": {"n_fft": 512, "hop": 128},
           "model": {"name": "unet", "base_ch": 4, "depth": 2, "skip": True, "out_mode": out_mode}}
    torch.manual_seed(0)
    torch.save({"model": build_model(cfg).state_dict(), "config": cfg}, path)
    return path


@pytest.fixture
def setup(tmp_path: Path) -> dict:
    rec = tmp_path / "kayit"
    (rec / "orijinal").mkdir(parents=True)
    rng = np.random.default_rng(0)
    sf.write(rec / "a.wav", 0.1 * rng.standard_normal(SR), SR)                 # 1 sn
    sf.write(rec / "b.flac", 0.1 * rng.standard_normal(SR // 2 + 37), 48000)  # farklı hız, tek sayılı uzunluk
    sf.write(rec / "orijinal" / "c.wav", rng.standard_normal(SR), SR)        # alt klasör: atlanmalı
    (rec / "not.txt").write_text("x", encoding="utf-8")
    return {"output_dir": "cikti", "input_dir": "kayit", "extensions": [".wav", ".flac"],
            "models": {"m1": str(_checkpoint(tmp_path / "m1.pt", "mask").relative_to(tmp_path)),
                       "m2": str(_checkpoint(tmp_path / "m2.pt", "direct").relative_to(tmp_path))},
            "levels": {"frame_seconds": 0.02, "noise_percentile": 10, "speech_percentile": 95},
            "device": "cpu"}


def test_find_recordings_skips_subdirs_and_other_files(tmp_path: Path, setup: dict) -> None:
    files = find_recordings(tmp_path / "kayit", [".WAV", ".flac"])
    assert [p.name for p in files] == ["a.wav", "b.flac"]
    with pytest.raises(FileNotFoundError):
        find_recordings(tmp_path / "kayit", [".ogg"])


def test_levels_db_known_values() -> None:
    frame = int(0.02 * SR)
    wave = np.concatenate([np.full(frame * 9, 0.01), np.full(frame, 1.0)])  # 9 sessiz + 1 yüksek çerçeve
    noise, speech = levels_db(wave, SR, 0.02, 10, 100)
    assert noise == pytest.approx(-40, abs=1e-3) and speech == pytest.approx(0, abs=1e-3)
    with pytest.raises(ValueError):
        levels_db(np.zeros(frame - 1), SR, 0.02, 10, 95)


def test_process_writes_outputs_and_refuses_overwrite(tmp_path: Path, setup: dict) -> None:
    rows = process(setup, tmp_path)
    out = tmp_path / "cikti"
    assert [r["dosya"] for r in rows] == ["a.wav", "b.flac"]
    assert {"m1_taban_dususu_db", "m2_konusma_dususu_db"} <= set(rows[0])
    for stem, n in (("a", SR), ("b", SR // 6 + 12)):  # 48 kHz -> 16 kHz yeniden örnekleme
        for suffix in ("gurultulu", "m1", "m2"):
            wave, sr = sf.read(out / f"{stem}_{suffix}.wav")
            assert sr == SR and abs(len(wave) - n) <= 1 and np.abs(wave).max() <= 1 + 1e-6
        assert (out / f"{stem}.png").stat().st_size > 0
    assert (out / "ozet.csv").read_text(encoding="utf-8").count("\n") == 3
    assert (out / "config.yaml").exists()
    with pytest.raises(FileExistsError):
        process(setup, tmp_path)

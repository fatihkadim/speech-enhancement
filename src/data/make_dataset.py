"""Temiz konuşma + gürültü karışımlarını üretir, manifest.csv yazar ve sızıntı kontrolü yapar.

Kullanım:
    uv run python -m src.data.make_dataset --config configs/baseline.yaml [--overwrite]

Manifest'teki `clean_path` ve `noisy_path` manifest dosyasının klasörüne göre,
`noise_path` ise proje köküne göre göreli yazılır (POSIX biçiminde, işletim sisteminden bağımsız).
"""
from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml
from tqdm import tqdm

from src.utils.audio import load_audio, mix_at_snr
from src.utils.plotting import save_example_figure

log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")
UNSEEN_SPLIT = "test_unseen_noise"
MANIFEST_COLUMNS = ["id", "clean_path", "noisy_path", "noise_path", "noise_class", "snr_db", "speaker_id", "split"]


@dataclass(frozen=True)
class Utterance:
    path: Path
    speaker_id: str


# --- Kaynak verileri listeleme ---------------------------------------------

def list_librispeech(clean_dir: Path) -> list[Utterance]:
    """`konuşmacı/bölüm/*.flac` yapısındaki kayıtları sıralı listeler."""
    files = sorted(clean_dir.glob("*/*/*.flac"))
    if not files:
        raise FileNotFoundError(f"{clean_dir} altında .flac bulunamadı")
    return [Utterance(p, p.parent.parent.name) for p in files]


def read_speaker_genders(speakers_txt: Path) -> dict[str, str]:
    """LibriSpeech SPEAKERS.TXT -> {konuşmacı: 'M' | 'F'}. Dosya yoksa boş sözlük."""
    if not speakers_txt.exists():
        log.warning("%s bulunamadı; bölme cinsiyete göre dengelenmeyecek", speakers_txt)
        return {}
    genders = {}
    for line in speakers_txt.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(";") or "|" not in line:
            continue
        speaker, gender = (part.strip() for part in line.split("|")[:2])
        genders[speaker] = gender
    return genders


def list_esc50(noise_dir: Path) -> pd.DataFrame:
    """ESC-50 meta tablosu: `path`, `fold`, `category` sütunları."""
    meta = pd.read_csv(noise_dir / "meta" / "esc50.csv")
    meta["path"] = [noise_dir / "audio" / name for name in meta["filename"]]
    return meta[["path", "fold", "category"]]


# --- Bölme ------------------------------------------------------------------

def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def split_speakers(
    speakers: list[str], ratios: dict[str, float], seed: int, genders: dict[str, str] | None = None
) -> dict[str, str]:
    """Konuşmacıları train/val/test'e böler.

    Cinsiyet bilgisi varsa her cinsiyet ayrı karıştırılıp sırayla dizilir (M, F, M, F, ...);
    böylece küçük val/test bölmeleri de dengeli olur.
    """
    rng = np.random.default_rng(seed)
    genders = genders or {}
    groups: dict[str, list[str]] = {}
    for speaker in sorted(set(speakers)):
        groups.setdefault(genders.get(speaker, "?"), []).append(speaker)
    for key in sorted(groups):
        rng.shuffle(groups[key])
    queues = [groups[key] for key in sorted(groups)]
    order = [q[i] for i in range(max(map(len, queues))) for q in queues if i < len(q)]

    n = len(order)
    n_test = _round_half_up(n * ratios["test"])
    n_val = _round_half_up(n * ratios["val"])
    if min(n_test, n_val, n - n_test - n_val) < 1:
        raise ValueError(f"{n} konuşmacı {ratios} oranlarıyla bölünemiyor; her bölmeye en az 1 kişi düşmeli")
    assignment = {s: "test" for s in order[:n_test]}
    assignment |= {s: "val" for s in order[n_test:n_test + n_val]}
    assignment |= {s: "train" for s in order[n_test + n_val:]}
    return assignment


def limit_per_speaker(utterances: list[Utterance], max_per_speaker: int | None) -> list[Utterance]:
    """Her konuşmacıdan en fazla `max_per_speaker` kayıt tutar (sıralı ilk kayıtlar)."""
    if max_per_speaker is None:
        return utterances
    counts: dict[str, int] = {}
    kept = []
    for u in utterances:
        if counts.get(u.speaker_id, 0) < max_per_speaker:
            kept.append(u)
            counts[u.speaker_id] = counts.get(u.speaker_id, 0) + 1
    return kept


def noise_pools(meta: pd.DataFrame, folds: dict[str, list[int]], unseen_classes: list[str]) -> dict[str, pd.DataFrame]:
    """Her bölme için kullanılabilecek gürültü kliplerini döndürür.

    Görülmemiş sınıflar yalnızca `test_unseen_noise` havuzuna girer; diğer sınıflar fold'a göre ayrılır.
    """
    unknown = set(unseen_classes) - set(meta["category"])
    if unknown:
        raise ValueError(f"Gürültü veri setinde olmayan sınıflar: {sorted(unknown)}")
    seen = meta[~meta["category"].isin(unseen_classes)]
    pools = {split: seen[seen["fold"].isin(folds[split])].reset_index(drop=True) for split in SPLITS}
    pools[UNSEEN_SPLIT] = meta[meta["category"].isin(unseen_classes)].reset_index(drop=True)
    for split, pool in pools.items():
        if pool.empty:
            raise ValueError(f"'{split}' bölmesi için gürültü klibi kalmadı")
    return pools


# --- Sinyal işlemleri -------------------------------------------------------

def trim_silence(wave: torch.Tensor, silence_db: float) -> torch.Tensor:
    """Tepe değerin `silence_db` altında kalan baş ve son kısımları kırpar."""
    threshold = wave.abs().max() * 10 ** (silence_db / 20)
    active = torch.nonzero(wave.abs() > threshold).squeeze(1)
    if active.numel() == 0:
        raise ValueError("Klip tamamen sessiz")
    return wave[active[0]: active[-1] + 1]


@lru_cache(maxsize=256)
def _load_noise(path: Path, sr: int, silence_db: float) -> torch.Tensor:
    return trim_silence(load_audio(path, sr), silence_db)


def fit_noise_length(noise: torch.Tensor, length: int, rng: np.random.Generator) -> torch.Tensor:
    """Gürültüyü `length` örneğe getirir: kısaysa tekrarlar, uzunsa rastgele kesit alır."""
    if noise.numel() < length:
        noise = noise.repeat(math.ceil(length / noise.numel()))
    start = int(rng.integers(0, noise.numel() - length + 1))
    return noise[start:start + length]


def mix_pair(clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(temiz, gürültülü) çifti üretir. Karışım 1'i aşarsa ikisi de aynı katsayıyla küçültülür; SNR korunur."""
    noisy = mix_at_snr(clean, noise, snr_db)
    peak = noisy.abs().max()
    if peak > 1:
        clean, noisy = clean / peak, noisy / peak
    return clean, noisy


# --- Sızıntı kontrolü -------------------------------------------------------

def check_leakage(df: pd.DataFrame, unseen_classes: list[str]) -> None:
    """Konuşmacı, gürültü sınıfı ve gürültü klibi düzeyinde sızıntı varsa ValueError fırlatır."""
    group = df["split"].replace({UNSEEN_SPLIT: "test"})  # test ve test_unseen_noise aynı konuşmacıları paylaşır
    problems = []
    for column in ("speaker_id", "noise_path"):
        sets = {g: set(df.loc[group == g, column]) for g in SPLITS}
        for i, a in enumerate(SPLITS):
            for b in SPLITS[i + 1:]:
                shared = sets[a] & sets[b]
                if shared:
                    problems.append(f"{column}: {a} ve {b} ortak {len(shared)} değer içeriyor (ör. {sorted(shared)[0]})")
    leaked = set(df.loc[df["split"] != UNSEEN_SPLIT, "noise_class"]) & set(unseen_classes)
    if leaked:
        problems.append(f"görülmemiş sınıflar başka bölmelerde kullanılmış: {sorted(leaked)}")
    foreign = set(df.loc[df["split"] == UNSEEN_SPLIT, "noise_class"]) - set(unseen_classes)
    if foreign:
        problems.append(f"{UNSEEN_SPLIT} görülmüş sınıflar içeriyor: {sorted(foreign)}")
    if problems:
        raise ValueError("Veri sızıntısı:\n  " + "\n  ".join(problems))


# --- Ana akış ---------------------------------------------------------------

def build_dataset(cfg: dict, root: Path, overwrite: bool = False) -> pd.DataFrame:
    """Karışımları üretip diske yazar, manifest'i kaydeder ve döndürür."""
    d = cfg["data"]
    sr, silence_db = d["sr"], d["noise_silence_db"]
    clean_dir, noise_dir = root / d["clean_dir"], root / d["noise_dir"]
    manifest_path = root / d["manifest"]
    out_dir = manifest_path.parent
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"{manifest_path} zaten var; yeniden üretmek için --overwrite kullanın")

    utterances = list_librispeech(clean_dir)
    genders = read_speaker_genders(clean_dir.parent / "SPEAKERS.TXT")
    speaker_split = split_speakers([u.speaker_id for u in utterances], d["split_ratios"], cfg["seed"], genders)
    utterances = limit_per_speaker(utterances, d["max_utterances_per_speaker"])
    pools = noise_pools(list_esc50(noise_dir), d["noise_folds"], d["unseen_noise_classes"])

    jobs = []
    for u in utterances:
        split = speaker_split[u.speaker_id]
        jobs.append((u, split))
        if split == "test":  # aynı test cümlesi görülmemiş gürültüyle de karıştırılır (E6)
            jobs.append((u, UNSEEN_SPLIT))

    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for u, split in tqdm(jobs, desc="karışımlar"):
        pool = pools[split]
        noise_row = pool.iloc[int(rng.integers(len(pool)))]
        snr_db = float(rng.choice(d["snr_list"]))
        clean = load_audio(u.path, sr)
        noise = fit_noise_length(_load_noise(noise_row["path"], sr, silence_db), clean.numel(), rng)
        clean, noisy = mix_pair(clean, noise, snr_db)

        example_id = f"{split}-{u.path.stem}"
        rel_clean = Path(split) / "clean" / f"{example_id}.wav"
        rel_noisy = Path(split) / "noisy" / f"{example_id}.wav"
        for rel, wave in ((rel_clean, clean), (rel_noisy, noisy)):
            (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            sf.write(out_dir / rel, wave.numpy(), sr, subtype="PCM_16")
        rows.append({
            "id": example_id,
            "clean_path": rel_clean.as_posix(),
            "noisy_path": rel_noisy.as_posix(),
            "noise_path": Path(noise_row["path"]).relative_to(root).as_posix(),
            "noise_class": noise_row["category"],
            "snr_db": snr_db,
            "speaker_id": u.speaker_id,
            "split": split,
        })

    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    check_leakage(df, d["unseen_noise_classes"])
    df.to_csv(manifest_path, index=False)
    log.info("%d örnek yazıldı -> %s", len(df), manifest_path)
    return df


def write_samples(df: pd.DataFrame, cfg: dict, root: Path) -> list[Path]:
    """İnsan kontrolü için validation'dan her SNR'dan bir örneği ses + spektrogram olarak yazar."""
    d = cfg["data"]
    sr, segment = d["sr"], int(d["segment_seconds"] * d["sr"])
    out_dir = root / d["samples_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    val = df[df["split"] == "val"]
    picks = [val[val["snr_db"] == snr].head(1) for snr in d["snr_list"]]
    picks = pd.concat(picks).head(d["num_samples"])
    manifest_dir = (root / d["manifest"]).parent
    written = []
    for _, row in picks.iterrows():
        clean = load_audio(manifest_dir / row["clean_path"], sr)[:segment]
        noisy = load_audio(manifest_dir / row["noisy_path"], sr)[:segment]
        stem = f"snr{int(row['snr_db']):+d}_{row['noise_class']}"
        sf.write(out_dir / f"{stem}_clean.wav", clean.numpy(), sr, subtype="PCM_16")
        sf.write(out_dir / f"{stem}_noisy.wav", noisy.numpy(), sr, subtype="PCM_16")
        figure = out_dir / f"{stem}.png"
        save_example_figure(clean, noisy, sr, cfg["stft"]["n_fft"], cfg["stft"]["hop"], figure,
                            title=f"{row['id']} · {row['noise_class']} · SNR {row['snr_db']:+.0f} dB")
        written.append(figure)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", help="mevcut manifest'i yeniden üret")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path.cwd()
    df = build_dataset(cfg, root, overwrite=args.overwrite)
    log.info("Bölmeler: %s", df["split"].value_counts().to_dict())
    log.info("Sızıntı kontrolü geçti")
    figures = write_samples(df, cfg, root)
    log.info("%d örnek %s altına yazıldı", len(figures), cfg["data"]["samples_dir"])


if __name__ == "__main__":
    main()

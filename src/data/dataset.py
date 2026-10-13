"""manifest.csv'den (gürültülü, temiz) dalga formu çiftleri okuyan PyTorch veri seti."""
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.utils.audio import load_audio


class NoisySpeechDataset(Dataset):
    """Bir bölmenin örneklerini döndürür.

    `segment_seconds` verilirse her örnek bu uzunluğa getirilir: train'de rastgele kesit,
    diğer bölmelerde baştan sabit kesit; kısa kayıtlar sondan sıfırla doldurulur.
    `None` ise kayıt tam uzunluğuyla döner.
    """

    def __init__(self, manifest_path: Path, split: str, segment_seconds: float | None = None, sr: int = 16000) -> None:
        manifest_path = Path(manifest_path)
        df = pd.read_csv(manifest_path)
        self.rows = df[df["split"] == split].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"{manifest_path} içinde '{split}' bölmesine ait örnek yok")
        self.root = manifest_path.parent
        self.sr = sr
        self.segment = None if segment_seconds is None else int(round(segment_seconds * sr))
        self.random_crop = split == "train"

    def __len__(self) -> int:
        return len(self.rows)

    def _crop(self, clean: torch.Tensor, noisy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length = clean.numel()
        if length < self.segment:
            pad = (0, self.segment - length)
            return F.pad(clean, pad), F.pad(noisy, pad)
        start = int(torch.randint(0, length - self.segment + 1, (1,))) if self.random_crop else 0
        end = start + self.segment
        return clean[start:end], noisy[start:end]

    def __getitem__(self, idx: int) -> dict:
        row = self.rows.iloc[idx]
        clean = load_audio(self.root / row["clean_path"], self.sr)
        noisy = load_audio(self.root / row["noisy_path"], self.sr)
        if self.segment is not None:
            clean, noisy = self._crop(clean, noisy)
        return {"noisy": noisy, "clean": clean, "snr_db": float(row["snr_db"]), "id": str(row["id"])}

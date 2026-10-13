"""src/data/dataset.py testleri."""
from pathlib import Path

import pytest
import torch

from src.data.dataset import NoisySpeechDataset
from src.data.make_dataset import build_dataset
from tests.test_make_dataset import SR, make_cfg, make_fake_corpus


@pytest.fixture(scope="module")
def manifest(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("corpus")
    make_fake_corpus(root)
    build_dataset(make_cfg(), root)
    return root / "data/processed/manifest.csv"


def test_item_format(manifest: Path) -> None:
    ds = NoisySpeechDataset(manifest, "train", segment_seconds=0.5, sr=SR)
    item = ds[0]
    assert set(item) == {"noisy", "clean", "snr_db", "id"}
    assert item["noisy"].shape == item["clean"].shape == (SR // 2,)
    assert item["noisy"].dtype == torch.float32
    assert isinstance(item["snr_db"], float) and isinstance(item["id"], str)


def test_train_crop_is_random_and_aligned(manifest: Path) -> None:
    ds = NoisySpeechDataset(manifest, "train", segment_seconds=0.25, sr=SR)
    torch.manual_seed(0)
    a, b = ds[0], ds[0]
    assert not torch.equal(a["clean"], b["clean"])  # farklı kesitler
    full = NoisySpeechDataset(manifest, "train", sr=SR)[0]
    # temiz ve gürültülü aynı noktadan kesilmeli. Sahte konuşma periyodik bir sinüs olduğundan temiz kesit
    # birçok konumla eşleşir; bu yüzden ikisinin birlikte eşleştiği bir konum aranır (gürültü rastgele, tek konum).
    seg = a["clean"].numel()
    aligned = [s for s in range(full["clean"].numel() - seg + 1)
               if torch.equal(full["clean"][s:s + seg], a["clean"]) and torch.equal(full["noisy"][s:s + seg], a["noisy"])]
    assert len(aligned) == 1


def test_eval_crop_is_fixed(manifest: Path) -> None:
    ds = NoisySpeechDataset(manifest, "val", segment_seconds=0.25, sr=SR)
    full = NoisySpeechDataset(manifest, "val", sr=SR)[0]
    assert torch.equal(ds[0]["clean"], ds[0]["clean"])
    assert torch.equal(ds[0]["clean"], full["clean"][: SR // 4])


def test_short_utterance_is_zero_padded(manifest: Path) -> None:
    ds = NoisySpeechDataset(manifest, "test", segment_seconds=3.0, sr=SR)  # kayıtlar 1-2 sn
    full = NoisySpeechDataset(manifest, "test", sr=SR)[0]
    item = ds[0]
    n = full["clean"].numel()
    assert item["clean"].numel() == 3 * SR
    assert torch.equal(item["clean"][:n], full["clean"]) and item["clean"][n:].abs().sum() == 0


def test_unknown_split_raises(manifest: Path) -> None:
    with pytest.raises(ValueError):
        NoisySpeechDataset(manifest, "yok", sr=SR)

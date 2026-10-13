"""Modeli config'e göre eğitir.

Kullanım:
    uv run python -m src.train --config configs/baseline.yaml

`output_dir` altına config.yaml, run_info.json, log.csv, best.pt (en iyi doğrulama kaybı) ve last.pt yazılır.
Var olan, boş olmayan bir deney klasörünün üzerine yazılmaz.
"""
import argparse
import csv
import json
import logging
import math
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import NoisySpeechDataset
from src.losses import LOSSES, WAVE_LOSSES, si_snr
from src.models.enhancer import SpectralEnhancer, build_model
from src.utils.repro import run_info, seed_everything, seed_worker

log = logging.getLogger(__name__)
LOG_COLUMNS = ["epoch", "train_loss", "val_loss", "val_si_snr", "lr", "seconds"]


def resolve_device(cfg: dict) -> torch.device:
    name = cfg.get("device", "auto")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def compute_loss(name: str, model: SpectralEnhancer, out: dict, clean: torch.Tensor) -> torch.Tensor:
    """Kaybı doğru uzayda hesaplar: SI-SNR dalga formunda, L1/MSE genlikte (float32)."""
    if name in WAVE_LOSSES:
        return LOSSES[name](out["est_wave"].float(), clean)
    return LOSSES[name](out["est_mag"].float(), model.magnitude(clean))


def make_loaders(cfg: dict, root: Path, generator: torch.Generator) -> tuple[DataLoader, DataLoader]:
    d, t = cfg["data"], cfg["train"]
    manifest = root / d["manifest"]
    common = {"batch_size": t["batch_size"], "num_workers": t["num_workers"], "worker_init_fn": seed_worker}
    train_ds = NoisySpeechDataset(manifest, "train", d["segment_seconds"], d["sr"])
    val_ds = NoisySpeechDataset(manifest, "val", d["segment_seconds"], d["sr"])
    return (DataLoader(train_ds, shuffle=True, generator=generator, **common),
            DataLoader(val_ds, shuffle=False, **common))


def run_epoch(model: SpectralEnhancer, loader: DataLoader, loss_name: str, device: torch.device, amp: bool,
              optimizer: torch.optim.Optimizer | None = None, scaler: torch.amp.GradScaler | None = None,
              grad_clip: float | None = None) -> tuple[float, float]:
    """Bir epoch çalıştırır. optimizer verilirse eğitir, verilmezse değerlendirir.

    Döner: (ortalama kayıp, ortalama SI-SNR dB; eğitimde nan).
    """
    training = optimizer is not None
    model.train(training)
    loss_sum, si_sum, count = 0.0, 0.0, 0
    with torch.set_grad_enabled(training):
        for batch in tqdm(loader, desc="train" if training else "val", leave=False):
            noisy, clean = batch["noisy"].to(device), batch["clean"].to(device)
            with torch.autocast(device.type, enabled=amp):
                out = model(noisy)
            loss = compute_loss(loss_name, model, out, clean)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if grad_clip:
                    scaler.unscale_(optimizer)  # kırpma gerçek gradyan büyüklüğüne uygulanmalı
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                si_sum += si_snr(out["est_wave"].float(), clean).sum().item()
            loss_sum += loss.item() * noisy.shape[0]
            count += noisy.shape[0]
    return loss_sum / count, (math.nan if training else si_sum / count)


def train(cfg: dict, root: Path) -> Path:
    """Eğitimi çalıştırır, deney klasörünü döndürür."""
    t = cfg["train"]
    out_dir = root / cfg["output_dir"]
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} boş değil; yeni deney için config'de farklı bir output_dir verin")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (out_dir / "run_info.json").write_text(json.dumps(run_info(root), indent=2), encoding="utf-8")

    seed_everything(cfg["seed"])
    device = resolve_device(cfg)
    amp = bool(t["amp"]) and device.type == "cuda"
    generator = torch.Generator().manual_seed(cfg["seed"])
    train_loader, val_loader = make_loaders(cfg, root, generator)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=t["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=t["lr_factor"], patience=t["lr_patience"])
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    loss_name = cfg["loss"]["name"]
    log.info("cihaz %s, AMP %s, %d parametre, train %d / val %d örnek", device, amp,
             sum(p.numel() for p in model.parameters()), len(train_loader.dataset), len(val_loader.dataset))

    log_path = out_dir / "log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, LOG_COLUMNS).writeheader()

    best_val, epochs_since_best = math.inf, 0
    for epoch in range(1, t["epochs"] + 1):
        start = time.perf_counter()
        train_loss, _ = run_epoch(model, train_loader, loss_name, device, amp, optimizer, scaler, t["grad_clip"])
        val_loss, val_si_snr = run_epoch(model, val_loader, loss_name, device, amp)
        scheduler.step(val_loss)

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_si_snr": val_si_snr,
               "lr": optimizer.param_groups[0]["lr"], "seconds": round(time.perf_counter() - start, 2)}
        with log_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, LOG_COLUMNS).writerow(row)

        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                 "val_loss": val_loss, "config": cfg}
        torch.save(state, out_dir / "last.pt")
        improved = val_loss < best_val
        if improved:
            best_val, epochs_since_best = val_loss, 0
            torch.save(state, out_dir / "best.pt")
        else:
            epochs_since_best += 1
        log.info("epoch %3d | train %.4f | val %.4f | val SI-SNR %.2f dB | lr %.1e | %.1f sn%s",
                 epoch, train_loss, val_loss, val_si_snr, row["lr"], row["seconds"], " *" if improved else "")
        if epochs_since_best >= t["early_stop_patience"]:
            log.info("Erken durdurma: %d epoch boyunca doğrulama kaybı iyileşmedi", epochs_since_best)
            break
    log.info("En iyi doğrulama kaybı %.4f -> %s", best_val, out_dir / "best.pt")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="SesTemiz eğitimi")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    train(yaml.safe_load(args.config.read_text(encoding="utf-8")), Path.cwd())


if __name__ == "__main__":
    main()

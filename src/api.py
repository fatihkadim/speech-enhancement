"""Web arayüzü için HTTP API (FastAPI).

Kullanım:
    uv run python -m src.api            # http://localhost:8000
    uv run python -m src.api --port 9000 --device cpu

`web/dist` derlenmişse (cd web && npm run build) arayüz de aynı adresten sunulur.
Geliştirme sırasında arayüz `npm run dev` ile 5173'te çalışır ve /api isteklerini buraya yönlendirir.
"""
import argparse
import io
import json
import logging
import math
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from src.evaluate import compute_metrics
from src.infer import enhance
from src.models.enhancer import SpectralEnhancer, load_model
from src.utils.audio import load_audio
from src.utils.plotting import DYNAMIC_RANGE_DB, spectrogram_db

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
REAL_DIR = ROOT / "samples" / "gercek"   # gerçek telefon kayıtları (E7), temiz referansı yok
REAL_PREFIX = "gercek_"
RELEASE_URL = "https://github.com/fatihkadim/speech-enhancement/releases/download/modeller-v1"
SEED = 42
MAX_SECONDS = 120          # daha uzun kayıtlar reddedilir (GPU belleği)
MAX_FRAMES = 1600          # spektrogram zaman ekseni bu sayıya seyreltilir (yanıt boyutu)
MAX_RESULTS = 24           # bellekte tutulan son sonuç sayısı
EPS = 1e-8

# Arayüzde gösterilen modeller: Release'teki en iyi iki model (bkz. README sonuç tablosu)
MODELS = {
    "e3_depth5": {"label": "Derin U-Net", "detail": "derinlik 5, L1 kaybı; PESQ ve STOI'de en iyisi"},
    "e1_si_snr": {"label": "SI-SNR kaybı", "detail": "derinlik 4; SI-SNR'da en iyisi"},
}
NOISE_NAMES = {"hen": "tavuk", "mouse_click": "fare tıklaması", "train": "tren", "sheep": "koyun",
               "rooster": "horoz", "thunderstorm": "fırtına", "footsteps": "ayak sesi",
               "vacuum_cleaner": "elektrikli süpürge", "siren": "siren"}


def model_path(name: str) -> Path:
    """Yerel checkpoint; yoksa Release'ten models/ altına indirilir (demo notebook'la aynı kural)."""
    local = ROOT / "experiments" / name / f"seed{SEED}" / "best.pt"
    if local.exists():
        return local
    cached = ROOT / "models" / f"{name}_seed{SEED}.pt"
    if not cached.exists():
        cached.parent.mkdir(exist_ok=True)
        log.info("%s indiriliyor...", cached.name)
        torch.hub.download_url_to_file(f"{RELEASE_URL}/{cached.name}", str(cached))
    return cached


class Engine:
    """Modelleri bir kez yükler, sonuçları sınırlı bir önbellekte tutar."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.models: dict[str, tuple[SpectralEnhancer, dict]] = {}
        self.results: OrderedDict[str, dict] = OrderedDict()
        self.lock = threading.Lock()

    def model(self, name: str) -> tuple[SpectralEnhancer, dict]:
        if name not in MODELS:
            raise HTTPException(404, f"Bilinmeyen model: {name}")
        with self.lock:
            if name not in self.models:
                self.models[name] = load_model(model_path(name), self.device)
            return self.models[name]

    def store(self, result: dict) -> str:
        rid = uuid.uuid4().hex[:12]
        self.results[rid] = result
        while len(self.results) > MAX_RESULTS:
            self.results.popitem(last=False)
        return rid

    def run(self, noisy: torch.Tensor, name: str, clean: torch.Tensor | None, title: str) -> dict:
        model, cfg = self.model(name)
        sr, n_fft, hop = cfg["data"]["sr"], cfg["stft"]["n_fft"], cfg["stft"]["hop"]
        if noisy.numel() > MAX_SECONDS * sr:
            raise HTTPException(413, f"Kayıt {MAX_SECONDS} saniyeden uzun olamaz.")
        if noisy.numel() < n_fft:
            raise HTTPException(422, "Kayıt çok kısa.")

        start = time.perf_counter()
        with torch.no_grad():
            out = model(noisy[None].to(self.device))
            mag = model.magnitude(noisy[None].to(self.device))
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        est = out["est_wave"][0].float().cpu()
        est = est / max(1.0, float(est.abs().max()))  # kliplenmeyi önle
        mask = (out["est_mag"] / (mag + EPS)).clamp(0, 1)[0].float().cpu()

        db_noisy = spectrogram_db(noisy, n_fft, hop)
        db_est = spectrogram_db(est, n_fft, hop)
        vmax = float(max(db_noisy.max(), db_est.max()))
        stride = max(1, math.ceil(db_noisy.shape[1] / MAX_FRAMES))

        metrics = None
        if clean is not None:
            metrics = {"noisy": _clean_nan(compute_metrics(noisy, clean, sr)),
                       "enhanced": _clean_nan(compute_metrics(est, clean, sr))}
        removed_db = 10 * math.log10(float(noisy.pow(2).sum()) / max(float(est.pow(2).sum()), EPS))

        rid = self.store({"sr": sr, "noisy": noisy, "enhanced": est, "clean": clean, "title": title})
        return {
            "id": rid,
            "title": title,
            "model": name,
            "sr": sr,
            "duration": noisy.numel() / sr,
            "processing_ms": round(elapsed * 1000, 1),
            "device": self.device,
            "removed_db": round(removed_db, 2),
            "metrics": metrics,
            "has_clean": clean is not None,
            "spectrogram": {
                "bins": int(db_noisy.shape[0]),
                "frames": int(db_noisy[:, ::stride].shape[1]),
                "max_hz": sr / 2,
                "noisy": _to_u8((db_noisy[:, ::stride] - (vmax - DYNAMIC_RANGE_DB)) / DYNAMIC_RANGE_DB),
                "enhanced": _to_u8((db_est[:, ::stride] - (vmax - DYNAMIC_RANGE_DB)) / DYNAMIC_RANGE_DB),
                "mask_bins": int(mask.shape[0]),
                "mask": _to_u8(mask[:, ::stride]),
            },
        }


def _to_u8(x: torch.Tensor) -> str:
    """[0, 1] matris -> satır sıralı (frekans x zaman) uint8 baytlarının base64'ü."""
    import base64

    arr = (x.clamp(0, 1) * 255).round().to(torch.uint8).contiguous().numpy()
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _clean_nan(d: dict) -> dict:
    return {k: (None if isinstance(v, float) and math.isnan(v) else round(v, 3)) for k, v in d.items()}


def _samples() -> list[dict]:
    """Test bölmesi örnekleri (SNR'a göre) ve ardından gerçek kayıtlar (temiz hali yok)."""
    items = []
    for noisy in (ROOT / "samples").glob("snr*_noisy.wav"):
        stem = noisy.name.removesuffix("_noisy.wav")      # ör. snr+10_mouse_click
        snr, noise = stem.split("_", 1)
        items.append({"id": stem, "kind": "test", "snr": int(snr[3:]), "noise": noise,
                      "label": NOISE_NAMES.get(noise, noise.replace("_", " ")),
                      "duration": round(sf.info(noisy).duration, 2),
                      "has_clean": noisy.with_name(f"{stem}_clean.wav").exists()})
    items.sort(key=lambda s: s["snr"])
    return items + _real_samples()


def _real_samples() -> list[dict]:
    """`samples/gercek/kayitlar.json`'da listelenen telefon kayıtları (E7)."""
    meta_path = REAL_DIR / "kayitlar.json"
    if not meta_path.exists():
        return []
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    items = []
    for stem, m in sorted(meta.items()):
        noisy = REAL_DIR / f"{stem}_noisy.wav"
        if not noisy.exists():
            log.warning("Gerçek kayıt eksik: %s", noisy)
            continue
        items.append({"id": f"{REAL_PREFIX}{stem}", "kind": "real", "snr": None, "noise": None,
                      "label": m["label"], "duration": round(sf.info(noisy).duration, 2), "has_clean": False})
    return items


def _sample_path(sample: dict, kind: str) -> Path:
    """Örneğin gürültülü/temiz dosyasının yolu."""
    if sample["kind"] == "real":
        return REAL_DIR / f"{sample['id'].removeprefix(REAL_PREFIX)}_{kind}.wav"
    return ROOT / "samples" / f"{sample['id']}_{kind}.wav"


def _sample_title(sample: dict) -> str:
    label = sample["label"][0].upper() + sample["label"][1:]
    return f"{label}, gerçek kayıt" if sample["kind"] == "real" else f"{label}, {sample['snr']:+d} dB"


def _wav_bytes(wave: torch.Tensor, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wave.numpy(), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _read_upload(data: bytes, sr: int) -> torch.Tensor:
    try:
        arr, file_sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as exc:  # soundfile wav/flac/ogg okur; arayüz her şeyi wav'a çevirip gönderir
        raise HTTPException(415, "Ses dosyası okunamadı. WAV, FLAC veya OGG yükleyin.") from exc
    wave = torch.from_numpy(np.ascontiguousarray(arr)).mean(dim=1)
    if file_sr != sr:
        wave = AF.resample(wave, orig_freq=file_sr, new_freq=sr)
    return wave.contiguous()


def create_app(device: str) -> FastAPI:
    engine = Engine(device)
    app = FastAPI(title="SesTemiz API")

    @app.get("/api/info")
    def info() -> dict:
        gpu = torch.cuda.get_device_name(0) if device.startswith("cuda") else None
        return {"device": device, "gpu": gpu, "max_seconds": MAX_SECONDS,
                "models": [{"id": k, **v} for k, v in MODELS.items()], "samples": _samples()}

    @app.get("/api/samples/{sample_id}/{kind}.wav")
    def sample_audio(sample_id: str, kind: str) -> FileResponse:
        sample = next((s for s in _samples() if s["id"] == sample_id), None)
        if kind not in ("noisy", "clean") or sample is None or (kind == "clean" and not sample["has_clean"]):
            raise HTTPException(404, "Örnek bulunamadı.")
        return FileResponse(_sample_path(sample, kind), media_type="audio/wav")

    @app.post("/api/enhance/sample")
    def enhance_sample(sample_id: str = Form(...), model: str = Form(...)) -> dict:
        sample = next((s for s in _samples() if s["id"] == sample_id), None)
        if sample is None:
            raise HTTPException(404, "Örnek bulunamadı.")
        sr = engine.model(model)[1]["data"]["sr"]
        noisy = load_audio(_sample_path(sample, "noisy"), sr)
        clean = load_audio(_sample_path(sample, "clean"), sr) if sample["has_clean"] else None
        return engine.run(noisy, model, clean, _sample_title(sample))

    @app.post("/api/enhance")
    async def enhance_upload(file: UploadFile = File(...), model: str = Form(...),
                             title: str = Form("Kayıt")) -> dict:
        sr = engine.model(model)[1]["data"]["sr"]
        noisy = _read_upload(await file.read(), sr)
        return engine.run(noisy, model, None, title[:120])

    @app.get("/api/results/{rid}/{kind}.wav")
    def result_audio(rid: str, kind: str) -> Response:
        res = engine.results.get(rid)
        if res is None or kind not in ("noisy", "enhanced", "clean") or res.get(kind) is None:
            raise HTTPException(404, "Sonuç bulunamadı; sayfayı yenileyip tekrar deneyin.")
        headers = {"Content-Disposition": f'inline; filename="sestemiz_{kind}.wav"'}
        return Response(_wav_bytes(res[kind], res["sr"]), media_type="audio/wav", headers=headers)

    dist = ROOT / "web" / "dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="web")
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="SesTemiz web API'si")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import uvicorn

    log.info("cihaz: %s | http://%s:%d", args.device, args.host, args.port)
    uvicorn.run(create_app(args.device), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

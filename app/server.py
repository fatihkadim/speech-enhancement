"""SesTemiz web arayüzü için yerel sunucu (yalnızca standart kütüphane + projenin kendi bağımlılıkları).

`web/` klasörünü ve `samples/` seslerini sunar, `/api/enhance` ile modeli çalıştırır.

Kullanım:
    uv run python -m app.server                 # http://localhost:8080
    uv run python -m app.server --port 8080 --device cpu --open
"""
import argparse
import functools
import gzip as gzip_mod
import io
import json
import logging
import math
import mimetypes
import re
import socket
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
SAMPLES_DIR = ROOT / "samples"
RELEASE_URL = "https://github.com/fatihkadim/speech-enhancement/releases/download/modeller-v1"
REMOTE_MODELS = ("e3_depth5", "e1_si_snr")  # Release'te yayımlanan modeller
OPTIONAL_MODELS = ("baseline", "e1_mse", "e2_no_skip", "e3_depth3", "e4_direct")  # yalnızca yerelde varsa
SEED = 42
MAX_BYTES = 60 * 1024 * 1024
MAX_SECONDS = 120

log = logging.getLogger("sestemiz")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("audio/wav", ".wav")


def local_checkpoint(name: str) -> Path | None:
    """experiments/ veya models/ altındaki checkpoint'i döndürür; yoksa None."""
    for p in (ROOT / "experiments" / name / f"seed{SEED}" / "best.pt", ROOT / "models" / f"{name}_seed{SEED}.pt"):
        if p.exists():
            return p
    return None


def available_models() -> list[dict]:
    models = [{"id": n, "local": local_checkpoint(n) is not None} for n in REMOTE_MODELS]
    models += [{"id": n, "local": True} for n in OPTIONAL_MODELS if local_checkpoint(n) is not None]
    return models


class Engine:
    """Modelleri tembel yükler ve önbellekte tutar; çıkarım tek kilit altında yapılır."""

    def __init__(self, device: str) -> None:
        self.device_arg = device
        self._models: dict[str, tuple] = {}
        self._lock = threading.Lock()
        self._torch = None

    @property
    def torch(self):
        if self._torch is None:
            import torch  # ağır; statik dosya sunmak için gerekmez
            self._torch = torch
        return self._torch

    @property
    def device(self) -> str:
        if self.device_arg == "auto":
            return "cuda" if self.torch.cuda.is_available() else "cpu"
        return self.device_arg

    def checkpoint(self, name: str) -> Path:
        path = local_checkpoint(name)
        if path is not None:
            return path
        if name not in REMOTE_MODELS:
            raise ValueError(f"Bilinmeyen model: {name}")
        cached = ROOT / "models" / f"{name}_seed{SEED}.pt"
        cached.parent.mkdir(exist_ok=True)
        log.info("Model indiriliyor: %s", cached.name)
        self.torch.hub.download_url_to_file(f"{RELEASE_URL}/{cached.name}", str(cached))
        return cached

    def model(self, name: str):
        if name not in self._models:
            from src.models.enhancer import load_model
            model, cfg = load_model(self.checkpoint(name), self.device)
            with self.torch.no_grad():  # CUDA çekirdeklerini ısıt: ilk gerçek istek yavaş kalmasın
                model(self.torch.zeros(1, cfg["data"]["sr"], device=self.device))
            self._models[name] = (model, cfg)
        return self._models[name]

    def enhance(self, wav_bytes: bytes, name: str, ref: str | None) -> tuple[bytes, dict]:
        import soundfile as sf
        import torchaudio.functional as AF

        from src.infer import enhance

        torch = self.torch
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
        wave = torch.from_numpy(data.mean(axis=1))
        with self._lock:
            model, cfg = self.model(name)
            target_sr = cfg["data"]["sr"]
            if sr != target_sr:
                wave = AF.resample(wave, sr, target_sr)
            wave = wave[: MAX_SECONDS * target_sr]
            t0 = time.perf_counter()
            est = enhance(model, wave)
            if self.device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
        peak = float(est.abs().max())
        if peak > 1:
            est = est / peak
        info = {"elapsed": elapsed, "device": self.device, "metrics": None}
        if ref:
            info["metrics"] = self.metrics(wave, est, ref, target_sr)
        out = io.BytesIO()
        sf.write(out, est.numpy(), target_sr, format="WAV", subtype="PCM_16")
        return out.getvalue(), info

    @staticmethod
    def metrics(noisy, est, ref: str, sr: int) -> dict | None:
        """Örnek kayıtlarda temiz referansa göre SI-SNR / PESQ / STOI (önce ve sonra)."""
        path = SAMPLES_DIR / f"{ref}_clean.wav"
        if not re.fullmatch(r"(tr/)?[\w+\-]+", ref) or not path.exists():
            return None
        try:
            from src.evaluate import compute_metrics
            from src.utils.audio import load_audio

            clean = load_audio(path, sr)
            n = min(len(clean), len(noisy), len(est))
            clean_n, noisy_n, est_n = clean[:n], noisy[:n], est[:n]
            clean_v = lambda d: {k: (None if v is None or math.isnan(v) else v) for k, v in d.items()}
            return {"noisy": clean_v(compute_metrics(noisy_n, clean_n, sr)),
                    "enhanced": clean_v(compute_metrics(est_n, clean_n, sr))}
        except Exception:  # metrikler isteğe bağlı; çıkarımı bozmasın
            log.exception("Metrik hesaplanamadı")
            return None


@functools.lru_cache(maxsize=256)
def _read_cached(path: Path, mtime_ns: int, gzip: bool) -> bytes:
    """Dosya içeriği (isteğe bağlı gzip'li); mtime anahtarın parçası olduğundan değişince yenilenir."""
    data = path.read_bytes()
    return gzip_mod.compress(data, compresslevel=6) if gzip else data


class Handler(BaseHTTPRequestHandler):
    engine: Engine
    server_version = "SesTemiz/1.0"

    def log_message(self, fmt, *args):  # sessiz erişim günlüğü
        log.debug("%s %s", self.address_string(), fmt % args)

    # ---- yanıt yardımcıları
    def _send(self, status: int, body: bytes, ctype: str, headers: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store" if ctype.startswith("application/json") else "no-cache")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        self._send(status, json.dumps(obj).encode(), "application/json; charset=utf-8")

    def _error(self, status: int, msg: str) -> None:
        self._send(status, msg.encode(), "text/plain; charset=utf-8")

    # ---- yönlendirme
    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/api/status":
            try:
                self._json({"ok": True, "device": self.engine.device, "models": available_models()})
            except Exception as e:  # torch kurulu değilse
                self._json({"ok": False, "error": str(e)}, 503)
            return
        if path.startswith("/samples/"):
            return self._static(SAMPLES_DIR, path[len("/samples/"):])
        return self._static(WEB_DIR, path.lstrip("/") or "index.html")

    def do_POST(self) -> None:
        url = urlparse(self.path)
        if url.path != "/api/enhance":
            return self._error(404, "Bulunamadı")
        q = parse_qs(url.query)
        name = q.get("model", [REMOTE_MODELS[0]])[0]
        ref = q.get("ref", [None])[0]
        if name not in {m["id"] for m in available_models()}:
            return self._error(400, f"Bilinmeyen model: {name}")
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_BYTES:
            return self._error(413, "Ses dosyası boş ya da çok büyük")
        try:
            body, info = self.engine.enhance(self.rfile.read(length), name, ref)
        except Exception as e:
            log.exception("Çıkarım hatası")
            return self._error(500, str(e))
        self._send(200, body, "audio/wav", {
            "X-Elapsed": f"{info['elapsed']:.4f}",
            "X-Device": info["device"],
            "X-Metrics": json.dumps(info["metrics"]),
        })

    def _static(self, base: Path, rel: str) -> None:
        """Statik dosya: ETag ile önbellek doğrulama (değişmediyse 304, gövde yok) ve metinlerde gzip."""
        target = (base / rel).resolve()
        if not target.is_relative_to(base.resolve()) or not target.is_file():
            return self._error(404, "Bulunamadı")
        st = target.stat()
        etag = f'"{st.st_mtime_ns:x}-{st.st_size:x}"'
        if etag in (self.headers.get("If-None-Match") or ""):
            self.send_response(304)
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            return
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        text = ctype.startswith("text/") or ctype.endswith(("javascript", "json", "svg+xml"))
        if text:
            ctype += "; charset=utf-8"
        headers = {"ETag": etag, "Vary": "Accept-Encoding"}
        body = _read_cached(target, st.st_mtime_ns, gzip=False)
        if text and "gzip" in (self.headers.get("Accept-Encoding") or "") and len(body) > 1024:
            body = _read_cached(target, st.st_mtime_ns, gzip=True)
            headers["Content-Encoding"] = "gzip"
        self._send(200, body, ctype, headers)


def main() -> None:
    parser = argparse.ArgumentParser(description="SesTemiz web arayüzü")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--open", action="store_true", help="tarayıcıyı aç")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    Handler.engine = Engine(args.device)
    # torch importu birkaç saniye sürer; ilk /api/status isteği beklemesin diye arka planda ısıt
    threading.Thread(target=lambda: Handler.engine.device, daemon=True).start()
    servers = make_servers(args.host, args.port)
    for extra in servers[1:]:
        threading.Thread(target=extra.serve_forever, daemon=True).start()
    url = f"http://{args.host}:{args.port}/"
    log.info("SesTemiz arayüzü: %s", url)
    if args.open:
        webbrowser.open(url)
    try:
        servers[0].serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for s in servers:
            s.server_close()


class _V6Server(ThreadingHTTPServer):
    address_family = socket.AF_INET6


def make_servers(host: str, port: int) -> list[ThreadingHTTPServer]:
    """`localhost` için hem 127.0.0.1 hem ::1 dinlenir. Aksi halde Windows'ta tarayıcı önce ::1'i deneyip
    her istekte zaman aşımı bekler (senkronda gecikme olarak görünür)."""
    if host != "localhost":
        return [ThreadingHTTPServer((host, port), Handler)]
    servers = [ThreadingHTTPServer(("127.0.0.1", port), Handler)]
    try:
        servers.append(_V6Server(("::1", port), Handler))
    except OSError:
        log.info("IPv6 (::1) dinlenemedi; yalnızca 127.0.0.1 kullanılıyor")
    return servers


if __name__ == "__main__":
    main()

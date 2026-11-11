"""Web sunucusu ile model arasındaki senkronun testi.

Arayüzün yaptığı gibi 16 kHz PCM16 WAV gönderilir; sunucunun döndürdüğü ses, aynı modelin doğrudan
`src.infer.enhance` çıktısıyla örnek örnek karşılaştırılır (uzunluk, gecikme, genlik, metrikler).
"""
import io
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pytest
import soundfile as sf
import torch

from app import server as srv
from src.infer import enhance
from src.models.enhancer import build_model

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = "snr-5_hen"


@pytest.fixture(scope="module")
def running(tmp_path_factory):
    """Rastgele ağırlıklı küçük bir modeli sunucuya enjekte edip arka planda başlatır."""
    torch.manual_seed(0)
    cfg = {"model": {"name": "unet", "base_ch": 8, "depth": 3, "skip": True, "out_mode": "mask"},
           "stft": {"n_fft": 512, "hop": 128}, "data": {"sr": 16000}}
    model = build_model(cfg).eval()
    engine = srv.Engine("cpu")
    engine._models["e3_depth5"] = (model, cfg)
    srv.Handler.engine = engine
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}", model
    httpd.shutdown()


def _post(base: str, wav: bytes, query: str):
    req = Request(f"{base}/api/enhance?{query}", data=wav, method="POST", headers={"Content-Type": "audio/wav"})
    with urlopen(req) as r:
        return r.read(), dict(r.headers)


def _pcm16(x: np.ndarray, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_status(running):
    base, _ = running
    with urlopen(f"{base}/api/status") as r:
        j = json.loads(r.read())
    assert j["ok"] and j["device"] == "cpu"
    assert {m["id"] for m in j["models"]} >= {"e3_depth5", "e1_si_snr"}


def test_enhance_matches_direct_inference(running):
    base, model = running
    noisy, sr = sf.read(ROOT / "samples" / f"{SAMPLE}_noisy.wav", dtype="float32")
    body, headers = _post(base, _pcm16(noisy), f"model=e3_depth5&ref={SAMPLE}")
    out, out_sr = sf.read(io.BytesIO(body), dtype="float32")

    direct = enhance(model, torch.from_numpy(noisy)).numpy()
    direct = direct / max(1.0, np.abs(direct).max())

    assert out_sr == sr == 16000
    assert len(out) == len(noisy)                                   # örnek sayısı birebir
    lag = np.argmax(np.correlate(out, direct, "full")) - (len(direct) - 1)
    assert lag == 0                                                 # zaman kayması yok
    err = 10 * np.log10(np.sum(direct ** 2) / np.sum((out - direct) ** 2))
    assert err > 40                                                 # yalnızca PCM16 nicemleme farkı

    m = json.loads(headers["X-Metrics"])
    assert m["noisy"]["si_snr"] == pytest.approx(-5.0, abs=0.2)     # −5 dB SNR'da karıştırılmıştı (SI-SNR ≈ SNR)
    assert float(headers["X-Elapsed"]) > 0 and headers["X-Device"] == "cpu"


def test_resamples_other_rates(running):
    base, _ = running
    noisy, _ = sf.read(ROOT / "samples" / f"{SAMPLE}_noisy.wav", dtype="float32")
    x44 = np.interp(np.linspace(0, len(noisy) - 1, int(len(noisy) * 44100 / 16000)), np.arange(len(noisy)), noisy)
    body, _ = _post(base, _pcm16(x44.astype(np.float32), 44100), "model=e3_depth5")
    out, out_sr = sf.read(io.BytesIO(body))
    assert out_sr == 16000 and abs(len(out) - len(noisy)) <= 2


@pytest.mark.parametrize("query,code", [("model=yok", 400), ("model=e3_depth5&ref=../README", 200)])
def test_rejects_bad_input(running, query, code):
    base, _ = running
    noisy, _ = sf.read(ROOT / "samples" / f"{SAMPLE}_noisy.wav", dtype="float32")
    try:
        _, headers = _post(base, _pcm16(noisy), query)
        assert code == 200 and json.loads(headers["X-Metrics"]) is None  # geçersiz referans yok sayılır
    except Exception as e:  # urllib HTTPError
        assert getattr(e, "code", None) == code


def test_static_path_traversal_blocked(running):
    base, _ = running
    with pytest.raises(Exception) as e:
        urlopen(f"{base}/samples/..%2F..%2FREADME.md")
    assert getattr(e.value, "code", None) == 404

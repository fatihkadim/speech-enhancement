"""Web API'sinin model gerektirmeyen uçları."""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from src.api import MODELS, REAL_PREFIX, create_app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app("cpu"))


def test_info_lists_models_and_samples(client: TestClient) -> None:
    info = client.get("/api/info").json()
    assert [m["id"] for m in info["models"]] == list(MODELS)
    assert info["device"] == "cpu"
    for s in info["samples"]:
        if s["kind"] == "test":
            assert s["id"].startswith("snr") and -5 <= s["snr"] <= 15
        else:
            assert s["kind"] == "real" and s["id"].startswith(REAL_PREFIX)
            assert s["snr"] is None and not s["has_clean"] and s["duration"] > 0


def test_real_samples_listed_after_test_samples(client: TestClient) -> None:
    kinds = [s["kind"] for s in client.get("/api/info").json()["samples"]]
    assert kinds == sorted(kinds, key=lambda k: k == "real")  # önce test, sonra gerçek


def test_real_sample_audio(client: TestClient) -> None:
    real = [s for s in client.get("/api/info").json()["samples"] if s["kind"] == "real"]
    if not real:
        pytest.skip("samples/gercek boş")
    sid = real[0]["id"]
    res = client.get(f"/api/samples/{sid}/noisy.wav")
    assert res.status_code == 200 and res.headers["content-type"] == "audio/wav"
    assert client.get(f"/api/samples/{sid}/clean.wav").status_code == 404  # temiz hali yok


def test_unknown_model_and_sample_are_404(client: TestClient) -> None:
    assert client.post("/api/enhance/sample", data={"sample_id": "snr+0_hen", "model": "yok"}).status_code == 404
    assert client.get("/api/samples/yok/noisy.wav").status_code == 404
    assert client.get("/api/results/yok/enhanced.wav").status_code == 404

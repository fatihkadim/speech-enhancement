"""Web API'sinin model gerektirmeyen uçları."""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from src.api import MODELS, create_app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app("cpu"))


def test_info_lists_models_and_samples(client: TestClient) -> None:
    info = client.get("/api/info").json()
    assert [m["id"] for m in info["models"]] == list(MODELS)
    assert info["device"] == "cpu"
    for s in info["samples"]:
        assert s["id"].startswith("snr") and -5 <= s["snr"] <= 15


def test_unknown_model_and_sample_are_404(client: TestClient) -> None:
    assert client.post("/api/enhance/sample", data={"sample_id": "snr+0_hen", "model": "yok"}).status_code == 404
    assert client.get("/api/samples/yok/noisy.wav").status_code == 404
    assert client.get("/api/results/yok/enhanced.wav").status_code == 404

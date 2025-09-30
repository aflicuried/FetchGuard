from fastapi.testclient import TestClient
from src.api.main import app


client = TestClient(app)


def test_predict_stub():
    payload = {"features": [0.0, 0.0, 0.0]}
    resp = client.post("/api/ctg/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["label"] in {"Normal", "Suspect", "Pathologic"}



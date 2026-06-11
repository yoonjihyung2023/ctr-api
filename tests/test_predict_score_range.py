from fastapi.testclient import TestClient

try:
    from app.main import app
except Exception:
    try:
        from src.main import app
    except Exception:
        from main import app


client = TestClient(app)


def test_predict_score_range():
    payloads = [
        {"features": {}},
        {"user_id": "u1", "item_id": "i1", "features": {}},
        {"user_id": "u1", "item_id": "i1"},
    ]

    for payload in payloads:
        response = client.post("/predict", json=payload)
        if response.status_code == 200:
            data = response.json()
            assert "score" in data
            assert 0.0 <= float(data["score"]) <= 1.0
            return

    assert False, "No valid /predict response returned"

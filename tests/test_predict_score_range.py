from fastapi.testclient import TestClient

from app.main import app, clamp_score


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_predict_score_range():
    response = client.post(
        "/predict",
        json={
            "user_id": "u1",
            "item_id": "i1",
            "features": {"site_id": "s1", "device_type": 1},
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert 0.0 <= float(data["score"]) <= 1.0


def test_clamp_score():
    assert clamp_score(-1.0) == 0.0
    assert clamp_score(2.0) == 1.0
    assert clamp_score(0.7) == 0.7

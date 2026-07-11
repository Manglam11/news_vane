import pytest
from fastapi.testclient import TestClient

from newsvane.api.main import app


@pytest.fixture
def client(monkeypatch):
    # I don't want a live database just to test the doorway. I swap save() for a
    # stand-in that records what it was handed, so the test stays fast and offline.
    saved = []

    def fake_save(text, prediction):
        saved.append((text, prediction))
        return None

    monkeypatch.setattr("newsvane.api.main.save", fake_save)
    test_client = TestClient(app)
    test_client.saved = saved
    return test_client


def test_classify_returns_the_contract(client):
    response = client.post("/classify", json={"text": "Markets rallied today."})

    assert response.status_code == 200
    assert set(response.json().keys()) == {"label", "score", "sentiment"}


def test_classify_persists_the_prediction(client):
    client.post("/classify", json={"text": "Markets rallied today."})

    assert len(client.saved) == 1
    assert client.saved[0][0] == "Markets rallied today."


def test_classify_rejects_a_bad_payload(client):
    response = client.post("/classify", json={"wrong_key": "oops"})

    assert response.status_code == 422

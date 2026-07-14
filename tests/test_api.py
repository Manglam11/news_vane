"""API tests -- the doorway, against a real database.

The model is still stubbed here on purpose: I am testing the endpoint, not
the brain. If a model bug broke these tests, the failure would point at the
wrong box. But storage is NOT stubbed any more -- a constraint or a default
only exists inside Postgres, so only Postgres can prove it fires.
"""

import pytest
from fastapi.testclient import TestClient

from newsvane.api import main
from newsvane.api.main import app


@pytest.fixture(autouse=True)
def stub_model(monkeypatch):
    # One box under test at a time. The doorway does not care what the brain
    # says, only that it passes the answer through unchanged.
    monkeypatch.setattr(
        main,
        "predict",
        lambda text: {"label": "Sports", "score": 0.9, "sentiment": "neutral"},
    )


@pytest.fixture
def client():
    return TestClient(app)


def test_classify_returns_the_model_contract_plus_an_id(client):
    response = client.post("/classify", json={"text": "the match ended in a draw"})

    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "Sports"
    assert body["score"] == 0.9
    assert body["sentiment"] == "neutral"
    # The id is what makes the prediction addressable -- without it no human
    # could ever point back at this row and correct it.
    assert body["id"] > 0


def test_classify_rejects_empty_text(client):
    response = client.post("/classify", json={"text": ""})

    assert response.status_code == 422


def test_feedback_is_recorded_against_a_real_prediction(client):
    prediction_id = client.post("/classify", json={"text": "a goal"}).json()["id"]

    response = client.post(
        "/feedback",
        json={"prediction_id": prediction_id, "correct_label": "World"},
    )

    assert response.status_code == 200
    assert response.json()["prediction_id"] == prediction_id


def test_feedback_cannot_be_given_twice(client):
    prediction_id = client.post("/classify", json={"text": "a goal"}).json()["id"]
    payload = {"prediction_id": prediction_id, "correct_label": "World"}
    client.post("/feedback", json=payload)

    response = client.post("/feedback", json=payload)

    # The unique constraint fires inside Postgres. A stub could never prove this.
    assert response.status_code == 409


def test_feedback_on_an_unknown_prediction_is_rejected(client):
    response = client.post(
        "/feedback",
        json={"prediction_id": 999999, "correct_label": "World"},
    )

    # The foreign key fires inside Postgres. Same story.
    assert response.status_code == 409


def test_predictions_are_returned_newest_first(client):
    client.post("/classify", json={"text": "first"})
    client.post("/classify", json={"text": "second"})

    body = client.get("/predictions").json()

    assert [row["text"] for row in body] == ["second", "first"]


def test_predictions_can_be_filtered_by_label(client):
    client.post("/classify", json={"text": "a goal"})

    assert len(client.get("/predictions", params={"label": "Sports"}).json()) == 1
    assert len(client.get("/predictions", params={"label": "World"}).json()) == 0

def test_save_articles_skips_duplicates():
    """The scraper re-reads the same front page every day. If the same story could
    land twice, tomorrow's volume trend would be pure invention -- so I prove that
    the database itself refuses the second copy.
    """
    from datetime import UTC, datetime

    from newsvane.storage.repository import save_articles

    published = datetime(2026, 7, 14, 9, 0, tzinfo=UTC)
    batch = [
        {"text": "Apple sues OpenAI.", "topic": "Sci/Tech", "timestamp": published},
        {"text": "India wins the series.", "topic": "Sports", "timestamp": published},
    ]

    assert save_articles(batch) == 2

    # Same two stories, plus one genuinely new one. Only the new one may land.
    batch.append(
        {"text": "Rupee steadies against dollar.", "topic": "Business", "timestamp": published}
    )
    assert save_articles(batch) == 1
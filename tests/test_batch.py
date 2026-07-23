"""Prove the exam room -- the model marked against the sections it never saw.

classify_articles is the only piece that decides an unreadable article becomes an
ALARM rather than POISON, and on a normal morning that branch never runs. So it is
proven here deliberately, or it is not proven at all. The model is stubbed: I am
testing the loop and its failure handling, not the brain.
"""

from newsvane.models import batch


def article(text: str, topic: str) -> dict:
    # The bare DATA contract, exactly as the scraper hands it over.
    return {"text": text, "topic": topic, "timestamp": None}


def test_every_article_comes_back_with_the_model_answer(monkeypatch):
    monkeypatch.setattr(
        batch,
        "predict",
        lambda text: {"label": "Sports", "score": 0.9, "sentiment": "neutral"},
    )

    annotated, failures = batch.classify_articles(
        [article("a goal", "Sports"), article("a merger", "Business")]
    )

    assert failures == []
    assert [row["predicted_label"] for row in annotated] == ["Sports", "Sports"]
    assert [row["predicted_score"] for row in annotated] == [0.9, 0.9]


def test_the_original_harvest_is_left_untouched(monkeypatch):
    # The DATA box handed me that list. It gets it back exactly as it gave it --
    # I copy each row rather than writing my columns into someone else's object.
    monkeypatch.setattr(
        batch,
        "predict",
        lambda text: {"label": "Sports", "score": 0.9, "sentiment": "neutral"},
    )
    original = [article("a goal", "Sports")]

    batch.classify_articles(original)

    assert original == [{"text": "a goal", "topic": "Sports", "timestamp": None}]


def test_one_unreadable_article_costs_one_annotation(monkeypatch):
    # The same rule the scraper learned when a single ReadTimeout could kill a day:
    # a failure on row two must not cost rows one and three.
    def flaky(text: str) -> dict:
        if text == "broken":
            raise RuntimeError("could not read the graph")
        return {"label": "Sports", "score": 0.9, "sentiment": "neutral"}

    monkeypatch.setattr(batch, "predict", flaky)

    annotated, failures = batch.classify_articles(
        [article("a goal", "Sports"), article("broken", "World"), article("a try", "Sports")]
    )

    assert len(annotated) == 3
    assert [row["predicted_label"] for row in annotated] == ["Sports", None, "Sports"]
    # NULL means "never asked" and stays findable forever as predicted_label IS NULL.
    # That is why this is an alarm, never poison -- nothing is thrown away.
    assert annotated[1]["predicted_score"] is None
    assert len(failures) == 1
    assert "World" in failures[0]


def test_agreement_counts_only_what_the_model_answered():
    annotated = [
        {"topic": "Sports", "predicted_label": "Sports"},
        {"topic": "World", "predicted_label": "Business"},
        {"topic": "World", "predicted_label": None},
    ]

    # Two questions asked, one right. The unanswered row is in neither number --
    # a model outage must shrink the exam, never lower the mark.
    assert batch.agreement(annotated) == (1, 2)


def test_agreement_on_a_fully_unclassified_harvest():
    annotated = [{"topic": "World", "predicted_label": None}]

    agreed, scored = batch.agreement(annotated)

    assert (agreed, scored) == (0, 0)
    # The caller must be able to skip printing a rate rather than divide by zero.
    assert scored == 0

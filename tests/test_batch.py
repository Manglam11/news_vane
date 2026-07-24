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


def test_every_article_comes_back_with_a_mood(monkeypatch):
    monkeypatch.setattr(
        batch,
        "predict",
        lambda text: {"label": "Sports", "score": 0.9, "sentiment": "neutral"},
    )
    monkeypatch.setattr(batch, "read_sentiment", lambda text: ("positive", 0.4))

    annotated, failures = batch.classify_articles([article("a goal", "Sports")])

    assert failures == []
    assert annotated[0]["sentiment"] == "positive"
    assert annotated[0]["sentiment_score"] == 0.4


def test_a_dead_topic_brain_still_leaves_the_moods(monkeypatch):
    # The whole reason the two engines sit in separate try blocks. The topic
    # brain needs a 64 MB graph on disk; the mood reader needs nothing but the
    # text. A missing weights file must cost me the topics and STILL leave me a
    # mood series -- one engine falling over is not permission to lose the
    # other one's work.
    def no_graph(text: str) -> dict:
        raise RuntimeError("no such file: distilbert.int8.onnx")

    monkeypatch.setattr(batch, "predict", no_graph)
    monkeypatch.setattr(batch, "read_sentiment", lambda text: ("negative", -0.5))

    annotated, failures = batch.classify_articles([article("a war", "World")])

    assert annotated[0]["predicted_label"] is None
    assert annotated[0]["sentiment_score"] == -0.5
    assert len(failures) == 1


def test_a_dead_mood_reader_still_leaves_the_topics(monkeypatch):
    # And the same promise, read backwards.
    def no_lexicon(text: str) -> tuple[str, float]:
        raise RuntimeError("could not build the analyzer")

    monkeypatch.setattr(
        batch,
        "predict",
        lambda text: {"label": "World", "score": 0.8, "sentiment": "neutral"},
    )
    monkeypatch.setattr(batch, "read_sentiment", no_lexicon)

    annotated, failures = batch.classify_articles([article("a war", "World")])

    assert annotated[0]["predicted_label"] == "World"
    assert annotated[0]["sentiment"] is None
    assert annotated[0]["sentiment_score"] is None
    assert len(failures) == 1


def test_mood_by_section_reports_a_section_that_read_nothing():
    # Seeded from the SECTION LIST, never from the harvest. A mean built only
    # from what arrived can never show what did not arrive -- and None is the
    # honest answer, because 0.0 is what genuinely neutral news scores.
    annotated = [{"topic": "World", "sentiment_score": -0.5}]

    by_section = batch.mood_by_section(annotated, ["World", "Sports"])

    assert by_section["Sports"] == (None, 0)
    assert by_section["World"] == (-0.5, 1)


def test_mood_by_section_averages_only_what_it_could_read():
    # The denominator travels with the mean, and it counts READINGS, not rows.
    # An unread article must not drag the average towards zero.
    annotated = [
        {"topic": "Business", "sentiment_score": 0.6},
        {"topic": "Business", "sentiment_score": 0.2},
        {"topic": "Business", "sentiment_score": None},
    ]

    mean, read = batch.mood_by_section(annotated, ["Business"])["Business"]

    assert read == 2
    assert mean == 0.4


def test_mood_by_section_counts_a_label_it_did_not_expect():
    # The harvest_counts rule, applied to moods. A breakdown that silently omits
    # a row lies in exactly the way a total that hides a zero does.
    annotated = [{"topic": "Opinion", "sentiment_score": 0.1}]

    by_section = batch.mood_by_section(annotated, ["World"])

    assert by_section["Opinion"] == (0.1, 1)
    assert by_section["World"] == (None, 0)

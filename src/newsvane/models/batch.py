"""Ask the model what it thinks each scraped article is, and how it feels.

The scraper already knows the topic -- the section URL is ground truth -- so I
hand the model the text alone and never the section. That turns every harvest
into a small exam the model sits blind, and the columns it fills in are what let
drift ask a question about the MODEL rather than about my own scraping quota.

Mood has no such exam. Nothing on a news page tells me the right answer, so the
sentiment reading is recorded and trended, never marked.

This lives inside the MODEL box because it is the box's own work, batched. The
frozen contract predict(text) -> {label, score, sentiment} is not widened here;
it is called once per article, which is the only way a frozen door should ever
absorb new behaviour.
"""

from collections.abc import Iterable

from newsvane.models import predict
from newsvane.models.sentiment import read_sentiment


def classify_articles(articles: list[dict]) -> tuple[list[dict], list[str]]:
    """Return the harvest with the model's answers attached, plus any failures.

    Each returned dict is the DATA contract's {text, topic, timestamp} with four
    keys added -- I copy rather than mutate, so the list the DATA box handed me
    is still exactly what it handed me.

    Anything the model could not read keeps None. NULL means "never asked",
    which is a different fact from a wrong answer, and ANALYTICS has to be able
    to tell them apart. One unreadable article costs one annotation, never the
    whole harvest -- the same rule the scraper learned when a single ReadTimeout
    could kill an entire day.

    The two engines are asked in two separate blocks on purpose. They fail for
    completely different reasons -- the topic brain needs a 64 MB graph on disk,
    the mood reader needs nothing but the text -- so a missing weights file must
    cost me the topics and still leave me the moods. One engine falling over is
    not permission to lose the other one's work.
    """
    annotated: list[dict] = []
    failures: list[str] = []

    for article in articles:
        row = dict(article)

        try:
            answer = predict(article["text"])
            row["predicted_label"] = answer["label"]
            row["predicted_score"] = answer["score"]
        except Exception as error:
            row["predicted_label"] = None
            row["predicted_score"] = None
            failures.append(f"{article['topic']}: could not classify -- {error}")

        try:
            # I call the engine directly here, not the door. The door hands back
            # the mood as a word because that is what the frozen contract says;
            # a daily mood average needs the raw compound behind it, and this is
            # the one caller that needs the number. Both live inside this box.
            mood, compound = read_sentiment(article["text"])
            row["sentiment"] = mood
            row["sentiment_score"] = compound
        except Exception as error:
            row["sentiment"] = None
            row["sentiment_score"] = None
            failures.append(f"{article['topic']}: could not read mood -- {error}")

        annotated.append(row)

    return annotated, failures


def agreement(articles: list[dict]) -> tuple[int, int]:
    """Count how many classified rows the model labelled the same as their section.

    Rows the model never answered are excluded from BOTH numbers. Counting an
    unasked question as a wrong answer would quietly drag the score down and
    make a model failure look like a model mistake.

    Mood deliberately has no equivalent function. There is no section URL for
    "how this article feels", so there is nothing to mark it against.
    """
    scored = [a for a in articles if a.get("predicted_label") is not None]
    agreed = sum(1 for a in scored if a["predicted_label"] == a["topic"])
    return agreed, len(scored)


def mood_by_section(
    articles: list[dict],
    sections: Iterable[str],
) -> dict[str, tuple[float | None, int]]:
    """Average compound score per section, with the count it was averaged over.

    One global mood number would be the same disease as "saved 58 new" -- an
    aggregate I would act on, hiding whichever section went silent. So the
    reading is broken down by the dimension that matters, and it is seeded from
    the SECTION LIST rather than from the harvest, because a mean built only
    from what arrived can never show what did not.

    The count travels beside the mean because a mean without its denominator is
    a decimal pretending to be an answer. An average of +0.4 over two articles
    and over forty are not the same fact.

    A section nobody could read returns None rather than 0.0. Neutral news
    genuinely scores 0.0, so returning a zero for "never read" would make an
    engine failure indistinguishable from a quiet news day.
    """
    totals: dict[str, float] = dict.fromkeys(sections, 0.0)
    counts: dict[str, int] = dict.fromkeys(totals, 0)

    for article in articles:
        compound = article.get("sentiment_score")
        if compound is None:
            continue
        # A label I did not expect is counted, never dropped -- the same rule
        # harvest_counts follows. A breakdown that silently omits a row lies in
        # exactly the way a total that hides a zero does.
        topic = article["topic"]
        totals[topic] = totals.get(topic, 0.0) + compound
        counts[topic] = counts.get(topic, 0) + 1

    return {topic: (totals[topic] / read if read else None, read) for topic, read in counts.items()}

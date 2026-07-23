"""Ask the model what it thinks each scraped article is.

The scraper already knows the answer -- the section URL is ground truth -- so I
hand the model the text alone and never the section. That turns every harvest
into a small exam the model sits blind, and the two columns it fills in are what
let drift ask a question about the MODEL rather than about my own scraping quota.

This lives inside the MODEL box because it is the box's own work, batched. The
frozen contract predict(text) -> {label, score, sentiment} is not widened here;
it is called once per article, which is the only way a frozen door should ever
absorb new behaviour.
"""

from newsvane.models import predict


def classify_articles(articles: list[dict]) -> tuple[list[dict], list[str]]:
    """Return the harvest with the model's answer attached, plus any failures.

    Each returned dict is the DATA contract's {text, topic, timestamp} with
    predicted_label and predicted_score added -- I copy rather than mutate, so
    the list the DATA box handed me is still exactly what it handed me.

    A row the model could not read keeps both as None. NULL means "never asked",
    which is a different fact from a wrong answer, and ANALYTICS has to be able
    to tell them apart. One unreadable article costs one annotation, never the
    whole harvest -- the same rule the scraper learned when a single ReadTimeout
    could kill an entire day.
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
        annotated.append(row)

    return annotated, failures


def agreement(articles: list[dict]) -> tuple[int, int]:
    """Count how many classified rows the model labelled the same as their section.

    Rows the model never answered are excluded from BOTH numbers. Counting an
    unasked question as a wrong answer would quietly drag the score down and
    make a model failure look like a model mistake.
    """
    scored = [a for a in articles if a.get("predicted_label") is not None]
    agreed = sum(1 for a in scored if a["predicted_label"] == a["topic"])
    return agreed, len(scored)

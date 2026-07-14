"""Read and write predictions -- the STORAGE box's actual job.

This is the repository: the single place that knows how to turn a prediction
into a saved row and hand back what landed in the database. Every other box
goes through here, so nothing else ever touches Postgres directly.
"""


import hashlib
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError

from newsvane.storage.database import SessionLocal
from newsvane.storage.models import Article, Feedback, Prediction


def save(text: str, prediction: dict) -> Prediction:
    """Save one prediction as a row and return it, now carrying its db id.

    I take the original text plus the model's {label, score, sentiment}
    answer, build a row, and commit it. The created_at timestamp is stamped
    by Postgres itself, not by Python.
    """
    with SessionLocal() as session:
        row = Prediction(
            text=text,
            label=prediction["label"],
            score=prediction["score"],
            sentiment=prediction["sentiment"],
        )
        session.add(row)  # stage the row -- in the cart, not bought yet
        session.commit()  # checkout -- the row is now permanent in Postgres
        session.refresh(row)  # get my receipt: the new id and created_at
        return row


def save_feedback(prediction_id: int, correct_label: str) -> Feedback | None:
    """Record a human correction against one prediction.

    Returns None when the row cannot be written -- either the prediction id
    does not exist, or this prediction has already been corrected once. I let
    the database be the judge of both, because it is the only thing that can
    answer truthfully under concurrent writes.
    """
    with SessionLocal() as session:
        row = Feedback(prediction_id=prediction_id, correct_label=correct_label)
        session.add(row)
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            return None
        session.refresh(row)
        return row

def fetch(
    start: datetime,
    end: datetime,
    label: str | None = None,
    limit: int = 100,
) -> list[Prediction]:
    """Fetch predictions made inside a time window, newest first.

    The window is half-open -- start is included, end is not -- so calling
    this for consecutive days can never count the same row twice. That
    property is what lets ANALYTICS bucket rows by day without double-counting.
    """
    with SessionLocal() as session:
        stmt = (
            select(Prediction)
            .where(Prediction.created_at >= start, Prediction.created_at < end)
            .order_by(Prediction.created_at.desc())
            .limit(limit)
        )
        if label is not None:
            stmt = stmt.where(Prediction.label == label)
        return list(session.scalars(stmt))

def save_articles(articles: list[dict]) -> int:
    """Bulk-insert scraped articles, skipping any I already have. Returns the count saved.

    The scraper runs daily and a front page barely changes overnight, so most of what
    it hands me is already in the table. I let Postgres be the judge: INSERT everything
    and tell it to ignore any row whose text_hash already exists. A SELECT-then-INSERT
    would race, and a duplicate story is not a harmless duplicate -- it is a fake spike
    in tomorrow's trend line.
    """
    if not articles:
        return 0

    rows = [
        {
            "text": article["text"],
            "topic": article["topic"],
            "published_at": article["timestamp"],
            "text_hash": hashlib.sha256(article["text"].encode("utf-8")).hexdigest(),
        }
        for article in articles
    ]

    stmt = (
        insert(Article)
        .values(rows)
        .on_conflict_do_nothing(constraint="uq_articles_text_hash")
        .returning(Article.id)
    )

    with SessionLocal() as session:
        saved = list(session.scalars(stmt))
        session.commit()
        return len(saved)
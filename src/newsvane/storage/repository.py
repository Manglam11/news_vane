"""Read and write predictions -- the STORAGE box's actual job.

This is the repository: the single place that knows how to turn a prediction
into a saved row and hand back what landed in the database. Every other box
goes through here, so nothing else ever touches Postgres directly.
"""

from sqlalchemy.exc import IntegrityError

from newsvane.storage.database import SessionLocal
from newsvane.storage.models import Feedback, Prediction


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
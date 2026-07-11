"""Read and write predictions -- the STORAGE box's actual job.

This is the repository: the single place that knows how to turn a prediction
into a saved row and hand back what landed in the database. Every other box
goes through here, so nothing else ever touches Postgres directly.
"""

from newsvane.storage.database import SessionLocal
from newsvane.storage.models import Prediction


def save(text: str, prediction: dict) -> Prediction:
    """Save one prediction as a row and return it, now carrying its db id.

    I take the original text plus the model's {label, score, sentiment}
    answer, build a row, and commit it. The created_at timestamp fills in
    automatically from the column default.
    """
    with SessionLocal() as session:
        row = Prediction(
            text=text,
            label=prediction["label"],
            score=prediction["score"],
            sentiment=prediction["sentiment"],
        )
        session.add(row)       # stage the row -- in the cart, not bought yet
        session.commit()       # checkout -- the row is now permanent in Postgres
        session.refresh(row)   # get my receipt: the new id and created_at
        return row
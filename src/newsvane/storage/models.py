"""Database tables for the STORAGE box, written as SQLAlchemy ORM models.

Each class here is one table; each instance is one row. I define them as
Python classes so I never hand-write raw SQL -- SQLAlchemy builds the real
tables from these definitions.
"""

from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    # Every table-class inherits from this shared base so SQLAlchemy can
    # track them all together and create them in one go.
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    # A surrogate primary key -- an auto-incrementing id for every row.
    id: Mapped[int] = mapped_column(primary_key=True)

    # The article text that was sent in. Text, not String: a headline plus a
    # description has no sane length cap, and I refuse to invent one.
    text: Mapped[str] = mapped_column(Text, nullable=False)

    # The model's answer -- the three fields of my MODEL contract.
    label: Mapped[str] = mapped_column(String(32), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    sentiment: Mapped[str] = mapped_column(String(16), nullable=False)

    # When this prediction was made. Postgres stamps it, not Python: the
    # database is the single clock every row is measured against, even if a
    # future scraper or worker writes from a different machine.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        # ANALYTICS will always ask "what happened between these two times,
        # for this topic?" -- so I index exactly that question. Without these,
        # every trend query is a full table scan that grows with the project.
        Index("ix_predictions_created_at", "created_at"),
        Index("ix_predictions_label_created_at", "label", "created_at"),
    )

class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(primary_key=True)

    # Which prediction was wrong. The foreign key means Postgres itself
    # refuses feedback pointing at a prediction that does not exist.
    prediction_id: Mapped[int] = mapped_column(
        ForeignKey("predictions.id", ondelete="CASCADE"),
        nullable=False,
    )

    # What the human says the label should have been. This column is the
    # entire point of the table -- it is the training data for v2.
    correct_label: Mapped[str] = mapped_column(String(32), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        # One correction per prediction. A human cannot vote twice.
        UniqueConstraint("prediction_id", name="uq_feedback_prediction_id"),
    )
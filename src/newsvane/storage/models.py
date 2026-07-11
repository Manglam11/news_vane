"""Database tables for the STORAGE box, written as SQLAlchemy ORM models.

Each class here is one table; each instance is one row. I define them as
Python classes so I never hand-write raw SQL -- SQLAlchemy builds the real
tables from these definitions.
"""

from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    # Every table-class inherits from this shared base so SQLAlchemy can
    # track them all together and create them in one go.
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    # A surrogate primary key -- an auto-incrementing id for every row.
    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # The text that was sent in for classification.
    text: Mapped[str] = mapped_column(String, nullable=False)

    # The model's answer -- the three fields of my MODEL contract.
    label: Mapped[str] = mapped_column(String, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    sentiment: Mapped[str] = mapped_column(String, nullable=False)

    # When this prediction was made. This timestamp is the seed of the whole
    # time-series story -- trends, anomalies and drift all read this column.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

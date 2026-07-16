"""FastAPI application -- the doorway into NewsVane.

Three read/write endpoints plus the pulse. POST /classify takes text and hands
back a prediction in the frozen MODEL shape, saving it on the way out. POST
/feedback stores a human's correction of one of those predictions. GET
/predictions reads them back over a window. GET /pulse serves the ANALYTICS
box's summarise() -- the radar's reading of the news over time.

Every endpoint only ever talks to the stable predict(), repository, and
summarise() contracts, so swapping the model, the database engine, or a maths
engine changes nothing here.
"""

from datetime import UTC, datetime, timedelta

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from newsvane.analytics.summary import summarise
from newsvane.models import predict
from newsvane.storage.repository import fetch, save, save_feedback

app = FastAPI(title="NewsVane", version="0.2.0")


class ClassifyRequest(BaseModel):
    # The only thing a caller must send me: the text to classify.
    text: str = Field(min_length=1)


class ClassifyResponse(BaseModel):
    # My MODEL contract, plus the row id. The id is what makes the prediction
    # addressable -- without it a human has no way to say "that one was wrong".
    id: int
    label: str
    score: float
    sentiment: str


class FeedbackRequest(BaseModel):
    prediction_id: int
    correct_label: str = Field(min_length=1)


class FeedbackResponse(BaseModel):
    id: int
    prediction_id: int
    correct_label: str


@app.post("/classify")
def classify(request: ClassifyRequest) -> ClassifyResponse:
    # I predict, then persist. Saving here -- not inside the model -- keeps
    # each box single-purpose: the model only thinks, the repository only
    # remembers, and this endpoint is the only place that knows about both.
    prediction = predict(request.text)
    row = save(request.text, prediction)
    return ClassifyResponse(
        id=row.id,
        label=row.label,
        score=row.score,
        sentiment=row.sentiment,
    )


@app.post("/feedback")
def feedback(request: FeedbackRequest) -> FeedbackResponse:
    row = save_feedback(request.prediction_id, request.correct_label)
    if row is None:
        # Either no such prediction, or it was already corrected. Both are the
        # caller's mistake, not mine -- 409, not 500.
        raise HTTPException(
            status_code=409,
            detail="Unknown prediction_id, or feedback already recorded for it.",
        )
    return FeedbackResponse(
        id=row.id,
        prediction_id=row.prediction_id,
        correct_label=row.correct_label,
    )


class PredictionOut(BaseModel):
    # Pydantic reads the ORM row's attributes directly instead of me
    # hand-copying six fields into a dict.
    model_config = ConfigDict(from_attributes=True)

    id: int
    text: str
    label: str
    score: float
    sentiment: str
    created_at: datetime


@app.get("/predictions")
def read_predictions(
    start: datetime | None = None,
    end: datetime | None = None,
    label: str | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[PredictionOut]:
    # A caller who names no window means "the last 24 hours" -- an unbounded
    # default would let one request drag the whole table over the wire.
    now = datetime.now(UTC)
    start = start or now - timedelta(days=1)
    end = end or now
    rows = fetch(start=start, end=end, label=label, limit=limit)
    return [PredictionOut.model_validate(row) for row in rows]


class TrendPoint(BaseModel):
    # One topic's article-count on one day -- the atom of a momentum series.
    day: datetime
    count: int


class DistributionOut(BaseModel):
    # Today's topic-mix, the recent norm, and the distance between the two.
    # today/norm map each topic name to its share of the day, so a plain dict.
    day: datetime
    distance: float
    today: dict[str, float]
    norm: dict[str, float]


class AnomalyOut(BaseModel):
    # One topic whose last-day volume broke far past its own recent normal.
    topic: str
    day: datetime
    count: int
    baseline: float
    z_score: float


class PulseResponse(BaseModel):
    # The frozen ANALYTICS contract, now typed so /docs describes every key.
    # drift is None until Phase 6 -- the shape is honoured today regardless.
    trends: dict[str, list[TrendPoint]]
    distribution: DistributionOut
    anomalies: list[AnomalyOut]
    drift: None = None


@app.get("/pulse")
def read_pulse(
    start: datetime | None = None,
    end: datetime | None = None,
) -> PulseResponse:
    # The maths needs several days of history to say anything, so an unnamed
    # window is the last week -- not the last day like /predictions, where a
    # single row is still a sensible answer.
    now = datetime.now(UTC)
    start = start or now - timedelta(days=7)
    end = end or now
    return summarise(start, end)
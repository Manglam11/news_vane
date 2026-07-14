"""FastAPI application -- the doorway into NewsVane.

Two endpoints so far. POST /classify takes text and hands back a prediction
in the frozen MODEL shape, saving it on the way out. POST /feedback takes a
human's correction of one of those predictions and stores it.

Every endpoint only ever talks to the stable predict() and repository
contracts, so swapping the model or the database engine changes nothing here.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from newsvane.models import predict
from newsvane.storage.repository import save, save_feedback

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
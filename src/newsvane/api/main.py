"""FastAPI application for the walking skeleton.

This box is the doorway into NewsVane. For now it exposes one endpoint --
POST /classify -- which takes a piece of text and hands back a prediction
in the frozen MODEL shape {label, score, sentiment}.

The endpoint only ever talks to the stable predict() contract, so when I
swap the dummy model for the real baseline in Phase 1, nothing here changes.
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from newsvane.models import predict
from newsvane.storage.repository import save

app = FastAPI(title="NewsVane", version="0.1.0")


class ClassifyRequest(BaseModel):
    # The only thing a caller must send me: the text to classify.
    text: str = Field(min_length=1)


class ClassifyResponse(BaseModel):
    # The exact shape my MODEL contract promises back.
    label: str
    score: float
    sentiment: str


@app.post("/classify")
def classify(request: ClassifyRequest) -> ClassifyResponse:
    # I predict, then persist. Saving here -- not inside the model -- keeps
    # each box single-purpose: the model only thinks, the repository only
    # remembers, and this endpoint is the only place that knows about both.
    prediction = predict(request.text)
    save(request.text, prediction)
    return ClassifyResponse(**prediction)
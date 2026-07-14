import pytest
from config.settings import settings

from newsvane.models import predict

# The MODEL box now loads a trained artefact off disk, and that file is a build
# output, not source -- it does not exist in a fresh clone or in CI until training
# has run. I skip rather than fail, because a missing model is a "not built yet"
# condition, not a broken contract.
needs_trained_model = pytest.mark.skipif(
    not settings.baseline_model_path.exists(),
    reason="no trained baseline on disk -- run `uv run python -m scripts.train`",
)


@needs_trained_model
def test_predict_returns_the_contract_shape():
    # The MODEL box promises {label, score, sentiment}. Everything downstream
    # is built on that promise, so I lock it down here.
    result = predict("Some ordinary news text.")

    assert set(result.keys()) == {"label", "score", "sentiment"}


@needs_trained_model
def test_predict_returns_sane_types_and_ranges():
    result = predict("Another headline about the economy.")

    assert isinstance(result["label"], str)
    assert isinstance(result["sentiment"], str)
    assert isinstance(result["score"], float)
    assert 0.0 <= result["score"] <= 1.0


@needs_trained_model
def test_predict_puts_obvious_sports_text_in_sports():
    # A shape test passes even if the model is nonsense, so I also check that it
    # gets one blatantly easy case right. This is a smoke test, not an evaluation
    # -- the honest scoring lives in the training script and in MLflow.
    result = predict("Barcelona beat Real Madrid 3-1 in a stunning comeback at the Nou Camp.")

    assert result["label"] == "Sports"
    assert result["score"] > 0.5

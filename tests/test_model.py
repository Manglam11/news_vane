from newsvane.models import predict


def test_predict_returns_the_contract_shape():
    # The MODEL box promises {label, score, sentiment}. Everything downstream
    # is built on that promise, so I lock it down here.
    result = predict("Some ordinary news text.")

    assert set(result.keys()) == {"label", "score", "sentiment"}


def test_predict_returns_sane_types_and_ranges():
    result = predict("Another headline about the economy.")

    assert isinstance(result["label"], str)
    assert isinstance(result["sentiment"], str)
    assert isinstance(result["score"], float)
    assert 0.0 <= result["score"] <= 1.0

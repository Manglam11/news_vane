"""Train the TF-IDF + Naive Bayes baseline and report an honest score.

This is the bar every future model has to beat. It pulls text through the DATA
box -- never re-implementing the cleaning -- fits a pipeline, scores it on the
held-out test split, and saves the fitted pipeline to disk so the MODEL box can
load it at serve time. One implementation of the transform, so training and
serving can never drift apart.
"""

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from config.settings import settings
from newsvane.data import get_articles


def load_split(split: str) -> tuple[list[str], list[str]]:
    """Pull one split through the DATA box and unzip it into X and y."""
    articles = get_articles(split)
    texts = [article["text"] for article in articles]
    topics = [article["topic"] for article in articles]
    return texts, topics


def build_pipeline() -> Pipeline:
    """Wire the vectoriser and the classifier into a single fitted-together unit.

    Keeping them in one Pipeline object is the whole trick: the exact vocabulary
    and IDF weights learned during training travel with the classifier into the
    saved artefact. At serve time I load one object and call predict on raw text.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=settings.tfidf_max_features,
                    ngram_range=(1, settings.tfidf_ngram_max),
                    min_df=settings.tfidf_min_df,
                    stop_words="english",
                    sublinear_tf=True,
                ),
            ),
            ("nb", MultinomialNB(alpha=settings.naive_bayes_alpha)),
        ]
    )


def main() -> None:
    x_train, y_train = load_split("train")
    x_test, y_test = load_split("test")
    print(f"train: {len(x_train)} rows | test: {len(x_test)} rows")

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\nmacro F1    : {macro_f1:.4f}")
    print(f"weighted F1 : {weighted_f1:.4f}")

    print("\nper-class report")
    print(classification_report(y_test, y_pred, digits=4))

    # I print the labels alongside the matrix because a bare grid of numbers is
    # unreadable -- I need to know which row is which class to spot the confusions.
    labels = sorted(set(y_test))
    print("confusion matrix (rows = true, cols = predicted)")
    print(f"labels: {labels}")
    print(confusion_matrix(y_test, y_pred, labels=labels))

    settings.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, settings.baseline_model_path)
    print(f"\nsaved -> {settings.baseline_model_path}")


if __name__ == "__main__":
    main()
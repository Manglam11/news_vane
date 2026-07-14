"""Train the TF-IDF + Naive Bayes baseline and report an honest score.

This is the bar every future model has to beat. It pulls text through the DATA
box -- never re-implementing the cleaning -- fits a pipeline, scores it on the
held-out test split, and saves the fitted pipeline to disk so the MODEL box can
load it at serve time. One implementation of the transform, so training and
serving can never drift apart.

Every run is recorded in MLflow: its settings, its scores, its confusion matrix
and the model itself. A number that only ever lived in my terminal is a number
I have already lost.
"""

import json
import tempfile
from pathlib import Path

import joblib
import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from config.settings import settings
from newsvane.data import get_articles


def setup_tracking() -> None:
    """Point MLflow at the SQLite backend and make sure the experiment exists.

    I create the experiment explicitly the first time so I can pin where its
    artefacts land. Left to itself MLflow would drop them relative to whatever
    directory I happened to launch from -- the same working-directory trap that
    already bit me once.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    settings.mlflow_artifacts_dir.mkdir(parents=True, exist_ok=True)

    if mlflow.get_experiment_by_name(settings.mlflow_experiment) is None:
        mlflow.create_experiment(
            settings.mlflow_experiment,
            artifact_location=settings.mlflow_artifact_uri,
        )
    mlflow.set_experiment(settings.mlflow_experiment)


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


def log_evaluation(y_true: list[str], y_pred: list[str], labels: list[str]) -> None:
    """Write the confusion matrix and per-class report to MLflow as a text artefact.

    A metric has to be a single number, so a 4x4 grid cannot be one. Anything
    shaped like a table, a plot or a file goes in as an artefact instead.
    """
    lines = [
        "confusion matrix (rows = true, cols = predicted)",
        f"labels: {labels}",
        str(confusion_matrix(y_true, y_pred, labels=labels)),
        "",
        "per-class report",
        classification_report(y_true, y_pred, digits=4),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "evaluation.txt"
        path.write_text("\n".join(lines), encoding="utf-8")
        mlflow.log_artifact(str(path))


def main() -> None:
    setup_tracking()

    x_train, y_train = load_split("train")
    x_test, y_test = load_split("test")
    print(f"train: {len(x_train)} rows | test: {len(x_test)} rows")

    with mlflow.start_run(run_name="tfidf-naive-bayes"):
        # Params are the knobs I turned. Metrics are what came out. Keeping them
        # apart is what makes two runs comparable months from now.
        mlflow.log_params(
            {
                "model": "TfidfVectorizer + MultinomialNB",
                "tfidf_max_features": settings.tfidf_max_features,
                "tfidf_ngram_range": f"(1, {settings.tfidf_ngram_max})",
                "tfidf_min_df": settings.tfidf_min_df,
                "tfidf_stop_words": "english",
                "tfidf_sublinear_tf": True,
                "naive_bayes_alpha": settings.naive_bayes_alpha,
                "train_rows": len(x_train),
                "test_rows": len(x_test),
            }
        )

        pipeline = build_pipeline()
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)

        macro_f1 = f1_score(y_test, y_pred, average="macro")
        weighted_f1 = f1_score(y_test, y_pred, average="weighted")

        # I log the per-class F1 too, because a single headline number can hide a
        # class that is quietly failing.
        labels = sorted(set(y_test))
        per_class = f1_score(y_test, y_pred, average=None, labels=labels)
        class_metrics = {
            f"f1_{label.replace('/', '_').lower()}": score
            for label, score in zip(labels, per_class, strict=True)
        }

        mlflow.log_metrics({"macro_f1": macro_f1, "weighted_f1": weighted_f1, **class_metrics})
        log_evaluation(y_test, y_pred, labels)

        print(f"\nmacro F1    : {macro_f1:.4f}")
        print(f"weighted F1 : {weighted_f1:.4f}")
        print("\nper-class report")
        print(classification_report(y_test, y_pred, digits=4))
        print("confusion matrix (rows = true, cols = predicted)")
        print(f"labels: {labels}")
        print(confusion_matrix(y_test, y_pred, labels=labels))

        settings.models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, settings.baseline_model_path)

        # The MODEL box loads the file on disk, so the class order it maps scores to
        # has to be recorded next to it -- never re-derived and hoped to match.
        labels_path = settings.models_dir / "baseline_labels.json"
        labels_path.write_text(json.dumps(list(pipeline.classes_)), encoding="utf-8")

        # The same artefacts go into MLflow too, so the run stands on its own without
        # depending on whatever happens to be sitting in models/ months from now.
        mlflow.log_artifact(str(settings.baseline_model_path))
        mlflow.log_artifact(str(labels_path))

        print(f"\nsaved -> {settings.baseline_model_path}")
        print(f"mlflow run -> {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
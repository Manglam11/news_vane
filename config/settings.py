"""All tunable values for NewsVane, loaded from the environment.

I keep every setting here instead of hardcoding it, and I read secrets like
the database URL from a local .env file (which git ignores). Changing
behaviour means editing a setting, never touching code.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # I anchor the .env to the project root rather than the working directory, so the API,
    # pytest and a notebook running from notebooks/ all read the exact same file.
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        extra="ignore",
    )

    database_url: str

    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    train_file: str = "train.csv"
    test_file: str = "test.csv"

    # Where the trained baseline lands. Training writes it, the MODEL box reads it.
    models_dir: Path = PROJECT_ROOT / "models"
    baseline_model_file: str = "baseline_tfidf_nb.joblib"

    # Baseline hyper-parameters. I keep them here so a retrain is a config edit, not a code edit.
    tfidf_max_features: int = 50_000
    tfidf_ngram_max: int = 2
    tfidf_min_df: int = 2
    naive_bayes_alpha: float = 0.1

    # MLflow keeps run metadata in a SQLite file and the heavier artefacts on disk.
    # The old file-only backend is deprecated as of MLflow 3.14, so I go straight to a
    # database backend -- swapping this URI for a real server later is a config change,
    # not a code change.
    mlflow_db_file: Path = PROJECT_ROOT / "mlflow.db"
    mlflow_artifacts_dir: Path = PROJECT_ROOT / "mlruns"
    mlflow_experiment: str = "newsvane-baseline"

    # --- SCRAPER (live DATA source) ---
    # The key is MY label; the URL is where I go to get it. The section I ask for IS
    # the label -- and these four keys must stay exactly the four classes the model
    # was trained on. A fifth key here would produce a label the model cannot predict.
    scraper_sections: dict[str, str] = {
        "World": "https://www.thehindu.com/news/international/",
        "Sports": "https://www.thehindu.com/sport/",
        "Business": "https://www.thehindu.com/business/",
        "Sci/Tech": "https://www.thehindu.com/sci-tech/",
    }

    # A default python user-agent gets refused by most news sites. I identify honestly
    # as a browser rather than pretending to be invisible.
    scraper_user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )

    scraper_limit: int = 15  # articles per section per run
    scraper_timeout: float = 20.0  # seconds before I give up on one request
    scraper_delay: float = 1.0  # seconds between requests -- I am a guest on their server
    # --- ANALYTICS ---
    # How many standard deviations above a topic's own normal counts as a spike.
    # 3.0 is the textbook "rare event" line; I keep it here so tuning the radar's
    # sensitivity is a config edit, never a code edit.
    anomaly_z_threshold: float = 3.0

    # The drift yardstick: the topic-mix the model was trained on. AG News is a
    # balanced benchmark, so the training split is a clean 25% per class -- I derived
    # these once with probe_reference.py and froze them here rather than reading them
    # live, because data/raw/ is git-ignored and simply does not exist at serve or CI
    # time. Drift compares live news against this fixed shape.
    drift_reference: dict[str, float] = {
        "World": 0.25,
        "Sports": 0.25,
        "Business": 0.25,
        "Sci/Tech": 0.25,
    }

    # How far live news may drift from the training mix before I raise the alarm.
    # JS divergence is bounded [0, 1]; 0.1 is a deliberately loud early line for a
    # fresh radar -- like the z-threshold, tuning it is a config edit, never a code edit.
    drift_threshold: float = 0.1

    distilbert_checkpoint: str = "distilbert-base-uncased"
    distilbert_max_length: int = 128

    # DistilBERT fine-tuning knobs. A retrain is a config edit, never a code edit.
    distilbert_epochs: int = 3
    distilbert_train_batch_size: int = 16
    distilbert_eval_batch_size: int = 32
    distilbert_learning_rate: float = 5e-5
    distilbert_weight_decay: float = 0.01
    distilbert_warmup_ratio: float = 0.1
    distilbert_output_file: str = "distilbert"
    distilbert_fp16: bool = True

    # The serving artefacts. The fine-tuned PyTorch model is a TRAINING output; these
    # two are what actually ship. The float32 graph is an intermediate I keep for
    # comparison, and the int8 graph is the one the container serves -- it has to fit
    # inside a 512 MB free tier, which torch never could.
    distilbert_onnx_file: str = "distilbert.onnx"
    distilbert_onnx_int8_file: str = "distilbert.int8.onnx"

    @property
    def baseline_model_path(self) -> Path:
        return self.models_dir / self.baseline_model_file

    @property
    def mlflow_tracking_uri(self) -> str:
        return f"sqlite:///{self.mlflow_db_file.as_posix()}"

    @property
    def mlflow_artifact_uri(self) -> str:
        return self.mlflow_artifacts_dir.as_uri()

    @property
    def distilbert_output_dir(self) -> Path:
        return self.models_dir / self.distilbert_output_file

    @property
    def distilbert_onnx_path(self) -> Path:
        return self.models_dir / self.distilbert_onnx_file

    @property
    def distilbert_onnx_int8_path(self) -> Path:
        return self.models_dir / self.distilbert_onnx_int8_file


settings = Settings()

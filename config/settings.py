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

    scraper_limit: int = 15       # articles per section per run
    scraper_timeout: float = 20.0  # seconds before I give up on one request
    scraper_delay: float = 1.0     # seconds between requests -- I am a guest on their server
    # --- ANALYTICS ---
    # How many standard deviations above a topic's own normal counts as a spike.
    # 3.0 is the textbook "rare event" line; I keep it here so tuning the radar's
    # sensitivity is a config edit, never a code edit.
    anomaly_z_threshold: float = 3.0

    @property
    def baseline_model_path(self) -> Path:
        return self.models_dir / self.baseline_model_file

    @property
    def mlflow_tracking_uri(self) -> str:
        return f"sqlite:///{self.mlflow_db_file.as_posix()}"

    @property
    def mlflow_artifact_uri(self) -> str:
        return self.mlflow_artifacts_dir.as_uri()


settings = Settings()
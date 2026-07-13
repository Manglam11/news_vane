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

    @property
    def baseline_model_path(self) -> Path:
        return self.models_dir / self.baseline_model_file


settings = Settings()
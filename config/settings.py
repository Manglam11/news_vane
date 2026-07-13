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


settings = Settings()

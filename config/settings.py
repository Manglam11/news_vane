"""All tunable values for NewsVane, loaded from the environment.

I keep every setting here instead of hardcoding it, and I read secrets like
the database URL from a local .env file (which git ignores). Changing
behaviour means editing a setting, never touching code.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str


settings = Settings()

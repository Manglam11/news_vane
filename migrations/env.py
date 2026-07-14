"""Alembic's runtime wiring for NewsVane.

Two things I plug in here, and only two:
  1. the database URL -- pulled from my settings, never hardcoded in a .ini
  2. target_metadata -- the Base my ORM models hang off, so Alembic can
     compare my Python classes against the live database and write the diff.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from config.settings import settings
from newsvane.storage.models import Base

config = context.config

# The URL comes from .env via settings, so the secret never enters a file
# that git tracks. This overrides whatever alembic.ini says.
config.set_main_option("sqlalchemy.url", settings.database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# This is the "what my code says the schema should be" side of the comparison.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    # Generates SQL text without connecting -- useful when a DBA must review
    # the statements before they touch production.
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    # The normal path: open a real connection and apply the migrations.
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Without this, Alembic ignores a column whose type I changed.
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
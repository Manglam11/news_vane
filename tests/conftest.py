"""Shared test fixtures -- the machinery every test file leans on.

The big change here: tests now talk to a real, disposable Postgres instead of
a monkeypatched stub. A stub can prove my wiring is right, but it can never
prove a unique constraint, a foreign key or a server default actually fires --
those only exist inside the database. So the database is now under test too.
"""

import os

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://newsvane:newsvane@localhost:5434/newsvane_test",
)


@pytest.fixture(scope="session")
def engine():
    # One engine for the whole run. I create the schema from my ORM models
    # rather than from Alembic: the migrations are tested separately, and a
    # test suite should never depend on a chain of them replaying correctly.
    from newsvane.storage.models import Base

    engine = create_engine(TEST_DATABASE_URL)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture(autouse=True)
def db(engine, monkeypatch):
    # Every test gets an empty database. I redirect the repository's session
    # factory at the test engine, so save() and fetch() write here and never
    # touch my real predictions.
    from newsvane.storage import repository
    from newsvane.storage.models import Base

    for table in reversed(Base.metadata.sorted_tables):
        with engine.begin() as conn:
            conn.execute(table.delete())

    monkeypatch.setattr(repository, "SessionLocal", sessionmaker(bind=engine))
    yield

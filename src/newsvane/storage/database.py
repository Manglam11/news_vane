"""Database connection setup for the STORAGE box.

I build one SQLAlchemy engine (a managed pool of connections to Postgres)
and one session factory. The connection string never lives here -- it comes
from config, which reads it from my local .env, so no secret is hardcoded.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config.settings import settings

# The engine manages a reusable pool of live connections to Postgres.
# I build it once and share it for the whole app's lifetime.
engine = create_engine(settings.database_url)

# SessionLocal is a factory: calling it gives me a fresh session -- one short
# "conversation" with the database that I open, use, and close per request.
SessionLocal = sessionmaker(bind=engine)
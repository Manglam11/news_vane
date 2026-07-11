# I expose predict() from the models package itself, so every caller
# imports it from one stable place no matter which file implements it.
# Today that's dummy.py; in Phase 1 it becomes the real baseline -- and
# only this single line changes, never the code that imports it.
from newsvane.models.dummy import predict

__all__ = ["predict"]
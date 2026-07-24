"""The pulse radar -- the one box a human actually looks at.

This reads the API and nothing else. It never imports the model, never opens a
database connection, and deliberately never imports config.settings: that module
demands a DATABASE_URL at import time, and a page that needs a production
database secret to draw a chart has broken the contract it was given.

This file is the running order, nothing more. One fetch, then each panel is
handed the slice of the reading it owns.
"""

import sys
from pathlib import Path

# The deploy host runs this file as a loose script and never installs the package,
# so src/ is not on the path and `import newsvane...` would fail there while
# working perfectly here. The entrypoint is the one honest place to fix that: it
# is the only file that knows where it sits on disk. Nothing below it needs this.
SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import httpx  # noqa: E402
import streamlit as st  # noqa: E402

from newsvane.dashboard.client import API_URL, fetch_pulse  # noqa: E402
from newsvane.dashboard.panels import alarms, mix, momentum, mood  # noqa: E402

st.set_page_config(page_title="NewsVane", page_icon="📡", layout="wide")

st.title("NewsVane 📡")
st.caption("What the news as a whole is doing -- not what one article says.")

try:
    pulse = fetch_pulse()
except httpx.HTTPError as error:
    # A dashboard that draws an empty chart from a dead source is lying about
    # the news being quiet. I say the source is down instead.
    st.error(f"Could not reach the API at {API_URL}")
    st.exception(error)
    st.stop()

alarms.render(pulse)
momentum.render(pulse["trends"])
mood.render(pulse["trends"])
mix.render(pulse["distribution"])

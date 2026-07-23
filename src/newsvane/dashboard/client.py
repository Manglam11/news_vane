"""The only thing in this box that touches the network.

Every panel gets its numbers from here, so there is exactly one place that knows
the API's address, its timeout, and how long a reading stays warm.
"""

import os

import httpx
import streamlit as st

# Config, not code -- the same law as settings.py, but this box ships to a
# different host, so its config surface is the environment, not my .env.
API_URL = os.environ.get("NEWSVANE_API_URL", "https://news-vane.onrender.com")

# Render's free tier sleeps after 15 minutes idle and takes about fifty seconds
# to wake. A ten-second default would report my own service as dead every time I
# open this page first thing in the morning.
API_TIMEOUT = 90.0


@st.cache_data(ttl=300)
def fetch_pulse() -> dict:
    """One call, held for five minutes.

    The radar moves once a day. Re-fetching on every widget click would only
    wake a sleeping server to hand back the exact same numbers.
    """
    response = httpx.get(f"{API_URL}/pulse", timeout=API_TIMEOUT)
    response.raise_for_status()
    return response.json()

"""The momentum panel: which topics are rising or fading, day by day."""

import plotly.express as px
import streamlit as st

from newsvane.dashboard.shaping import momentum_frame


def render(trends: dict) -> None:
    st.subheader("Topic momentum")
    st.caption(
        "Articles harvested per day. Each section has a per-run cap, so a flat top is "
        "my own collection ceiling -- not the news going quiet."
    )

    frame = momentum_frame(trends)
    if frame.empty:
        st.info("No articles in this window yet.")
        return

    figure = px.line(frame, x="day", y="count", color="topic", markers=True)
    # One tick per day, because there is exactly one point per day. Hourly ticks
    # advertise a resolution this data does not have.
    figure.update_xaxes(title_text="", dtick="D1", tickformat="%b %d")
    figure.update_yaxes(title_text="articles harvested", rangemode="tozero")
    figure.update_layout(margin={"t": 10, "b": 0, "l": 0, "r": 0}, legend_title_text="")
    st.plotly_chart(figure, use_container_width=True)

"""The mix panel: what shape today's news has, against the shape it usually has."""

import plotly.express as px
import streamlit as st

from newsvane.dashboard.shaping import mix_frame


def render(distribution: dict) -> None:
    st.subheader("Today's topic mix")
    st.caption(
        "The share of today's articles each topic holds, beside the norm for the days "
        "before it. The gap between a pair of bars is what 'unusual' actually looks like."
    )

    frame = mix_frame(distribution)
    if frame.empty:
        st.info("Nothing harvested today yet.")
        return

    figure = px.bar(frame, x="topic", y="share", color="when", barmode="group")
    figure.update_xaxes(title_text="")
    # Shares arrive as fractions. I format the axis rather than multiplying the
    # data by 100, so the number the API sent is the number the chart holds.
    figure.update_yaxes(title_text="", tickformat=".0%", rangemode="tozero")
    figure.update_layout(margin={"t": 10, "b": 0, "l": 0, "r": 0}, legend_title_text="")
    st.plotly_chart(figure, use_container_width=True)

    st.caption(f"Distance from the norm: {distribution['distance']:.4f} — 0 means identical.")

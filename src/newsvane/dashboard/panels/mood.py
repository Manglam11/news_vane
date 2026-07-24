"""The mood panel: how each topic's tone is moving, day by day."""

import plotly.express as px
import streamlit as st

from newsvane.dashboard.shaping import MIN_FOR_MOOD, mood_frame


def render(trends: dict) -> None:
    st.subheader("Topic mood")
    st.caption(
        "Average tone of the articles read each day, from -1 (bleak) to +1 (upbeat). "
        "A line that dips is a day of darker coverage -- not a day of less coverage. "
        "A break in a line is a day whose articles were never read for mood."
    )

    frame = mood_frame(trends)
    if frame.empty:
        st.info("No article in this window has been read for mood yet.")
        return

    figure = px.line(
        frame,
        x="day",
        y="mood",
        color="topic",
        markers=True,
        hover_data={"count": True, "thin": False},
    )

    thin = frame[frame["thin"]]
    if not thin.empty:
        # A hollow ring over any point averaged from too few articles. The number
        # is real and I draw it, but a reader has to be able to see which points
        # are one story's score rather than a day's mood.
        figure.add_scatter(
            x=thin["day"],
            y=thin["mood"],
            mode="markers",
            marker={"size": 14, "symbol": "circle-open", "line": {"width": 2}},
            marker_color="grey",
            name=f"under {MIN_FOR_MOOD} articles",
            hoverinfo="skip",
        )

    # Zero is the only line on this chart that means something on its own: it is
    # the boundary between negative and positive coverage, not a chosen threshold.
    figure.add_hline(y=0, line_width=1, line_dash="dot", opacity=0.4)
    figure.update_xaxes(title_text="", dtick="D1", tickformat="%b %d")
    # Pinned to the scale the score actually lives on. Letting plotly fit the axis
    # to the data would blow a quiet 0.05 swing up into a dramatic climb.
    figure.update_yaxes(title_text="mood", range=[-1, 1], zeroline=False)
    figure.update_layout(margin={"t": 10, "b": 0, "l": 0, "r": 0}, legend_title_text="")
    st.plotly_chart(figure, use_container_width=True)

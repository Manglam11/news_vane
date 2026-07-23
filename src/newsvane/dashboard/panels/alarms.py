"""The alarm strip: is the model still trustworthy, and did anything break out?"""

import streamlit as st

from newsvane.dashboard.shaping import anomaly_frame, drift_verdict

# The verdict word decides the face. The words come from the pure module so the
# judgement can be tested; only the icon lives here.
VERDICT_ICONS = {
    "steady": "🟢",
    "drifting": "🔴",
    "too few marked": "🟡",
    "no reading": "⚪",
}


def render(pulse: dict) -> None:
    drift = pulse["drift"]
    anomalies = pulse["anomalies"]

    col_agreement, col_drift, col_anomalies = st.columns(3)

    # A rate without its denominator is a number I cannot judge. 13/15 and 866/1000
    # both read as "86.7%", and only one of them means anything -- so the denominator
    # sits on the face of the card, not hidden in a tooltip.
    if drift is None:
        col_agreement.metric("Model agreement", "no data")
    else:
        exam = drift["agreement"]
        rate = "n/a" if exam["rate"] is None else f"{exam['rate']:.1%}"
        col_agreement.metric(
            "Model agreement",
            rate,
            delta=f"{exam['agreed']} of {exam['scored']} marked",
            delta_color="off",
        )

    verdict, why = drift_verdict(drift)
    col_drift.metric(
        "Model drift", f"{VERDICT_ICONS[verdict]} {verdict}", delta=why, delta_color="off"
    )

    col_anomalies.metric("Breakouts today", len(anomalies))

    frame = anomaly_frame(anomalies)
    if frame.empty:
        # Nothing unusual is the answer most days, and it is worth saying out loud.
        # A blank space reads like a panel that failed to load.
        st.caption("🟢 No topic broke out of its normal range today.")
    else:
        st.dataframe(frame, use_container_width=True, hide_index=True)

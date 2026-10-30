# plot 1
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# 1. Load Summary Data & Aggregate to Subject Averages
DATA = pd.read_csv("./data/BDS/output/BDS_trial_summary.csv")
DATA["AgeGroup"] = pd.Categorical(
    DATA["AgeGroup"], categories=["Young", "Old"], ordered=True
)

meta_cols = ["Subject", "Vision", "Surface", "AgeGroup", "Gender"]
AVE_DATA = DATA.groupby(meta_cols, as_index=False, observed=True).mean(
    numeric_only=True
)
AVE_DATA["Condition"] = (
    AVE_DATA["Surface"].astype(str) + ":" + AVE_DATA["Vision"].astype(str)
)

cond_order = ["Firm:Open", "Firm:Closed", "Foam:Open", "Foam:Closed"]
cbPalette = ["#1E6654", "#CC5500", "#954535", "#5B7C99"]
cond_colors = {cond: cbPalette[i] for i, cond in enumerate(cond_order)}

# 2. Build Base Scatter Plot
fig1 = px.scatter(
    AVE_DATA,
    x="Condition",
    y="PathLength_Total",
    color="Condition",
    symbol="Condition",
    facet_col="AgeGroup",
    category_orders={"AgeGroup": ["Young", "Old"], "Condition": cond_order},
    color_discrete_map=cond_colors,
    custom_data=["Subject", "Age", "Gender", "PathLength_Total"],
    labels={"PathLength_Total": "Total Path Length (cm)", "Condition": "Condition"},
    title="<b>Postural Sway: Between- and Within-Subject Effects</b><br><sup>Hover points to inspect subject metadata; lines connect individual trajectories</sup>",
)

# 3. Add Within-Subject Trajectory Lines Across Subplots
for i, age_grp in enumerate(["Young", "Old"], start=1):
    sub_df = AVE_DATA[AVE_DATA["AgeGroup"] == age_grp]
    for sub_id, grp in sub_df.groupby("Subject"):
        grp_sorted = (
            grp.set_index("Condition").reindex(cond_order).dropna().reset_index()
        )
        fig1.add_trace(
            go.Scatter(
                x=grp_sorted["Condition"],
                y=grp_sorted["PathLength_Total"],
                mode="lines",
                line=dict(color="rgba(180, 180, 180, 0.5)", width=1),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=i,
        )

# Move trajectory lines behind scatter points
n_conditions = len(cond_order)
fig1.data = tuple(list(fig1.data[n_conditions:]) + list(fig1.data[:n_conditions]))

# 4. Format Custom Hover Tooltip & Layout
fig1.update_traces(
    selector=dict(mode="markers"),
    marker=dict(size=8, opacity=0.8),
    hovertemplate=(
        "<b>Subject ID:</b> %{customdata[0]}<br>"
        "<b>Age:</b> %{customdata[1]:.1f} years<br>"
        "<b>Gender:</b> %{customdata[2]}<br>"
        "<b>Sway (Path Length):</b> %{customdata[3]:.1f} cm<extra></extra>"
    ),
)

fig1.update_layout(
    template="plotly_white",
    legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
    height=500,
)

# Single final expression evaluated
fig1

# plot2

import pandas as pd
import numpy as np
import plotly.express as px

# 1. Prepare QA Data Status Formatting
QA_data = DATA.copy()
QA_data["Status"] = np.where(QA_data["flag_any"], "Flagged", "Clean")
QA_data["Flag_Symbol"] = np.where(QA_data["flag_any"], "⚠️ YES", "✅ No")

# 2. Build Interactive Scatter Plot
fig2 = px.scatter(
    QA_data,
    x="Weight",
    y="Fz_median",
    color="Status",
    facet_col="AgeGroup",
    category_orders={"AgeGroup": ["Young", "Old"], "Status": ["Clean", "Flagged"]},
    color_discrete_map={"Clean": "#1E6654", "Flagged": "#CC5500"},
    custom_data=[
        "Subject",
        "AgeGroup",
        "Gender",
        "Trial",
        "Surface",
        "Vision",
        "Weight",
        "Fz_median",
        "Fz_min",
        "duration_s",
        "n_samples",
        "Flag_Symbol",
    ],
    labels={
        "Weight": "Recorded Weight (kg)",
        "Fz_median": "Median Vertical Force Fz (N)",
        "Status": "Data Status",
    },
    title="<b>QA: Force Plate Calibration & Unweighting Verification</b><br><sup>Dashed line represents theoretical gravity (Fz = Mass * 9.81). Hover points to inspect anomalies.</sup>",
)

# 3. Add Gravity Reference Lines (Assigned to _ so reticulate ignores it)
for col_idx in [1, 2]:
    _ = fig2.add_shape(
        type="line",
        x0=35,
        y0=35 * 9.81,
        x1=110,
        y1=110 * 9.81,
        line=dict(color="gray", dash="dash"),
        row=1,
        col=col_idx,
    )

# 4. Format Hover Tooltip & Styling (Assigned to _ so reticulate ignores it)
_ = fig2.update_traces(
    marker=dict(size=8, opacity=0.7),
    hovertemplate=(
        "<b>Subject:</b> %{customdata[0]} (%{customdata[1]}, %{customdata[2]})<br>"
        "<b>Trial:</b> %{customdata[3]}<br>"
        "<b>Condition:</b> %{customdata[4]} - %{customdata[5]}<br>"
        "<b>Recorded Weight:</b> %{customdata[6]:.1f} kg<br>"
        "<b>Median Fz:</b> %{customdata[7]:.1f} N<br>"
        "<b>Min Fz:</b> %{customdata[8]:.1f} N<br>"
        "<b>Duration:</b> %{customdata[9]}s (%{customdata[10]} samples)<br>"
        "<b>Flagged?:</b> %{customdata[11]}<extra></extra>"
    ),
)

_ = fig2.update_layout(
    template="plotly_white",
    legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
    height=500,
)

# Render single figure
fig2

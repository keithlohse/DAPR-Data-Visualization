import os
import numpy as np
import pandas as pd
from scipy.stats import f
import matplotlib.pyplot as plt
import seaborn as sns

master = pd.read_parquet("./data/BDS/output/BDS_master.parquet")

# 1. Feature Extraction & Ellipse Math
def compute_ellipse95(x, y):
    n = len(x)
    cov_mat = np.cov(x, y)
    eigvals, eigvecs = np.linalg.eigh(cov_mat)
    f_val = f.ppf(0.95, 2, n - 2)
    area = 2 * np.pi * f_val * np.sqrt(np.maximum(0, eigvals[0] * eigvals[1]))
    return area, eigvals, eigvecs, f_val, np.mean(x), np.mean(y)

def ellipse_boundary_df(x, y, n_points=100):
    area, eigvals, eigvecs, f_val, cx, cy = compute_ellipse95(x, y)
    axis_lengths = np.sqrt(2 * f_val * np.maximum(0, eigvals))
    theta = np.linspace(0, 2 * np.pi, n_points)
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    pts = eigvecs @ np.diag(axis_lengths) @ circle
    return pd.DataFrame({'ex': cx + pts[0, :], 'ey': cy + pts[1, :]})

def compute_trial_features(df):
    duration = df["Time"].max() - df["Time"].min()
    dx = np.diff(df["COPx"])
    dy = np.diff(df["COPy"])
    
    pl_ml  = np.sum(np.abs(dx))
    pl_ap  = np.sum(np.abs(dy))
    pl_tot = np.sum(np.sqrt(dx**2 + dy**2))
    
    area, _, _, _, _, _ = compute_ellipse95(df["COPx"], df["COPy"])
    time_diffs = np.diff(df["Time"])
    
    return pd.Series({
        "PathLength_AP": pl_ap, "PathLength_ML": pl_ml, "PathLength_Total": pl_tot,
        "MeanVelocity_AP": pl_ap / duration, "MeanVelocity_ML": pl_ml / duration,
        "EllipseArea95": area, "n_samples": len(df), "duration_s": duration,
        "Fz_min": df["Fz"].min(), "Fz_median": df["Fz"].median(),
        "time_gap_max": time_diffs.max(), "time_nonmonotonic": (time_diffs <= 0).any()
    })

meta_cols = ["Trial", "Subject", "Vision", "Surface", "Age", "AgeGroup", "Gender", "Height", "Weight", "BMI"]
trial_summary = master.groupby(meta_cols, observed=True).apply(compute_trial_features).reset_index()

# 2. QA Flags
modal_n = trial_summary["n_samples"].mode()[0]
trial_summary["flag_sample_count"]   = (trial_summary["n_samples"] - modal_n).abs() > 10
trial_summary["flag_time_gap"]       = (trial_summary["time_gap_max"] > 0.02) | trial_summary["time_nonmonotonic"]
trial_summary["flag_fz_unweighting"] = trial_summary["Fz_min"] < (0.7 * trial_summary["Fz_median"])
trial_summary["flag_any"]            = trial_summary["flag_sample_count"] | trial_summary["flag_time_gap"] | trial_summary["flag_fz_unweighting"]

qa_log = trial_summary[trial_summary["flag_any"]]
qa_log.to_csv("./data/BDS/output/qa_log.csv", index=False)
trial_summary.to_csv("./data/BDS/output/BDS_trial_summary.csv", index=False)

# 3. Overview Outlier-Screening Figure
overview_data = trial_summary.melt(
    id_vars=["Trial", "Subject", "Vision", "Surface"],
    value_vars=["PathLength_Total", "MeanVelocity_AP", "MeanVelocity_ML", "EllipseArea95"],
    var_name="Measure", value_name="Value"
)

g = sns.catplot(
    data=overview_data, x="Surface", y="Value", hue="Vision", col="Measure",
    col_wrap=2, kind="box", height=3.5, aspect=1.2, sharey=False
)
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Trial-level DV Distributions (Quality Assurance Overview)")
plt.show()

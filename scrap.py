import numpy as np
import pandas as pd
from scipy.stats import f
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

bds_folder = Path() / "data" / "BDS"
bds_output_folder = bds_folder / "output"
master_file = bds_output_folder / "BDS_master.parquet"
master_data = pd.read_parquet(master_file)


# 1. Feature Extraction & Ellipse Math
def compute_ellipse95(
    x: pd.DataFrame, y: pd.DataFrame
) -> tuple[np.float64, np.ndarray, np.ndarray, np.float64, np.float64, np.float64]:
    n_values = len(x)
    covariance_matrix = np.cov(x, y)
    eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
    f_val = f.ppf(0.95, 2, n_values - 2)
    area = 2 * np.pi * f_val * np.sqrt(np.maximum(0, eigen_values[0] * eigen_values[1]))
    x_mean, y_mean = np.mean(x), np.mean(y)
    return area, eigen_values, eigen_vectors, f_val, x_mean, y_mean


def ellipse_boundary_df(
    x: pd.DataFrame, y: pd.DataFrame, n_points: int = 100
) -> pd.DataFrame:
    _, eigen_values, eigen_vectors, f_val, avg_x, avg_y = compute_ellipse95(x, y)
    axis_lengths = np.sqrt(2 * f_val * np.maximum(0, eigen_values))
    theta = np.linspace(0, 2 * np.pi, n_points)
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    pts = eigen_vectors @ np.diag(axis_lengths) @ circle
    return pd.DataFrame({"ex": avg_x + pts[0, :], "ey": avg_y + pts[1, :]})


def compute_trial_features(df: pd.DataFrame) -> pd.Series:
    duration = df["Time"].max() - df["Time"].min()
    dx = np.diff(df["COPx"])
    dy = np.diff(df["COPy"])

    pl_ml = np.sum(np.abs(dx))
    pl_ap = np.sum(np.abs(dy))
    pl_tot = np.sum(np.sqrt(dx**2 + dy**2))

    area, _, _, _, _, _ = compute_ellipse95(df["COPx"], df["COPy"])
    time_diffs = np.diff(df["Time"])

    return pd.Series(
        {
            "PathLength_AP": pl_ap,
            "PathLength_ML": pl_ml,
            "PathLength_Total": pl_tot,
            "MeanVelocity_AP": pl_ap / duration,
            "MeanVelocity_ML": pl_ml / duration,
            "EllipseArea95": area,
            "n_samples": len(df),
            "duration_s": duration,
            "Fz_min": df["Fz"].min(),
            "Fz_median": df["Fz"].median(),
            "time_gap_max": time_diffs.max(),
            "time_nonmonotonic": (time_diffs <= 0).any(),
        }
    )


meta_columns = [
    "Trial",
    "Subject",
    "Vision",
    "Surface",
    "Age",
    "AgeGroup",
    "Gender",
    "Height",
    "Weight",
    "BMI",
]
trial_summary = (
    master_data.groupby(meta_columns, observed=True)
    .apply(compute_trial_features)
    .reset_index()
)

# 2. QA Flags
modal_n = trial_summary["n_samples"].mode()[0]
trial_summary["flag_sample_count"] = (trial_summary["n_samples"] - modal_n).abs() > 10
trial_summary["flag_time_gap"] = (trial_summary["time_gap_max"] > 0.02) | trial_summary[
    "time_nonmonotonic"
]
trial_summary["flag_fz_unweighting"] = trial_summary["Fz_min"] < (
    0.7 * trial_summary["Fz_median"]
)
trial_summary["flag_any"] = (
    trial_summary["flag_sample_count"]
    | trial_summary["flag_time_gap"]
    | trial_summary["flag_fz_unweighting"]
)

qa_log = trial_summary[trial_summary["flag_any"]]
qa_log.to_csv(bds_output_folder / "qa_log.csv", index=False)
trial_summary.to_csv(bds_output_folder / "BDS_trial_summary.csv", index=False)

# 3. Overview Outlier-Screening Figure
overview_data = trial_summary.melt(
    id_vars=["Trial", "Subject", "Vision", "Surface"],
    value_vars=[
        "PathLength_Total",
        "MeanVelocity_AP",
        "MeanVelocity_ML",
        "EllipseArea95",
    ],
    var_name="Measure",
    value_name="Value",
)

fig = plt.figure()
sns.catplot(
    data=overview_data,
    x="Surface",
    y="Value",
    hue="Vision",
    col="Measure",
    col_wrap=2,
    kind="box",
    height=3.5,
    aspect=1.2,
    sharey=False,
)
fig.subplots_adjust(top=0.9)
fig.suptitle("Trial-level DV Distributions (Quality Assurance Overview)")
plt.show()

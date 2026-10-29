# 01

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

# qaPrep

import os
import numpy as np
import pandas as pd
from scipy.stats import f
import matplotlib.pyplot as plt
from pathlib import Path

# --- one-time setup, shared by every subject we display below ---
output_folder = Path() / "data" / "BDS" / "output"
master = pd.read_parquet(output_folder / "BDS_master.parquet")


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


master_aug = master.sort_values(["Trial", "Time"]).copy()
master_aug["dx"] = master_aug.groupby("Trial")["COPx"].diff().fillna(0)
master_aug["dy"] = master_aug.groupby("Trial")["COPy"].diff().fillna(0)
master_aug["StepDist"] = np.sqrt(master_aug["dx"] ** 2 + master_aug["dy"] ** 2)
master_aug["CumulativePath"] = master_aug.groupby("Trial")["StepDist"].cumsum()

rep_df = (
    master_aug[["Trial", "Subject", "Vision", "Surface"]]
    .drop_duplicates()
    .sort_values(["Subject", "Vision", "Surface", "Trial"])
)
rep_df["Rep"] = rep_df.groupby(["Subject", "Vision", "Surface"]).cumcount() + 1
master_aug = master_aug.merge(rep_df[["Trial", "Rep"]], on="Trial")

# Split ONCE into a dict keyed by Subject. This replaces the old pattern
# of masking master_aug[master_aug["Subject"] == sid] inside a
# 163-iteration loop, which re-scanned the full 11M-row frame each time.
subject_data = {sid: df for sid, df in master_aug.groupby("Subject")}


PALETTE = {1: "#1E6654", 2: "#CC5500", 3: "#5B7C99"}
SURFACES = ["Firm", "Foam"]


def make_subject_qa_plot(id_value: int, df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex="row", sharey="row")

    for surface_idx, surface in enumerate(SURFACES):
        df_surf = df[df["Surface"] == surface]
        ax_top = axes[0, surface_idx]
        ax_bot = axes[1, surface_idx]

        is_firm_surface = surface == "Firm"

        for (_, rep, vis), df_tr in df_surf.groupby(
            ["Trial", "Rep", "Vision"], observed=True
        ):
            ls = "-" if vis == "Open" else "--"
            c = PALETTE[rep]

            ax_top.plot(
                df_tr["Time"], df_tr["CumulativePath"], linestyle=ls, color=c, alpha=0.8
            )

            ax_bot.plot(
                df_tr["COPx"],
                df_tr["COPy"],
                linestyle=ls,
                color=c,
                alpha=0.3,
                linewidth=0.8,
            )

            ell_df = ellipse_boundary_df(df_tr["COPx"], df_tr["COPy"])
            ax_bot.plot(
                ell_df["ex"],
                ell_df["ey"],
                linestyle=ls,
                color=c,
                alpha=0.9,
                linewidth=1.2,
            )

        ax_top.set_title(f"Surface: {surface}")
        ax_top.set_xlabel("Time (s)")

        ax_bot.set_title(f"Surface: {surface}")
        ax_bot.set_xlabel("COPx (cm)")

        if is_firm_surface:
            ax_top.set_ylabel("Cumulative path (cm)")
            ax_bot.set_ylabel("COPy (cm)")

        ax_bot.set_aspect("equal", adjustable="box")

    fig.suptitle(
        f"Subject {id_value} - QA Diagnostic\nTop: Cumulative Path | Bottom: Stabilogram & 95% Ellipses",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


qa_dir = output_folder / "qa_plots"
os.makedirs(qa_dir, exist_ok=True)

for sid, df_sub in subject_data.items():
    out_file = qa_dir / f"subject_{sid:03d}_QA.png"
    if not os.path.exists(out_file):
        fig = make_subject_qa_plot(sid, df_sub)
        fig.savefig(out_file, dpi=100)
        plt.close(fig)

# The two subjects we spotlight below: the first ("clean") and the last
# ("built-in errors") subject in the sorted subject list.
subject_ids_sorted = sorted(subject_data.keys())
good_subject_id = subject_ids_sorted[0]
bad_subject_id = subject_ids_sorted[162]

# good subject

make_subject_qa_plot(good_subject_id, subject_data[good_subject_id])
plt.show()

# bad subject

make_subject_qa_plot(bad_subject_id, subject_data[bad_subject_id])
plt.show()
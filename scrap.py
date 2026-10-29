import os
import numpy as np
import pandas as pd
from scipy.stats import f
import matplotlib.pyplot as plt

# --- one-time setup, shared by every subject we display below ---
master = pd.read_parquet("./data/BDS/output/BDS_master.parquet")

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

master_aug = master.sort_values(["Trial", "Time"]).copy()
master_aug["dx"] = master_aug.groupby("Trial")["COPx"].diff().fillna(0)
master_aug["dy"] = master_aug.groupby("Trial")["COPy"].diff().fillna(0)
master_aug["StepDist"] = np.sqrt(master_aug["dx"]**2 + master_aug["dy"]**2)
master_aug["CumulativePath"] = master_aug.groupby("Trial")["StepDist"].cumsum()

rep_df = master_aug[["Trial", "Subject", "Vision", "Surface"]].drop_duplicates().sort_values(["Subject", "Vision", "Surface", "Trial"])
rep_df["Rep"] = rep_df.groupby(["Subject", "Vision", "Surface"]).cumcount() + 1
master_aug = master_aug.merge(rep_df[["Trial", "Rep"]], on="Trial")

# Split ONCE into a dict keyed by Subject. This replaces the old pattern
# of masking master_aug[master_aug["Subject"] == sid] inside a
# 163-iteration loop, which re-scanned the full 11M-row frame each time.
subject_data = {sid: df for sid, df in master_aug.groupby("Subject")}

def make_subject_qa_plot(subject_id, df_sub):
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex='row', sharey='row')
    palette = {1: "#1E6654", 2: "#CC5500", 3: "#5B7C99"}
    surfaces = ["Firm", "Foam"]

    for j, surf in enumerate(surfaces):
        df_surf = df_sub[df_sub["Surface"] == surf]
        ax_top = axes[0, j]
        ax_bot = axes[1, j]

        for (tr, rep, vis), df_tr in df_surf.groupby(["Trial", "Rep", "Vision"], observed=True):
            ls = "-" if vis == "Open" else "--"
            c = palette[rep]
            ax_top.plot(df_tr["Time"], df_tr["CumulativePath"], linestyle=ls, color=c, alpha=0.8)

        ax_top.set_title(f"Surface: {surf}")
        ax_top.set_xlabel("Time (s)")
        if j == 0:
            ax_top.set_ylabel("Cumulative path (cm)")

        for (tr, rep, vis), df_tr in df_surf.groupby(["Trial", "Rep", "Vision"], observed=True):
            ls = "-" if vis == "Open" else "--"
            c = palette[rep]
            ax_bot.plot(df_tr["COPx"], df_tr["COPy"], linestyle=ls, color=c, alpha=0.3, linewidth=0.8)

            ell_df = ellipse_boundary_df(df_tr["COPx"], df_tr["COPy"])
            ax_bot.plot(ell_df["ex"], ell_df["ey"], linestyle=ls, color=c, alpha=0.9, linewidth=1.2)

        ax_bot.set_title(f"Surface: {surf}")
        ax_bot.set_xlabel("COPx (cm)")
        if j == 0:
            ax_bot.set_ylabel("COPy (cm)")
        ax_bot.set_aspect("equal", adjustable="box")

    fig.suptitle(f"Subject {subject_id} - QA Diagnostic\nTop: Cumulative Path | Bottom: Stabilogram & 95% Ellipses", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig

qa_dir = "./data/BDS/output/qa_plots"
os.makedirs(qa_dir, exist_ok=True)

for sid, df_sub in subject_data.items():
    out_file = f"{qa_dir}/subject_{sid:03d}_QA.png"
    if not os.path.exists(out_file):
        fig = make_subject_qa_plot(sid, df_sub)
        fig.savefig(out_file, dpi=100)
        plt.close(fig)

# The two subjects we spotlight below: the first ("clean") and the last
# ("built-in errors") subject in the sorted subject list.
subject_ids_sorted = sorted(subject_data.keys())
good_subject_id = subject_ids_sorted[0]
bad_subject_id  = subject_ids_sorted[162]
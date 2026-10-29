
## PY-PLOT1
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA = pd.read_csv("./data/BDS/output/BDS_trial_summary.csv")
DATA["AgeGroup"] = pd.Categorical(DATA["AgeGroup"], categories=["Young", "Old"], ordered=True)

# Step 1: Basic scatter plot
fig, ax = plt.subplots(figsize=(6, 4))
sns.scatterplot(data=DATA, x="Surface", y="PathLength_Total", color="black", ax=ax)
ax.set_ylabel("Total Path Length (cm)")
sns.despine()
plt.show()

# PY-PLOT2
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Dodged points by Vision condition
fig, ax = plt.subplots(figsize=(6, 4))
sns.stripplot(
    data=DATA, 
    x="Surface", 
    y="PathLength_Total",
    hue="Vision", 
    dodge=True, 
    jitter=0.2, 
    ax=ax
)
ax.set_ylabel("Total Path Length (cm)")
sns.despine()
plt.show()

# py-plot3
cbPalette = ["#1E6654", "#CC5500", "#954535", "#5B7C99", "#8366BD", "#D1C071", "#F0EAD6", "#333333"]

# Step 3: Boxplots with jitter-dodged points
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.boxplot(
    data=DATA, x="Surface", y="PathLength_Total", hue="Vision",
    palette=cbPalette[:2], boxprops=dict(alpha=0.5), showfliers=False, ax=ax
)
sns.stripplot(
    data=DATA, x="Surface", y="PathLength_Total", hue="Vision",
    dodge=True, jitter=0.2, palette=cbPalette[:2], alpha=0.5, ax=ax
)

# Deduplicate legend entries
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], title="Vision")
ax.set_ylabel("Total Path Length (cm)")
sns.despine()
plt.show()

# py-plot 4
AVE_DATA = DATA.groupby(["Subject", "Vision", "Surface", "AgeGroup"], as_index=False).mean(numeric_only=True)

# Step 4: Formatted subject-level summary
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(
    data=AVE_DATA, x="Surface", y="PathLength_Total", hue="Vision",
    palette=cbPalette[:2], boxprops=dict(alpha=0.5), showfliers=False, ax=ax
)
sns.stripplot(
    data=AVE_DATA, x="Surface", y="PathLength_Total", hue="Vision",
    dodge=True, jitter=0.2, palette=cbPalette[:2], alpha=0.5, ax=ax
)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], title="Vision Condition", loc="upper right")

ax.set_title("Postural Sway: Total Path Length by Surface and Vision\n", fontweight="bold", fontsize=13)
ax.set_xlabel("Surface Condition", fontweight="bold")
ax.set_ylabel("Total Path Length (cm)", fontweight="bold")
sns.despine()
plt.show()

# py-plot5

# Step 5: Within-subject trajectory lines faceted by Surface
vis_palette = {"Open": cbPalette[0], "Closed": cbPalette[1]}
g = sns.FacetGrid(AVE_DATA, col="Surface", height=4.5, aspect=1.1)

for surf_name, ax in g.axes_dict.items():
    sub = AVE_DATA[AVE_DATA["Surface"] == surf_name]
    
    # Layer 1: Subject lines
    sns.lineplot(
        data=sub, x="Vision", y="PathLength_Total", units="Subject", 
        estimator=None, color="gray", alpha=0.5, ax=ax
    )
    
    # Layer 2: Overlay scatter points
    sns.scatterplot(
        data=sub, x="Vision", y="PathLength_Total", hue="Vision", 
        style="Vision", palette=vis_palette, s=60, alpha=0.8, legend=False, ax=ax
    )
    
    # Layer 3: Boxplot
    sns.boxplot(
        data=sub, x="Vision", y="PathLength_Total", 
        palette=vis_palette, boxprops=dict(alpha=0.4), showfliers=False, ax=ax
    )
    
    ax.set_title(f"Surface: {surf_name}", fontweight="bold")
    ax.set_xlabel("Vision Condition", fontweight="bold")
    ax.set_ylabel("Total Path Length (cm)", fontweight="bold")

g.fig.subplots_adjust(top=0.8)
g.fig.suptitle("Postural Sway: Within-Subject Visual Effects", fontweight="bold", fontsize=14)
sns.despine()
plt.show()

# py-plot6

AVE_DATA["Condition"] = AVE_DATA["Surface"].astype(str) + ":" + AVE_DATA["Vision"].astype(str)
cond_order = ["Firm:Open", "Firm:Closed", "Foam:Open", "Foam:Closed"]
cond_palette = {cond: cbPalette[i] for i, cond in enumerate(cond_order)}

g = sns.FacetGrid(AVE_DATA, col="AgeGroup", height=4.5, aspect=1.2)

for age_group, ax in g.axes_dict.items():
    sub = AVE_DATA[AVE_DATA["AgeGroup"] == age_group]
    
    # Layer 1: Subject lines
    sns.lineplot(
        data=sub, x="Condition", y="PathLength_Total", units="Subject", 
        estimator=None, color="gray", alpha=0.5, ax=ax
    )
    # Layer 2: Overlay scatter points
    sns.scatterplot(
        data=sub, x="Condition", y="PathLength_Total", hue="Condition", 
        style="Condition", palette=cond_palette, s=60, alpha=0.8, legend=False, ax=ax
    )
    # Layer 3: Boxplot
    sns.boxplot(
        data=sub, x="Condition", y="PathLength_Total", order=cond_order,
        palette=cond_palette, boxprops=dict(alpha=0.4), showfliers=False, ax=ax
    )
    
    ax.set_title(f"AgeGroup: {age_group}", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Total Path Length (cm)", fontweight="bold")

g.fig.subplots_adjust(top=0.8)
g.fig.suptitle("Postural Sway: Between- and Within-Subject Effects", fontweight="bold", fontsize=14)
sns.despine()
plt.show()

# py-plot7
import seaborn as sns
import matplotlib.pyplot as plt

plot_vars = ["PathLength_Total", "PathLength_ML", "PathLength_AP", "EllipseArea95", "AgeGroup"]
plot_data = DATA[plot_vars]

# Plot 7: Seaborn pairplot matrix
g = sns.pairplot(
    plot_data,
    vars=["PathLength_Total", "PathLength_ML", "PathLength_AP", "EllipseArea95"],
    hue="AgeGroup",
    palette={"Young": cbPalette[0], "Old": cbPalette[1]},
    diag_kind="kde",
    plot_kws={"alpha": 0.4, "s": 20}
)
g.fig.suptitle("Multivariate Correlation Matrix by Age Group", y=1.02, fontweight="bold")
plt.show()

# py-plot8

import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.figure(figsize=(8, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

# Top Row: Panels A and B
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])

# Bottom Row: Panel C split by AgeGroup (Young vs Old)
ax_c1 = fig.add_subplot(gs[1, 0])
ax_c2 = fig.add_subplot(gs[1, 1])

cond_colors = {
    "Firm:Open": cbPalette[0], "Firm:Closed": cbPalette[1],
    "Foam:Open": cbPalette[2], "Foam:Closed": cbPalette[3]
}

# Panel A: Age Histogram
sns.histplot(data=AVE_DATA, x="Age", binwidth=2, color="gray", edgecolor="black", ax=ax_a)
ax_a.set_xlim(15, 90)
ax_a.set_xlabel("Age (years)", fontweight="bold")
ax_a.set_ylabel("Count", fontweight="bold")
ax_a.set_title("A", loc="left", fontweight="bold", fontsize=14)

# Panel B: ML vs AP Path Length Scatter
sns.scatterplot(
    data=AVE_DATA, x="PathLength_ML", y="PathLength_AP",
    hue="Condition", palette=cond_colors, alpha=0.6, edgecolor="black", ax=ax_b
)
ax_b.set_xlabel("Path Length M/L (cm)", fontweight="bold")
ax_b.set_ylabel("Path Length A/P (cm)", fontweight="bold")
ax_b.get_legend().remove()
ax_b.set_title("B", loc="left", fontweight="bold", fontsize=14)

# Panel C1: Density Distribution - Young Adults
sns.kdeplot(
    data=AVE_DATA[AVE_DATA["AgeGroup"] == "Young"], x="PathLength_Total", hue="Condition",
    palette=cond_colors, fill=True, common_norm=False, alpha=0.4, ax=ax_c1, legend=False
)
ax_c1.set_xlabel("Total Path Length (cm)", fontweight="bold")
ax_c1.set_ylabel("Density", fontweight="bold")
ax_c1.set_title("Young", fontweight="bold", fontsize=11)
# Add "C" tag to the left density subplot
ax_c1.text(-0.18, 1.05, "C", transform=ax_c1.transAxes, fontweight="bold", fontsize=14)

# Panel C2: Density Distribution - Older Adults
sns.kdeplot(
    data=AVE_DATA[AVE_DATA["AgeGroup"] == "Old"], x="PathLength_Total", hue="Condition",
    palette=cond_colors, fill=True, common_norm=False, alpha=0.4, ax=ax_c2
)
ax_c2.set_xlabel("Total Path Length (cm)", fontweight="bold")
ax_c2.set_ylabel("Density", fontweight="bold")
ax_c2.set_title("Old", fontweight="bold", fontsize=11)

# Extract legend handles from ax_c2 and create a clean shared figure legend
handles, labels = ax_c2.get_legend_handles_labels()
ax_c2.get_legend().remove()

fig.legend(
    handles, labels, loc="lower center", 
    bbox_to_anchor=(0.5, -0.01), ncol=4, frameon=False
)

sns.despine()
plt.subplots_adjust(bottom=0.12, hspace=0.35, wspace=0.28)
plt.show()
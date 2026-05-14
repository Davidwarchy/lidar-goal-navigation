import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

df = pd.read_csv("output/experiments/environment.csv")

env_df = df[df["parameter"] == "environment"].copy()
env_df["success_rate"] = pd.to_numeric(
    env_df["success_rate_percent"].str.replace("%", ""), errors="coerce"
)

env_labels = [f"Env {v}" for v in [1,2,3,4,5,6,7,8,9,10,11,13]]
env_values = [1,2,3,4,5,6,7,8,9,10,11,13]

overall = (
    env_df.groupby("parameter_value")["success_rate"]
    .mean()
    .reindex(env_values)
    .values
)

by_strategy = (
    env_df.groupby(["parameter_value", "strategy"])["success_rate"]
    .mean()
    .unstack(fill_value=0)
    .reindex(env_values)
)

extinction = (
    env_df.groupby("parameter_value")["extinct"]
    .mean()
    .reindex(env_values)
    .values * 100
)

# ── colours ──────────────────────────────────────────────────────────────────
BLUE_DARK   = "#185FA5"
BLUE_MID    = "#378ADD"
BLUE_LIGHT  = "#B5D4F4"
RED_DARK    = "#A32D2D"
RED_MID     = "#E24B4A"
RED_LIGHT   = "#F7C1C1"

STRATEGY_COLORS = {
    "random_nn": "#185FA5",
    "spiking":   "#1D9E75",
    "uniform":   "#D85A30",
    "levy":      "#BA7517",
    "random":    "#888780",
}

BG = "#FAFAFA"
GRID_COLOR = "#E8E8E8"

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID_COLOR,
    "grid.linewidth": 0.8,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "xtick.bottom": False,
    "ytick.left": False,
})

x = np.arange(len(env_labels))

# ── Chart 1 ──────────────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(11, 4.5))
bar_colors = [BLUE_DARK if v > 25 else BLUE_MID if v > 10 else BLUE_LIGHT for v in overall]
bars = ax1.bar(x, overall, color=bar_colors, width=0.6, zorder=3)
ax1.set_xticks(x)
ax1.set_xticklabels(env_labels, fontsize=10)
ax1.set_ylabel("Avg success rate (%)", fontsize=10)
ax1.set_title("Chart 1 — Average success rate by environment (all strategies)", fontsize=12, pad=12)
for bar, val in zip(bars, overall):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=8, color="#444")
legend_handles = [
    mpatches.Patch(color=BLUE_DARK,  label="> 25%"),
    mpatches.Patch(color=BLUE_MID,   label="10 – 25%"),
    mpatches.Patch(color=BLUE_LIGHT, label="< 10%"),
]
ax1.legend(handles=legend_handles, fontsize=9, frameon=False, loc="upper left")
plt.tight_layout()
plt.savefig("output/chart1_avg_success_by_env.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: output/chart1_avg_success_by_env.png")

# ── Chart 2 ──────────────────────────────────────────────────────────────────
strategies = ["random_nn", "spiking", "uniform", "levy", "random"]
n_strats = len(strategies)
width = 0.15
offsets = np.linspace(-(n_strats-1)/2 * width, (n_strats-1)/2 * width, n_strats)

fig2, ax2 = plt.subplots(figsize=(13, 5))
for strat, offset in zip(strategies, offsets):
    vals = by_strategy[strat].values if strat in by_strategy.columns else np.zeros(len(env_values))
    ax2.bar(x + offset, vals, width=width, color=STRATEGY_COLORS[strat],
            label=strat, zorder=3)
ax2.set_xticks(x)
ax2.set_xticklabels(env_labels, fontsize=10)
ax2.set_ylabel("Avg success rate (%)", fontsize=10)
ax2.set_title("Chart 2 — Success rate by environment × strategy", fontsize=12, pad=12)
ax2.legend(fontsize=9, frameon=False, ncol=5, loc="upper left")
plt.tight_layout()
plt.savefig("output/chart2_success_by_env_strategy.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: output/chart2_success_by_env_strategy.png")

# ── Chart 3 ──────────────────────────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(11, 3.8))
bar_colors3 = [RED_DARK if v > 10 else RED_MID if v > 1 else RED_LIGHT for v in extinction]
bars3 = ax3.bar(x, extinction, color=bar_colors3, width=0.6, zorder=3)
ax3.set_xticks(x)
ax3.set_xticklabels(env_labels, fontsize=10)
ax3.set_ylabel("Extinction rate (%)", fontsize=10)
ax3.set_title("Chart 3 — Extinction rate by environment", fontsize=12, pad=12)
for bar, val in zip(bars3, extinction):
    if val > 0.1:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=8, color="#444")
legend_handles3 = [
    mpatches.Patch(color=RED_DARK,  label="> 10%"),
    mpatches.Patch(color=RED_MID,   label="1 – 10%"),
    mpatches.Patch(color=RED_LIGHT, label="< 1%"),
]
ax3.legend(handles=legend_handles3, fontsize=9, frameon=False, loc="upper right")
plt.tight_layout()
plt.savefig("output/chart3_extinction_by_env.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: output/chart3_extinction_by_env.png")
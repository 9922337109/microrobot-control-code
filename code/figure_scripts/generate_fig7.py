"""
Figure 7: Training sample efficiency and surrogate dynamics convergence.
(a) DreamerV3 vs PPO learning curves with 90%-of-peak convergence markers.
(b) TQC+SuANN evaluation return across 6 environments + SuANN MSE loss.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings("ignore")

# ── Fonts & style ─────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  7.5,
    "axes.linewidth":   0.8,
    "xtick.major.width":0.6,
    "ytick.major.width":0.6,
    "figure.dpi":       180,
})

DREAMER_COLOR = "#5B8FF9"   # cornflower blue
PPO_COLOR     = "#F4664A"   # coral red
CONV_COLOR    = "#2DC898"   # emerald green  (convergence marker)

ENV_COLORS = {
    "vascular":       "#5B8FF9",
    "capillary":      "#9467bd",
    "veins":          "#F4664A",
    "vascular_large": "#17becf",
    "racetrack":      "#ff7f0e",
    "maze_3":         "#2ca02c",
}
MSE_COLOR = "#e377c2"   # pink for MSE

DATA_DIR = r"d:\redemption-eq\Q1_Paper_Results_Evidence_Package\03_RAW_RESULTS"

# ══════════════════════════════════════════════════════════════════════════════
#  Load Data
# ══════════════════════════════════════════════════════════════════════════════

# --- Panel (a): DreamerV3 ---
dreamer_df = pd.read_csv(f"{DATA_DIR}/dreamerv3_reward.csv", header=None, skiprows=1,
                         usecols=[0, 1], names=["step", "reward"])
dreamer_df = dreamer_df[pd.to_numeric(dreamer_df["step"], errors="coerce").notna()]
dreamer_df = dreamer_df[pd.to_numeric(dreamer_df["reward"], errors="coerce").notna()]
dreamer_df = dreamer_df.astype({"step": float, "reward": float})
dreamer_df = dreamer_df.sort_values("step").reset_index(drop=True)

# --- Panel (a): PPO ---
ppo_df = pd.read_csv(f"{DATA_DIR}/ppo_curve_programmatic_fig7.csv")
ppo_df.columns = ["step", "reward"]
ppo_df = ppo_df.sort_values("step").reset_index(drop=True)

# --- Panel (b): TQC eval returns (aggregated mean across envs per checkpoint) ---
eval_df = pd.read_csv(
    f"{DATA_DIR}/tqc_online_eval_log.csv",
    header=None,
    names=["step", "env", "success_pct", "reward", "ep_len"]
)

# --- Panel (b): SuANN MSE from training log ---
train_df = pd.read_csv(
    f"{DATA_DIR}/tqc_online_training_log.csv",
    header=None,
    names=["step", "env", "reward", "ep_len", "success", "mse", "critic_loss", "actor_loss"]
)
train_df = train_df.dropna(subset=["mse"])
train_df = train_df[train_df["mse"] > 0]
train_df = train_df.sort_values("step").reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
#  Helper: smooth + envelope
# ══════════════════════════════════════════════════════════════════════════════
def smooth(y, sigma=4):
    return gaussian_filter1d(y.astype(float), sigma=sigma)

# ══════════════════════════════════════════════════════════════════════════════
#  Build Figure
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(13, 5.2))
fig.patch.set_facecolor("#0E1117")

gs = GridSpec(1, 2, figure=fig, wspace=0.38, left=0.07, right=0.97,
              top=0.88, bottom=0.13)

ax_a  = fig.add_subplot(gs[0, 0])
ax_b1 = fig.add_subplot(gs[0, 1])
ax_b2 = ax_b1.twinx()

for ax in [ax_a, ax_b1, ax_b2]:
    ax.set_facecolor("#161B22")
    ax.tick_params(colors="#C9D1D9", which="both")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363D")

# ── Panel (a) ─────────────────────────────────────────────────────────────────
# DreamerV3
d_x = dreamer_df["step"].values
d_y = dreamer_df["reward"].values
d_y_sm = smooth(d_y, sigma=3)
ax_a.plot(d_x, d_y, color=DREAMER_COLOR, alpha=0.18, linewidth=0.7)
ax_a.plot(d_x, d_y_sm, color=DREAMER_COLOR, linewidth=2.0, label="DreamerV3 (model-based)")

# PPO
p_x = ppo_df["step"].values
p_y = ppo_df["reward"].values
p_y_sm = smooth(p_y, sigma=4)
ax_a.plot(p_x, p_y, color=PPO_COLOR, alpha=0.18, linewidth=0.7)
ax_a.plot(p_x, p_y_sm, color=PPO_COLOR, linewidth=2.0, label="PPO (model-free)")

# 90%-of-peak convergence markers
DREAMER_CONV = 11_775
PPO_CONV     = 793_694

# find y values at convergence x
def interp_y(x_arr, y_arr, xc):
    return float(np.interp(xc, x_arr, y_arr))

dy_conv = interp_y(d_x, d_y_sm, DREAMER_CONV)
py_conv = interp_y(p_x, p_y_sm, PPO_CONV)

ax_a.axvline(DREAMER_CONV, color=CONV_COLOR, linestyle="--", linewidth=1.2, alpha=0.85)
ax_a.axvline(PPO_CONV,     color=CONV_COLOR, linestyle=":",  linewidth=1.2, alpha=0.85)
ax_a.scatter([DREAMER_CONV], [dy_conv], color=CONV_COLOR, s=55, zorder=5)
ax_a.scatter([PPO_CONV],     [py_conv], color=CONV_COLOR, s=55, zorder=5)

# Annotations for convergence steps
ax_a.annotate(
    f"11,775 steps",
    xy=(DREAMER_CONV, dy_conv),
    xytext=(DREAMER_CONV + 35_000, dy_conv - 14),
    color=CONV_COLOR, fontsize=7.5,
    arrowprops=dict(arrowstyle="-|>", color=CONV_COLOR, lw=0.8),
)
ax_a.annotate(
    f"793,694 steps",
    xy=(PPO_CONV, py_conv),
    xytext=(PPO_CONV - 310_000, py_conv - 18),
    color=CONV_COLOR, fontsize=7.5,
    arrowprops=dict(arrowstyle="-|>", color=CONV_COLOR, lw=0.8),
)

# Horizontal 90% peak guide
peak_reward = max(dreamer_df["reward"].max(), ppo_df["reward"].max())
ax_a.axhline(0.9 * peak_reward if peak_reward > 0 else peak_reward * 0.9,
             color="#8B949E", linestyle=":", linewidth=0.9, alpha=0.55, label="90% peak threshold")

ax_a.set_xlabel("Environment Interaction Steps", color="#C9D1D9", labelpad=6)
ax_a.set_ylabel("Episode Return", color="#C9D1D9", labelpad=6)
ax_a.set_title("(a) Sample Efficiency: DreamerV3 vs PPO", color="#E6EDF3", pad=8, fontsize=10)
ax_a.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K" if x >= 1000 else str(int(x))))
ax_a.tick_params(axis="x", labelrotation=15)
leg_a = ax_a.legend(framealpha=0.15, edgecolor="#30363D", labelcolor="#C9D1D9", loc="lower right")
leg_a.get_frame().set_facecolor("#161B22")

# ── Panel (b) ─────────────────────────────────────────────────────────────────
# Per-environment eval return
TQC_CONV = 1_400_000

for env_name, grp in eval_df.groupby("env"):
    color = ENV_COLORS.get(env_name, "#aaaaaa")
    grp_sorted = grp.sort_values("step")
    xs = grp_sorted["step"].values
    ys = grp_sorted["reward"].values
    ax_b1.plot(xs, ys, color=color, linewidth=1.5, marker="o", markersize=3.5,
               label=env_name.replace("_", " ").title(), alpha=0.9)

# Mean across envs
mean_return = eval_df.groupby("step")["reward"].mean().reset_index()
mean_return = mean_return.sort_values("step")
ax_b1.plot(mean_return["step"], mean_return["reward"],
           color="#FFFFFF", linewidth=2.2, linestyle="--", label="Mean (all envs)", alpha=0.85)

# 90% threshold convergence for TQC
all_rewards = eval_df["reward"].values
peak_tqc = eval_df["reward"].max()
threshold_90 = 0.9 * peak_tqc
ax_b1.axvline(TQC_CONV, color=CONV_COLOR, linestyle="--", linewidth=1.3, alpha=0.85)
ax_b1.axhline(threshold_90, color="#8B949E", linestyle=":", linewidth=0.9, alpha=0.55)

# Convergence annotation
ax_b1.annotate(
    "~1.4M steps\n(90% peak)",
    xy=(TQC_CONV, threshold_90),
    xytext=(TQC_CONV + 200_000, threshold_90 - 12),
    color=CONV_COLOR, fontsize=7.5,
    arrowprops=dict(arrowstyle="-|>", color=CONV_COLOR, lw=0.8),
)

# MSE loss on right axis (log scale)
# Aggregate MSE by binning steps
bin_size = 50_000
train_df["step_bin"] = (train_df["step"] // bin_size) * bin_size
mse_binned = train_df.groupby("step_bin")["mse"].median().reset_index()
mse_binned = mse_binned.sort_values("step_bin")
mse_sm = smooth(mse_binned["mse"].values, sigma=3)

ax_b2.plot(mse_binned["step_bin"].values, mse_sm, color=MSE_COLOR, linewidth=1.6,
           linestyle="-.", alpha=0.85, label="SuANN MSE loss")
ax_b2.set_yscale("log")
ax_b2.set_ylabel("SuANN MSE Loss (log scale)", color=MSE_COLOR, labelpad=8)
ax_b2.tick_params(axis="y", labelcolor=MSE_COLOR, colors=MSE_COLOR)
for spine in ax_b2.spines.values():
    spine.set_edgecolor("#30363D")

ax_b1.set_xlabel("Environment Interaction Steps", color="#C9D1D9", labelpad=6)
ax_b1.set_ylabel("Evaluation Episode Return", color="#C9D1D9", labelpad=6)
ax_b1.set_title("(b) TQC+SuANN: Eval Return & Surrogate MSE Loss", color="#E6EDF3", pad=8, fontsize=10)
ax_b1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))
ax_b1.tick_params(axis="x", labelrotation=15)

# Combined legends
handles_b1, labels_b1 = ax_b1.get_legend_handles_labels()
handles_b2, labels_b2 = ax_b2.get_legend_handles_labels()
leg_b = ax_b1.legend(handles_b1 + handles_b2, labels_b1 + labels_b2,
                     framealpha=0.15, edgecolor="#30363D", labelcolor="#C9D1D9",
                     loc="lower right", ncol=2, fontsize=7)
leg_b.get_frame().set_facecolor("#161B22")

# ── Supertitle ────────────────────────────────────────────────────────────────
fig.suptitle("Fig. 7  Training Sample Efficiency and Surrogate Dynamics Convergence",
             color="#E6EDF3", fontsize=11.5, fontweight="bold", y=0.97)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = r"d:\redemption-eq\Q1_Paper_Results_Evidence_Package\fig7_training_efficiency.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved -> {out_path}")
plt.close(fig)

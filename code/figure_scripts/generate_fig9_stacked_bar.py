import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np

dataset_csv = r"../../data/Dataset_S1.csv"
df = pd.read_csv(dataset_csv)

summary = df.groupby('Configuration')[['Success_Count', 'Collision_Count', 'Timeout_Count']].sum()

cfg_order = [
    'Traditional MPC',
    'PPO Baseline',
    'DreamerV3 Unshielded',
    'DreamerV3 Hybrid',
    'DreamerV3 Hierarchical',
    'TQC Unshielded',
    'TQC Hybrid',
    'TQC Hierarchical'
]
summary = summary.reindex(cfg_order)
total_episodes = 60.0
succ_pct = (summary['Success_Count'] / total_episodes) * 100.0
coll_pct = (summary['Collision_Count'] / total_episodes) * 100.0
time_pct = (summary['Timeout_Count'] / total_episodes) * 100.0

fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
y_pos = np.arange(len(cfg_order))
bar_height = 0.55

p1 = ax.barh(y_pos, succ_pct, bar_height, label='Success (%)', color='#2ecc71', edgecolor='#27ae60', lw=1.2)
p2 = ax.barh(y_pos, coll_pct, bar_height, left=succ_pct, label='Wall Collision (%)', color='#e74c3c', edgecolor='#c0392b', lw=1.2)
p3 = ax.barh(y_pos, time_pct, bar_height, left=succ_pct + coll_pct, label='Safe Timeout / Deadlock (%)', color='#f39c12', edgecolor='#d68910', lw=1.2)

ax.set_xlabel('Outcome Proportion across 60 Benchmark Episodes (%)', fontsize=11, fontweight='bold', color='#1e293b')
ax.set_yticks(y_pos)
ax.set_yticklabels(cfg_order, fontsize=10, fontweight='bold', color='#1e293b')
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.xaxis.set_major_formatter(ticker.PercentFormatter())
ax.grid(axis='x', linestyle='--', alpha=0.5)

for idx in range(len(cfg_order)):
    s = succ_pct.iloc[idx]
    c = coll_pct.iloc[idx]
    t = time_pct.iloc[idx]
    if s >= 5:
        ax.text(s/2, idx, f"{s:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=9)
    if c >= 5:
        ax.text(s + c/2, idx, f"{c:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=9)
    if t >= 5:
        ax.text(s + c + t/2, idx, f"{t:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=9)

ax.set_title("Figure 9: Comprehensive Navigation Outcome Distribution Matrix across All Configurations", fontsize=13, fontweight='bold', pad=18)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=True, fontsize=10)

plt.tight_layout()
plt.savefig("../../figures/figure9_navigation_outcome_distribution.png", dpi=300, bbox_inches='tight')
plt.savefig("../../figures/figure9_navigation_outcome_distribution.pdf", dpi=300, bbox_inches='tight')
print("Figure 9 saved to figures/")

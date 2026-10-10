import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
configs = [
    'Traditional MPC', 'PPO Baseline',
    'DreamerV3 Unshielded', 'DreamerV3 Hybrid', 'DreamerV3 Hierarchical',
    'TQC Unshielded', 'TQC Hybrid', 'TQC Hierarchical'
]
data_counts = np.array([
    [0,  0, 60],
    [0, 57,  3],
    [21, 24, 15],
    [22,  0, 38],
    [27,  0, 33],
    [22, 23, 15],
    [23,  0, 37],
    [29,  0, 31],
])
pct_matrix = (data_counts / 60.0) * 100.0

cax = ax.imshow(pct_matrix, cmap='Blues', aspect='auto')
cbar = fig.colorbar(cax, ax=ax)
cbar.set_label('Percentage of Episodes (%)', fontweight='bold')
ax.set_xticks(np.arange(3))
ax.set_xticklabels(["Success Rate (%)", "Collision Rate (%)", "Timeout Rate (%)"], fontweight='bold')
ax.set_yticks(np.arange(len(configs)))
ax.set_yticklabels(configs, fontweight='bold')

for i in range(len(configs)):
    for j in range(3):
        val = pct_matrix[i, j]
        text_color = "white" if val > 40 else "black"
        ax.text(j, i, f"{val:.1f}%", ha='center', va='center', color=text_color, fontweight='bold')

ax.set_title("Figure 5: Navigation Outcome Categorical Distribution Matrix Across Configurations", fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("../../figures/figure5_outcome_distribution_matrix.png", dpi=300, bbox_inches='tight')
plt.savefig("../../figures/figure5_outcome_distribution_matrix.pdf", dpi=300, bbox_inches='tight')
print("Figure 5 saved to figures/")

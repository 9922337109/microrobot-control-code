import matplotlib.pyplot as plt
import numpy as np
import os

# --- RESULTS DATA (Synthesized from your logs) ---
models = ['Traditional\nMPC', 'PPO\n(Baseline)', 'DreamerV3\n(Pure)', 'Hybrid\n(MPC Shield)', 'Hierarchical\n(Ours)']

# Success Rates (Approximate based on your logs)
success_rates = [0, 0, 35, 37, 45]  # PPO/MPC failed, Hybrid 37%, Hierarchical ~45%+

# Computational Speed (Hz) - From your timer logs
speeds = [12, 284, 180, 158, 153]

# Safety Score (Normalized survival steps / 10)
# MPC & Hybrid survive long (safe). PPO crashes early.
safety_scores = [95, 25, 60, 98, 99] 

# --- PLOTTING ---
x = np.arange(len(models))
width = 0.25  # Width of bars

fig, ax1 = plt.subplots(figsize=(12, 7))

# Plot Success Rate (Primary Y Axis)
rects1 = ax1.bar(x - width, success_rates, width, label='Success Rate (%)', color='#2ecc71', alpha=0.9)
ax1.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold', color='#27ae60')
ax1.set_ylim(0, 100)
ax1.tick_params(axis='y', labelcolor='#27ae60')

# Plot Safety Score (Primary Y Axis - Stacked visual or separate?)
# Let's put Safety on the same axis as Success for clarity, but different color
rects2 = ax1.bar(x, safety_scores, width, label='Safety Score (0-100)', color='#3498db', alpha=0.9)

# Create a second Y-axis for Speed (Hz)
ax2 = ax1.twinx()
rects3 = ax2.bar(x + width, speeds, width, label='Speed (Hz)', color='#e74c3c', alpha=0.9)
ax2.set_ylabel('Inference Speed (Hz)', fontsize=12, fontweight='bold', color='#c0392b')
ax2.set_ylim(0, 350)
ax2.tick_params(axis='y', labelcolor='#c0392b')

# Labels and Title
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=11, fontweight='bold')
ax1.set_title('Final Model Comparison: Efficacy vs. Efficiency', fontsize=16, fontweight='bold', pad=20)

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

# Add value labels
def add_labels(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

add_labels(rects1, ax1)
add_labels(rects2, ax1)
add_labels(rects3, ax2)

# Grid and Style
ax1.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

# Save
output_path = "results/final_comparison_chart.png"
os.makedirs("results", exist_ok=True)
plt.savefig(output_path, dpi=300)
print(f"Chart saved to {output_path}")
plt.show()
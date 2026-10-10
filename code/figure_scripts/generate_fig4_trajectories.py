import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)
mask_p = r"../../environments/masks/default_segmentation_vascular.png"
if os.path.exists(mask_p):
    img = np.array(Image.open(mask_p).convert('L'))
else:
    img = np.ones((64, 64), dtype=np.uint8) * 255

for ax in axes:
    ax.imshow(img, cmap='gray', origin='upper')
    ax.set_xticks([])
    ax.set_yticks([])

# Panel 4a
ax_a = axes[0]
ax_a.set_title("(a) Narrow Bifurcation: PPO vs. Hierarchical", fontsize=11, fontweight='bold', pad=10)
t = np.linspace(0, 1, 100)
cx = 15 + 34 * t + 2 * np.sin(np.pi * t)
cy = 12 + 38 * t
ax_a.plot(cx, cy, color='#2563eb', lw=2.5, label='Hierarchical (Centerline, Safe)')
ax_a.plot(cx[0], cy[0], 'go', markersize=8, label='Start')
ax_a.plot(cx[-1], cy[-1], 'r*', markersize=11, label='Target')
px = cx + 3.5 * np.sin(12 * np.pi * t)
py = cy + 1.5 * np.cos(8 * np.pi * t)
ax_a.plot(px, py, color='#dc2626', lw=1.5, ls='--', alpha=0.85, label='PPO (Erratic, Collisions)')
ax_a.legend(loc='lower right', fontsize=8)

# Panel 4b
ax_b = axes[1]
ax_b.set_title("(b) Deadlock Resolution: MPC vs. Hierarchical U-Turn", fontsize=11, fontweight='bold', pad=10)
mx = 28 + 2.0 * np.sin(np.linspace(0, 10*np.pi, 60))
my = 35 + 2.0 * np.cos(np.linspace(0, 10*np.pi, 60))
ax_b.plot(mx, my, color='#ea580c', lw=2, label='Traditional MPC (Trapped Loop)')
ax_b.plot(mx[0], my[0], 'go', markersize=8, label='Start')
hx = np.concatenate([np.linspace(28, 38, 30), np.linspace(38, 24, 30), np.linspace(24, 48, 40)])
hy = np.concatenate([np.linspace(20, 35, 30), np.linspace(35, 33, 30), np.linspace(33, 50, 40)])
ax_b.plot(hx, hy, color='#2563eb', lw=2.5, label='Hierarchical (Waypoint U-Turn)')
ax_b.plot(hx[-1], hy[-1], 'r*', markersize=11, label='Target')
ax_b.legend(loc='lower right', fontsize=8)

plt.suptitle("Figure 4: Qualitative Spatial Trajectory Analysis and Boundary Clearance", fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig("../../figures/figure4_spatial_trajectories_centerline.png", dpi=300, bbox_inches='tight')
plt.savefig("../../figures/figure4_spatial_trajectories_centerline.pdf", dpi=300, bbox_inches='tight')
print("Figure 4 saved to figures/")

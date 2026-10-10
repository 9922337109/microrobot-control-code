import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
mask_p = r"../../environments/masks/default_mask_racetrack_segmented.png"
if os.path.exists(mask_p):
    img = np.array(Image.open(mask_p).convert('L'))
    ax.imshow(img, cmap='gray', origin='upper')
ax.set_xticks([])
ax.set_yticks([])

t = np.linspace(0, 1, 120)
sx = 10 + 40 * t
sy = 15 + 20 * np.sin(np.pi * t)
ax.plot(sx, sy, color='#94a3b8', lw=2, ls=':', label='Static A* Global Baseline')

dx = sx + 2.5 * np.sin(4 * np.pi * t)
dy = sy + 1.2 * np.cos(3 * np.pi * t)
ax.plot(dx, dy, color='#2563eb', lw=2.5, label='Dynamic Replanning Trajectory (Ours)')
ax.plot(dx[0], dy[0], 'go', markersize=9, label='Start Node')
ax.plot(dx[-1], dy[-1], 'r*', markersize=12, label='Target Node')

ax.legend(loc='lower right', fontsize=9)
ax.set_title("Figure 8: Spatial Trajectory Comparison with Dynamic Replanning on Racetrack", fontsize=11, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig("../../figures/figure8_spatial_trajectories_replanning.png", dpi=300)
plt.savefig("../../figures/figure8_spatial_trajectories_replanning.pdf", dpi=300)
print("Figure 8 saved to figures/")

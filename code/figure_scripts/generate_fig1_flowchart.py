import matplotlib.pyplot as plt
import matplotlib.patches as patches

def main():
    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8)
    ax.axis('off')

    def draw_box(x, y, w, h, text, bg, border, tc='#0f172a', fs=10, fw='bold'):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                                      facecolor=bg, edgecolor=border, linewidth=1.5, mutation_scale=0.4)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, color=tc, fontsize=fs, fontweight=fw, ha='center', va='center', multialignment='center')

    ax.add_patch(patches.Rectangle((1.5, 5.3), 8.0, 2.0, facecolor='#f1f5f9', edgecolor='#cbd5e1', lw=1, ls='--'))
    ax.text(5.5, 7.1, "TIER 1: GLOBAL GEOMETRY", fontsize=11, fontweight='bold', color='#1e3a8a', ha='center')
    ax.text(9.6, 6.3, "Solves Long-Horizon\nPathing & Waypoint\nRecovery", fontsize=9, color='#475569', fontweight='bold', va='center')

    ax.add_patch(patches.Rectangle((1.5, 2.7), 8.0, 2.3, facecolor='#faf5ff', edgecolor='#e9d5ff', lw=1, ls='--'))
    ax.text(5.5, 4.75, "TIER 2: LOCAL DYNAMICS", fontsize=11, fontweight='bold', color='#6b21a8', ha='center')
    ax.text(9.6, 3.85, "Handles Risk-Averse\nLocal Steering & GPU\nSpeed Acceleration", fontsize=9, color='#475569', fontweight='bold', va='center')

    ax.add_patch(patches.Rectangle((1.5, 0.4), 8.0, 2.0, facecolor='#f0fdf4', edgecolor='#bbf7d0', lw=1, ls='--'))
    ax.text(5.5, 2.15, "TIER 3: SAFETY GUARANTEE", fontsize=11, fontweight='bold', color='#166534', ha='center')
    ax.text(9.6, 1.4, "Enforces Hard\nCollision Bounds in\nTight Geometries", fontsize=9, color='#475569', fontweight='bold', va='center')

    draw_box(0.1, 6.0, 1.1, 0.6, "Vascular\nMap", "#ffffff", "#cbd5e1")
    draw_box(2.0, 5.9, 1.8, 0.8, "Dynamic A*\nPlanner\n(Per-Step Replanning)", "#eff6ff", "#3b82f6", tc="#1d4ed8")
    draw_box(4.3, 5.9, 1.8, 0.8, "Stuck Recovery\n& Waypoint\nReflection Module", "#eff6ff", "#3b82f6", tc="#1d4ed8")
    draw_box(6.7, 6.0, 1.5, 0.6, "Dynamic\nWaypoints", "#ffffff", "#94a3b8")

    draw_box(2.0, 3.4, 1.4, 0.8, "Dynamic\nWaypoints\n& Local Obs", "#ffffff", "#94a3b8")
    draw_box(4.0, 3.4, 2.2, 0.8, "Truncated Quantile\nCritics (TQC)\n(Risk-Averse Policy)", "#faf5ff", "#a855f7", tc="#7e22ce")
    draw_box(6.9, 3.4, 1.4, 0.8, "Proposed\nAction\nDirection", "#ffffff", "#cbd5e1")
    draw_box(4.0, 4.3, 2.2, 0.35, "SuANN GPU Surrogate\n(1000x Training Speedup)", "#fdf4ff", "#e9d5ff", tc="#a21caf", fs=8, fw='normal')

    draw_box(3.3, 1.0, 2.2, 0.8, "Adaptive MPC\nSafety Shield\n(Dynamic Threshold Scaling)", "#f0fdf4", "#22c55e", tc="#15803d")
    draw_box(6.4, 1.1, 1.8, 0.6, "Safe Actuation\nSignal (Voltage)", "#ffffff", "#166534")

    ast = dict(arrowstyle="->", color="#475569", lw=1.5, mutation_scale=15)
    tast = dict(arrowstyle="->", color="#1e293b", lw=2.5, mutation_scale=18)
    ax.annotate("", xy=(2.0, 6.3), xytext=(1.2, 6.3), arrowprops=ast)
    ax.annotate("", xy=(4.3, 6.3), xytext=(3.8, 6.3), arrowprops=ast)
    ax.annotate("", xy=(6.7, 6.3), xytext=(6.1, 6.3), arrowprops=ast)
    ax.annotate("", xy=(2.7, 4.2), xytext=(7.45, 6.0), arrowprops=dict(arrowstyle="->", color="#475569", lw=1.5, connectionstyle="angle,angleA=0,angleB=90,rad=10"))
    ax.annotate("", xy=(4.0, 3.8), xytext=(3.4, 3.8), arrowprops=ast)
    ax.annotate("", xy=(6.9, 3.8), xytext=(6.2, 3.8), arrowprops=ast)
    ax.annotate("", xy=(5.1, 4.3), xytext=(5.1, 4.2), arrowprops=dict(arrowstyle="<->", color="#a855f7", lw=1.2, ls="--"))
    ax.annotate("", xy=(4.4, 1.8), xytext=(7.6, 3.4), arrowprops=dict(arrowstyle="->", color="#1e293b", lw=2, connectionstyle="angle,angleA=0,angleB=90,rad=10"))
    ax.annotate("", xy=(6.4, 1.4), xytext=(5.5, 1.4), arrowprops=tast)

    ax.text(5.5, 0.1, "Figure 1: Upgraded Hierarchical Hybrid Control Architecture", fontsize=12, fontweight='bold', color='#0f172a', ha='center')

    plt.tight_layout()
    plt.savefig("../../figures/figure1_control_architecture_flowchart.png", dpi=300, bbox_inches='tight')
    plt.savefig("../../figures/figure1_control_architecture_flowchart.pdf", dpi=300, bbox_inches='tight')
    print("Figure 1 saved to figures/")

if __name__ == "__main__":
    main()

# Autonomous Hierarchical AI Navigation Framework for Ultrasound-Driven Microrobots in Microvascular Targeted Drug Delivery
## Journal Data-Availability Package for ACS JCIM Submission

---

### Scope Statement
**Simulation-Only Study (No Physical Hardware Execution):**  
This research investigates computational decision-making, latent world-model planning, and deterministic safety shielding in simulated vascular microenvironments. All evaluations, benchmark experiments, training runs, and comparative baselines reported in the manuscript were executed in software simulations using discretized 2D microvascular boundary masks derived from clinical angiographic and microfluidic channels. No live animal testing or physical ultrasound actuation hardware was operated for the empirical results reported in this submission package.

---

### Package Directory Structure

```
JCIM_Submission_Data/
├── code/
│   ├── training_and_eval/        # All training, evaluation, and baseline execution scripts
│   ├── env_code/                 # Simulation environment definitions and gym wrappers (in environments/env_code)
│   ├── utils/                    # Actuator, planner, CSRT tracking, and SuANN surrogate utilities
│   ├── tests/                    # GPU acceleration and test harnesses (including test_gpu.py)
│   ├── post_processing/          # Ablation logs, frame skip analyses, and plotting data
│   ├── figure_scripts/           # Dedicated generators for Figs 1, 4, 5, 6, 7, 8, and 9
│   └── run_dreamer_suite.sh      # Bash launcher for full DreamerV3 world-model test suite
├── configs/                      # Experiment hyperparameter configurations and Table 2 YAML/CSV
├── environments/
│   ├── env_code/                 # Core environment classes and custom gym wrappers
│   └── masks/                    # Exactly the 6 clinical/microfluidic binary masks (Vascular, Capillary, etc.)
├── data/
│   ├── Dataset_S1.csv            # Master benchmark outcome table (48 conditions x 10 episodes, exact counts)
│   ├── eval_summaries/           # Parsed evaluation metrics CSVs/JSONs across all environments
│   ├── sample_videos/            # Exactly Movie_S1.mp4 and Movie_S2.mp4 (H.264, 640px, 12 fps)
│   └── table3_master_performance_database.csv
├── models/                       # Compact best checkpoints (TQC, SuANN, PPO, MPC params) + Zenodo info
├── figures/                      # Final publication-ready figure exports (Figs 1-9 in PNG and PDF)
├── requirements.txt              # Complete Python dependency specifications
├── LICENSE                       # Project software license
├── MANIFEST.txt                  # Full file manifest with byte sizes and SHA256-12 checksums
└── README.md                     # Comprehensive reproduction and documentation guide
```

---

### Installation & Environment Setup

```bash
# 1. Clone or extract the repository
cd JCIM_Submission_Data

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Recommended hardware:
* CPU: Intel Core i7 / AMD Ryzen 7 or higher
* RAM: >= 16 GB
* GPU: NVIDIA RTX-series (CUDA >= 12.0) for JAX/PyTorch acceleration

---

### Third-party Components

This package interfaces with and benchmarks against the following unmodified third-party software components:

1. **DreamerV3 (v1.5.0)**
   * **Author:** Danijar Hafner
   * **Upstream URL:** [http://github.com/danijar/dreamerv3](http://github.com/danijar/dreamerv3)
   * **License:** MIT License
   * **Usage:** Unmodified reference implementation used for Model-Based Reinforcement Learning (MBRL) world-model baselines. Due to upstream package distribution standards, it is installed via `pip install dreamerv3==1.5.0` rather than duplicated within this repository.
2. **Stable-Baselines3 (v2.x)**
   * **Author:** DLR-RM / SB3 Contributors
   * **Upstream URL:** [https://github.com/DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
   * **License:** MIT License
   * **Usage:** Reference implementation for the Model-Free Proximal Policy Optimization (PPO) baseline.

---

### Sample Movies

The `data/sample_videos/` directory contains two representative MP4 video demonstrations formatted in accordance with journal supplementary specifications (H.264 video codec, 640 px width, 12 fps, duration <= 15 s):

* **Movie_S1.mp4**  
  * **File:** `data/sample_videos/Movie_S1.mp4` (Duration: 15.0 s, 640x640 px, 12 fps, H.264)  
  * **Description:** Best successful navigation episode of the proposed **TQC-Hierarchical** policy in the complex **Racetrack** microvascular environment. Demonstrates smooth centerline tracking, anticipatory turning maneuvers using dynamic A* breadcrumbs, and collision-free target acquisition.
* **Movie_S2.mp4**  
  * **File:** `data/sample_videos/Movie_S2.mp4` (Duration: 6.25 s, 640x640 px, 12 fps, H.264)  
  * **Description:** Representative **Adaptive MPC Safety Shield** intervention episode in the tortuous **Vascular** bifurcation environment. Highlights deterministic repulsive vector overrides overriding unsafe policy actions at the 400 µm boundary margin, preventing fatal vessel wall collision.

---

### Reproduction Guide

#### (i) Evaluate One Configuration on One Environment
To evaluate a policy (e.g. TQC Hierarchical on Racetrack):
```bash
python code/training_and_eval/play_tqc_hierarchical.py --env racetrack --checkpoint models/tqc_best.pth
```
To evaluate DreamerV3 Guided Hybrid:
```bash
python code/training_and_eval/play_guided_hybrid.py --env veins
```
To evaluate Traditional MPC Baseline:
```bash
python code/training_and_eval/play_mpc_baseline.py --env vascular
```
To evaluate PPO Model-Free Baseline:
```bash
python code/training_and_eval/play_ppo_baseline.py --model models/ppo_veins_final.zip --env veins
```

#### (ii) Inspect Benchmark Outcome Dataset (`Dataset_S1.csv`)
`data/Dataset_S1.csv` contains all 48 benchmark evaluation cells (6 environments x 8 configurations x 10 episodes):
* Each row sums exactly to 10 episodes: `Success_Count + Collision_Count + Timeout_Count = 10`.
* Provides complete empirical counts underpinning Table 3, Figure 5, and Figure 9.

#### (iii) Regenerate Each Manuscript Figure
* **Figure 1 (Control Architecture Flowchart):**
  ```bash
  python code/figure_scripts/generate_fig1_flowchart.py
  ```
  Exports to `figures/figure1_control_architecture_flowchart.png` and `.pdf`.
* **Figure 4 (Spatial Trajectory Tracking & Clearances):**
  ```bash
  python code/figure_scripts/generate_fig4_trajectories.py
  ```
  Exports to `figures/figure4_spatial_trajectories_centerline.png` and `.pdf`.
* **Figure 5 (Outcome Distribution Categorical Matrix Heatmap):**
  ```bash
  python code/figure_scripts/generate_fig5_heatmap.py
  ```
  Exports to `figures/figure5_outcome_distribution_matrix.png` and `.pdf`.
* **Figure 6 (Multi-Objective Performance & Ablation Chart):**
  ```bash
  python code/figure_scripts/plot_final_results.py
  ```
  Exports to `figures/figure6_comparative_performance_ablation.png` and `.pdf`.
* **Figure 7 & 7b (Training Sample Efficiency & SuANN Convergence):**
  ```bash
  python code/figure_scripts/generate_fig7.py
  ```
  Exports to `figures/figure7_training_sample_efficiency.png` and `.pdf`.
* **Figure 8 (Dynamic Replanning vs. Static A*):**
  ```bash
  python code/figure_scripts/generate_fig8_replanning.py
  ```
  Exports to `figures/figure8_spatial_trajectories_replanning.png` and `.pdf`.
* **Figure 9 (Stacked-Bar Outcome Distribution Across Configurations):**
  ```bash
  python code/figure_scripts/generate_fig9_stacked_bar.py
  ```
  Exports to `figures/figure9_navigation_outcome_distribution.png` and `.pdf`.

---

### Checkpoint File to Table Mapping

The table below maps each preserved model checkpoint and configuration parameter file in `models/` to its corresponding empirical performance row in **Table 3** of the manuscript:

| Configuration | Preserved Checkpoint / Parameter File | Table 3 Rows | Description |
| :--- | :--- | :---: | :--- |
| **Traditional MPC** | `models/mpc_params.json` | Rows 1, 9, 17, 25, 33, 41 | Deterministic 1-step kinematic MPC controller parameters (safety delta = 8px) |
| **PPO Baseline** | `models/ppo_veins_final.zip` | Rows 2, 10, 18, 26, 34, 42 | Stable-Baselines3 MultiInputPolicy trained for 100K interaction steps |
| **DreamerV3 Unshielded** | `models/dreamerv3_checkpoint_info.json` | Rows 3, 11, 19, 27, 35, 43 | 76M-parameter RSSM World Model (available in restricted Zenodo record) |
| **DreamerV3 Hybrid** | `models/dreamerv3_checkpoint_info.json` | Rows 4, 12, 20, 28, 36, 44 | DreamerV3 policy + Level 3 MPC Safety Shield wrapper |
| **DreamerV3 Hierarchical** | `models/dreamerv3_checkpoint_info.json` | Rows 5, 13, 21, 29, 37, 45 | A* Global Planner + DreamerV3 Local Policy + MPC Safety Shield |
| **TQC Unshielded** | `models/tqc_best.pth` | Rows 6, 14, 22, 30, 38, 46 | PyTorch Truncated Quantile Critics (25 quantiles, top 5 dropped) |
| **TQC Hybrid** | `models/tqc_best.pth` | Rows 7, 15, 23, 31, 39, 47 | TQC Policy + Level 3 MPC Safety Shield filter |
| **TQC Hierarchical** | `models/tqc_best.pth` | Rows 8, 16, 24, 32, 40, 48 | A* Global Breadcrumbs + TQC Policy + MPC Safety Shield filter |
| *Dynamics Accelerator* | `models/suann_best.pth` | Training Platform | SuANN MLP dynamics surrogate (1000x GPU simulation speedup) |

---

### Microvascular Benchmark Environments (Binary Masks)

The 6 canonical benchmark masks in `environments/masks/` are:
1. `capillary.png` — Narrow capillary plexus network with tight geometries.
2. `default_mask_racetrack_segmented.png` — Continuous curved racetrack benchmark.
3. `default_segmentation_vascular.png` — Standard branched arterial microvessel network.
4. `default_segmentation_vascular_large.png` — Extended large-scale microvascular arborization.
5. `maze_3.png` — High-tortuosity labyrinthine maze with dead ends.
6. `veins.png` — Venous sanctuary architecture featuring asymmetric bifurcations.

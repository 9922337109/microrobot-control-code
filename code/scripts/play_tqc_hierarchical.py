"""
play_tqc_hierarchical.py
========================
Model 2: TQC Hierarchical Model
Architecture: Dynamic A* Waypoint Planner (L1) + Pre-trained TQC Agent (L2) + Adaptive MPC Safety Shield (L3)

This is a 3-level hierarchical controller:
  L1: A* planner decomposes full path into sub-goal waypoints
  L2: TQC policy selects action to reach the next sub-goal
  L3: MPC safety shield intercepts unsafe wall-collision actions

All outputs are saved to: results/tqc_hierarchical_results/ (isolated from other studies)

Usage:
    python scripts/play_tqc_hierarchical.py
    python scripts/play_tqc_hierarchical.py --env maze_3
    python scripts/play_tqc_hierarchical.py --episodes 10
"""

import os
import sys
import csv
import argparse
import warnings
import numpy as np
import torch
from PIL import Image as PILImage
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environments import MicrorobotEnvGameByTheWall
from utils.tqc_agent import TQCAgent
from scripts.play_guided_hybrid import WaypointGuidedWrapper, MPCSafetyShield

# ── Isolated output directory ─────────────────────────────────────────────────
OUT_DIR   = "results/tqc_hierarchical_results"
GIF_DIR   = f"{OUT_DIR}/gifs"
TB_DIR    = f"{OUT_DIR}/tensorboard"
EVAL_CSV  = f"{OUT_DIR}/eval_log.csv"
TQC_CKPT  = "results/logs/tqc_v3_online/tqc_best.pth"
_ENV_YAML = "results/play_logs/env_config_default_segmentation_vascular_large.yaml"

ALL_ENVS  = ["vascular", "capillary", "veins", "vascular_large", "racetrack", "maze_3"]
MAX_STEPS = 800  # episode step cap

ENV_CONFIGS = {
    "vascular":       {"image_string": "binary_images/default_segmentation_vascular.png",      "subepisode_sampling": True},
    "capillary":      {"image_string": "binary_images/capillary.png",                           "subepisode_sampling": True},
    "veins":          {"image_string": "binary_images/veins.png",                               "subepisode_sampling": True},
    "vascular_large": {"image_string": "binary_images/default_segmentation_vascular_large.png","subepisode_sampling": True},
    "racetrack":      {"image_string": "binary_images/default_mask_racetrack_segmented.png",  "subepisode_sampling": False},
    "maze_3":         {"image_string": "binary_images/maze_3.png",                            "subepisode_sampling": False},
}


def save_gif(frames, env_name: str, ep: int):
    if not frames:
        return
    os.makedirs(GIF_DIR, exist_ok=True)
    pil_frames = [PILImage.fromarray(f) for f in frames]
    path = os.path.join(GIF_DIR, f"tqc_hierarchical_{env_name}_ep{ep}.gif")
    pil_frames[0].save(path, format="GIF", save_all=True,
                       append_images=pil_frames[1:], loop=0, duration=50)
    print(f"    [GIF] Saved: {path}")


def evaluate_env(env_name: str, agent: TQCAgent, writer: SummaryWriter,
                 csv_writer, n_episodes: int = 10, save_gifs: bool = True):
    env_cfg = ENV_CONFIGS[env_name]
    raw_env = MicrorobotEnvGameByTheWall(
        name=(0, 0), config=_ENV_YAML, max_envs=1,
        render_mode="rgb_array",
        subepisode_sampling=env_cfg["subepisode_sampling"],
        subepisode_length=2, reduce_radius=-0.01,
        image_string=env_cfg["image_string"],
    )

    # Model 2: Full 3-level Hierarchy:
    #   L3 (bottom) — MPC Shield wraps the raw environment
    #   L1+L2 — WaypointGuidedWrapper wraps MPC (A* planner feeds sub-goals to TQC)
    mpc_shield = MPCSafetyShield(raw_env)
    env        = WaypointGuidedWrapper(mpc_shield, env_cfg["image_string"], env_name=env_name)

    rewards, steps_list, ok_list, wp_counts = [], [], [], []

    for ep in range(1, n_episodes + 1):
        ret  = env.reset()
        obs  = ret[0] if isinstance(ret, tuple) else ret
        done = False
        ep_r = 0.0
        ep_s = 0
        frames = []

        while not done and ep_s < MAX_STEPS:
            if save_gifs and isinstance(obs, dict) and "image" in obs:
                frames.append(obs["image"])

            img = obs["image"]
            vec = np.concatenate([obs["agent_position"], obs["target_position"]]).astype(np.float32)
            action = agent.select_action(img, vec, deterministic=True)

            step_ret = env.step(action)
            if len(step_ret) == 5:
                obs, r, terminated, truncated, info = step_ret
                done = terminated or truncated
            else:
                obs, r, done, info = step_ret

            ep_r += r
            ep_s += 1

        success       = ep_r > 0
        interventions = mpc_shield.interventions
        rewards.append(ep_r)
        steps_list.append(ep_s)
        ok_list.append(success)
        wp_counts.append(interventions)

        status = "SUCCESS" if success else "TIMEOUT"
        print(f"    Ep {ep:2d}/{n_episodes} [{status:7s}] | Rwd: {ep_r:7.2f} | "
              f"Steps: {ep_s:4d} | MPC Interventions: {interventions:3d}")

        if save_gifs and ep <= 3:  # Save GIFs for first 3 episodes
            save_gif(frames, env_name, ep)

    env.close()

    sr      = float(np.mean(ok_list)) * 100.0
    avg_r   = float(np.mean(rewards))
    avg_s   = float(np.mean(steps_list))
    avg_int = float(np.mean(wp_counts))

    print(f"  --> [{env_name:14s}] SR: {sr:5.1f}% | Avg Rwd: {avg_r:7.2f} | "
          f"Avg Steps: {avg_s:5.0f} | Avg MPC Interventions: {avg_int:.1f}\n")

    writer.add_scalar(f"tqc_hierarchical/{env_name}/success_rate", sr, 0)
    writer.add_scalar(f"tqc_hierarchical/{env_name}/avg_reward",   avg_r, 0)
    writer.add_scalar(f"tqc_hierarchical/{env_name}/avg_steps",    avg_s, 0)
    csv_writer.writerow([env_name, f"{sr:.1f}", f"{avg_r:.4f}", f"{avg_s:.1f}", f"{avg_int:.1f}"])

    return sr, avg_r, avg_s, avg_int


def main():
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="TQC Hierarchical Model Evaluator (A* + TQC + MPC)")
    parser.add_argument("--env",        type=str,  default="all", choices=ALL_ENVS + ["all"])
    parser.add_argument("--episodes",   type=int,  default=10)
    parser.add_argument("--no-gif",     action="store_true")
    parser.add_argument("--checkpoint", type=str,  default=TQC_CKPT)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TB_DIR,  exist_ok=True)

    print("=" * 70)
    print("  MODEL 2: TQC HIERARCHICAL (A* Planner -> TQC Policy -> MPC Shield)")
    print(f"  Device     : {device.upper()}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Episodes   : {args.episodes} per environment")
    print(f"  Output Dir : {OUT_DIR}")
    print("=" * 70)

    agent = TQCAgent(device=device)
    if os.path.exists(args.checkpoint):
        agent.load(args.checkpoint)
        print(f"[LOAD] Loaded TQC checkpoint: {args.checkpoint}\n")
    else:
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    writer = SummaryWriter(log_dir=TB_DIR)
    eval_f = open(EVAL_CSV, "w", newline="", encoding="utf-8")
    csv_w  = csv.writer(eval_f)
    csv_w.writerow(["env", "success_rate_%", "avg_reward", "avg_steps", "avg_mpc_interventions"])

    envs_to_run = ALL_ENVS if args.env == "all" else [args.env]
    summary     = []

    for env_name in envs_to_run:
        print(f"\n[{envs_to_run.index(env_name)+1}/{len(envs_to_run)}] Evaluating: {env_name.upper()}")
        print("-" * 60)
        sr, avg_r, avg_s, avg_int = evaluate_env(
            env_name, agent, writer, csv_w,
            n_episodes=args.episodes, save_gifs=not args.no_gif
        )
        summary.append((env_name, sr, avg_r, avg_s, avg_int))

    eval_f.flush()
    eval_f.close()
    writer.close()

    overall_sr = float(np.mean([sr for _, sr, _, _, _ in summary]))

    print("\n" + "=" * 70)
    print("  MODEL 2: TQC HIERARCHICAL — FINAL RESULTS ACROSS 6 ENVIRONMENTS")
    print("=" * 70)
    print(f"{'Environment':18s} | {'Success Rate':14s} | {'Avg Steps':10s} | {'MPC Interventions':18s}")
    print("-" * 70)
    for env_name, sr, avg_r, avg_s, avg_int in summary:
        print(f"{env_name:18s} | {sr:5.1f}%         | {avg_s:8.0f}   | {avg_int:14.1f}")
    print("-" * 70)
    print(f"{'OVERALL AVERAGE':18s} | {overall_sr:5.1f}%")
    print("=" * 70)
    print(f"\n[DONE] Eval CSV saved: {EVAL_CSV}")
    print(f"[DONE] GIFs saved:     {GIF_DIR}")


if __name__ == "__main__":
    main()

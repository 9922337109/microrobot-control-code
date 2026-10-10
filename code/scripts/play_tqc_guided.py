import os
import sys
import torch
import numpy as np
import cv2
import warnings
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environments import MicrorobotEnvGameByTheWall
from utils.tqc_agent import TQCAgent
from scripts.play_guided_hybrid import WaypointGuidedWrapper, MPCSafetyShield, frames_to_gif_bytes

# ── 6 Environments ───────────────────────────────────────────────────────────
ALL_ENVS = ["vascular", "capillary", "veins", "vascular_large", "racetrack", "maze_3"]

_LOG_DIRS = {env: f"results/play_logs_tqc/{env}" for env in ALL_ENVS}
_ENV_YAML = "results/play_logs/env_config_default_segmentation_vascular_large.yaml"

_env_base = {
    "name": (0, 0), "config": _ENV_YAML, "max_envs": 1,
    "render_mode": "rgb_array", "subepisode_sampling": True,
    "subepisode_length": 2, "reduce_radius": -0.01,
}

ENV_CONFIGS = {
    "vascular":       {**_env_base, "image_string": "binary_images/default_segmentation_vascular.png"},
    "capillary":      {**_env_base, "image_string": "binary_images/capillary.png"},
    "veins":          {**_env_base, "image_string": "binary_images/veins.png"},
    "vascular_large": {**_env_base, "image_string": "binary_images/default_segmentation_vascular_large.png"},
    "racetrack":      {**_env_base, "image_string": "binary_images/default_mask_racetrack_segmented.png", "subepisode_sampling": False},
    "maze_3":         {**_env_base, "image_string": "binary_images/maze_3.png", "subepisode_sampling": False},
}

TQC_CKP_PATH = "results/logs/tqc_model/tqc_checkpoint.pth"

def evaluate_tqc_environment(env_name, agent):
    log_base = _LOG_DIRS[env_name]
    tb_dir   = f"{log_base}/tensorboard"
    os.makedirs(log_base, exist_ok=True)
    os.makedirs(tb_dir,   exist_ok=True)

    env_cfg = ENV_CONFIGS[env_name]
    raw_env = MicrorobotEnvGameByTheWall(**env_cfg)
    
    mpc_shield = MPCSafetyShield(raw_env)
    env        = WaypointGuidedWrapper(mpc_shield, env_cfg['image_string'], env_name=env_name)

    writer  = SummaryWriter(log_dir=tb_dir)
    results = []

    for ep in range(1, 11):
        ret = env.reset()
        obs = ret[0] if isinstance(ret, tuple) else ret
        info = ret[1] if isinstance(ret, tuple) and len(ret) > 1 else {}
        
        done = False
        total_reward = 0.0
        steps = 0
        frames = []

        while not done:
            img = obs["image"]
            vec = np.concatenate([obs["agent_position"], obs["target_position"]])
            
            # Select action from TQC Agent
            action_idx = agent.select_action(img, vec, deterministic=True)
            
            # Estimate epistemic uncertainty to scale MPC safety threshold dynamically
            uncertainty = agent.estimate_uncertainty(img, vec)
            
            # Step environment
            step_ret = env.step(action_idx)
            if len(step_ret) == 5:
                obs, reward, terminated, truncated, info = step_ret
                done = terminated or truncated
            else:
                obs, reward, done, info = step_ret

            total_reward += reward
            steps += 1
            
            # Record frame for GIF
            if isinstance(obs, dict) and "image" in obs:
                frames.append(obs["image"])

        interventions = mpc_shield.interventions
        success = (total_reward > 0)
        results.append({
            "episode": ep, "reward": total_reward, "steps": steps,
            "interventions": interventions, "success": success
        })

        status = "SUCCESS" if success else "TIMEOUT"
        print(f"  [TQC-{env_name.upper()}] Ep {ep:2d}/10 [{status:7s}] | Reward: {total_reward:7.2f} | "
              f"Steps: {steps:4d} | Shield Interventions: {interventions:3d}")

        writer.add_scalar("eval/episode_reward",       total_reward,  ep)
        writer.add_scalar("eval/episode_length",       steps,         ep)
        writer.add_scalar("eval/shield_interventions", interventions, ep)

        if frames:
            _, pil_frames = frames_to_gif_bytes(frames, fps=20)
            gif_path = os.path.join(log_base, f"tqc_episode_{ep}.gif")
            pil_frames[0].save(
                gif_path, format="GIF", save_all=True,
                append_images=pil_frames[1:], loop=0, duration=50,
            )

        writer.flush()

    env.close()
    writer.close()

    successes = sum(1 for r in results if r["success"])
    avg_steps = np.mean([r["steps"] for r in results])
    avg_inter = np.mean([r["interventions"] for r in results])
    sr = (successes / len(results)) * 100.0 if results else 0.0

    print(f"  --> [TQC-{env_name.upper()}] Complete | Success Rate: {sr:.1f}% | Avg Steps: {avg_steps:.0f} | Avg Interventions: {avg_inter:.1f}\n")
    return sr, avg_steps, avg_inter

def main():
    warnings.filterwarnings("ignore")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print(f"  TQC MODEL (PAPER 4) EVALUATION -- 6 ENVIRONMENTS (Device: {device.upper()})")
    print("=" * 70)

    agent = TQCAgent(device=device)
    if os.path.exists(TQC_CKP_PATH):
        print(f"[LOAD] Loading TQC Checkpoint: {TQC_CKP_PATH}")
        agent.load(TQC_CKP_PATH)
    else:
        print(f"[WARNING] Checkpoint not found at {TQC_CKP_PATH}. Evaluating with initial weights.")

    summary_table = []
    for env_name in ALL_ENVS:
        print(f"[{ALL_ENVS.index(env_name)+1}/6] Benchmarking TQC Environment: {env_name.upper()}")
        sr, steps, inter = evaluate_tqc_environment(env_name, agent)
        summary_table.append((env_name, sr, steps, inter))

    print("\n" + "=" * 70)
    print("  CONSOLIDATED TQC MODEL EVALUATION RESULTS")
    print("=" * 70)
    print(f"{'Environment':<18} | {'TQC Success Rate':<18} | {'Avg Steps':<12} | {'Avg Interventions':<20}")
    print("-" * 70)
    for env_name, sr, steps, inter in summary_table:
        print(f"{env_name:<18} | {sr:5.1f}%             | {steps:8.0f}     | {inter:14.1f}")
    
    overall_sr = np.mean([sr for _, sr, _, _ in summary_table])
    print("-" * 70)
    print(f"{'TQC OVERALL AVERAGE':<18} | {overall_sr:5.1f}%             |")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()

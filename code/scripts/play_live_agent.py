"""
play_live_agent.py
===================
Visual live player for the trained TQC + SuANN agent (tqc_best.pth).
Renders the PyGame window live on screen so you can watch the microrobot move in real-time.

Usage:
    python scripts/play_live_agent.py --env veins
    python scripts/play_live_agent.py --env vascular
    python scripts/play_live_agent.py --env maze_3
    python scripts/play_live_agent.py --env all
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environments import MicrorobotEnvGameByTheWall
from utils.tqc_agent import TQCAgent
from scripts.play_guided_hybrid import WaypointGuidedWrapper, MPCSafetyShield

from PIL import Image as PILImage

_ENV_YAML = "results/play_logs/env_config_default_segmentation_vascular_large.yaml"
GIF_DIR   = "results/play_gifs"

ALL_ENVS = ["veins", "vascular", "capillary", "vascular_large", "racetrack", "maze_3"]

ENV_CONFIGS = {
    "vascular":       {"image_string": "binary_images/default_segmentation_vascular.png",       "subepisode_sampling": True},
    "capillary":      {"image_string": "binary_images/capillary.png",                            "subepisode_sampling": True},
    "veins":          {"image_string": "binary_images/veins.png",                                "subepisode_sampling": True},
    "vascular_large": {"image_string": "binary_images/default_segmentation_vascular_large.png", "subepisode_sampling": True},
    "racetrack":      {"image_string": "binary_images/default_mask_racetrack_segmented.png",   "subepisode_sampling": False},
    "maze_3":         {"image_string": "binary_images/maze_3.png",                             "subepisode_sampling": False},
}


def save_episode_gif(frames, env_name: str, ep: int):
    """Save captured RGB image frames as an animated GIF."""
    if not frames:
        return
    os.makedirs(GIF_DIR, exist_ok=True)
    pil_frames = [PILImage.fromarray(f) for f in frames]
    gif_path = os.path.join(GIF_DIR, f"{env_name}_episode_{ep}.gif")
    pil_frames[0].save(
        gif_path,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=50,
    )
    print(f"  [GIF SAVED] {gif_path}")


def play_env(env_name: str, agent: TQCAgent, episodes: int = 3, guided: bool = True, step_delay: float = 0.03):
    print(f"\n{'='*70}")
    print(f"  PLAYING LIVE: [{env_name.upper()}]  |  Guided (A*+MPC): {guided}  |  Episodes: {episodes}")
    print(f"{'='*70}")

    env_cfg = ENV_CONFIGS[env_name]
    raw_env = MicrorobotEnvGameByTheWall(
        name=(0, 0),
        config=_ENV_YAML,
        max_envs=1,
        render_mode="human",          # Live PyGame window render
        subepisode_sampling=env_cfg["subepisode_sampling"],
        subepisode_length=2,
        reduce_radius=-0.01,
        image_string=env_cfg["image_string"],
    )

    if guided:
        mpc_shield = MPCSafetyShield(raw_env)
        env        = WaypointGuidedWrapper(mpc_shield, env_cfg['image_string'], env_name=env_name)
    else:
        env = raw_env

    for ep in range(1, episodes + 1):
        ret = env.reset()
        obs = ret[0] if isinstance(ret, tuple) else ret
        done = False
        total_r = 0.0
        steps = 0
        frames = []

        print(f"\n--- Episode {ep}/{episodes} ({env_name.upper()}) ---")

        while not done:
            if isinstance(obs, dict) and "image" in obs:
                frames.append(obs["image"])

            img = obs["image"]
            vec = np.concatenate([obs["agent_position"], obs["target_position"]]).astype(np.float32)

            action_idx = agent.select_action(img, vec, deterministic=True)
            step_ret   = env.step(action_idx)

            if len(step_ret) == 5:
                obs, reward, terminated, truncated, info = step_ret
                done = terminated or truncated
            else:
                obs, reward, done, info = step_ret

            total_r += reward
            steps   += 1

            if step_delay > 0:
                time.sleep(step_delay)

        status = "SUCCESS (Target Reached!)" if total_r > 0 else "TIMEOUT"
        print(f"  Result: {status:28s} | Reward: {total_r:7.2f} | Steps: {steps}")
        save_episode_gif(frames, env_name, ep)

    env.close()


def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Live TQC Agent Visualizer")
    parser.add_argument("--env", type=str, default="veins", choices=ALL_ENVS + ["all"],
                        help="Environment to play (default: veins)")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes per environment")
    parser.add_argument("--checkpoint", type=str, default="results/logs/tqc_v3_online/tqc_best.pth",
                        help="Path to TQC checkpoint file")
    parser.add_argument("--no-guided", action="store_true", help="Disable A* waypoints + MPC safety shield")
    parser.add_argument("--delay", type=float, default=0.03, help="Step delay in seconds for visualization")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print(f"  LIVE TQC AGENT VISUALIZER")
    print(f"  Device    : {device.upper()}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Environment: {args.env.upper()}")
    print("=" * 70)

    agent = TQCAgent(device=device)
    if os.path.exists(args.checkpoint):
        agent.load(args.checkpoint)
        print(f"[LOAD] Successfully loaded model checkpoint: {args.checkpoint}")
    else:
        print(f"[ERROR] Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)

    guided = not args.no_guided

    if args.env == "all":
        for env_name in ALL_ENVS:
            play_env(env_name, agent, episodes=args.episodes, guided=guided, step_delay=args.delay)
    else:
        play_env(args.env, agent, episodes=args.episodes, guided=guided, step_delay=args.delay)


if __name__ == "__main__":
    main()

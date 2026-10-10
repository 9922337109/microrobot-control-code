import os
import sys
import numpy as np
import cv2

# Ensure project root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environments import MicrorobotEnvGameByTheWall
from scripts.play_guided_hybrid import WaypointGuidedWrapper, MPCSafetyShield

DATA_DIR = "results/tqc_data"
DATASET_PATH = os.path.join(DATA_DIR, "tqc_expert_dataset.npz")

ENV_CONFIGS = {
    "vascular":       {"image_string": "binary_images/default_segmentation_vascular.png"},
    "capillary":      {"image_string": "binary_images/capillary.png"},
    "veins":          {"image_string": "binary_images/veins.png"},
    "vascular_large": {"image_string": "binary_images/default_segmentation_vascular_large.png"},
    "racetrack":      {"image_string": "binary_images/default_mask_racetrack_segmented.png"},
    "maze_3":         {"image_string": "binary_images/maze_3.png"},
}

_ENV_YAML = "results/play_logs/env_config_default_segmentation_vascular_large.yaml"

def collect_expert_data(total_steps=100000):
    """
    Collects 100,000 expert navigation transition steps across all 6 environments
    using A* Global Planner + MPC Safety Shield to ensure high reward dataset quality.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    images = []
    vectors = []
    actions = []
    rewards = []
    next_images = []
    next_vectors = []
    dones = []

    steps_per_env = total_steps // len(ENV_CONFIGS)
    print("=" * 70)
    print(f"  COLLECTING HIGH-QUALITY EXPERT DATASET ({total_steps} STEPS TOTAL)")
    print("     Method: A* Global Planner + MPC Safety Shield Demonstration")
    print("=" * 70)

    for env_name, cfg in ENV_CONFIGS.items():
        env_cfg = {
            "name": (0, 0), "config": _ENV_YAML, "max_envs": 1,
            "render_mode": "rgb_array", "subepisode_sampling": True,
            "subepisode_length": 2, "reduce_radius": -0.01,
            "image_string": cfg["image_string"]
        }
        
        raw_env = MicrorobotEnvGameByTheWall(**env_cfg)
        mpc_shield = MPCSafetyShield(raw_env)
        env = WaypointGuidedWrapper(mpc_shield, env_cfg['image_string'], env_name=env_name)

        ret = env.reset()
        obs = ret[0] if isinstance(ret, tuple) else ret
        
        env_collected = 0
        while env_collected < steps_per_env:
            # Generate guided action using A* waypoint directional vector
            try:
                target_wp = obs["target_position"]  # [x_norm, y_norm]
                agent_pos = obs["agent_position"]   # [x_norm, y_norm]
                diff = target_wp - agent_pos
                
                # Convert directional angle to nearest discrete action index (0..7)
                angle = np.arctan2(diff[1], diff[0])
                action = int(round(angle / (2 * np.pi / 8))) % 8
            except Exception:
                action = env.action_space.sample()

            img = obs["image"]
            vec = np.concatenate([obs["agent_position"], obs["target_position"]])

            step_ret = env.step(action)
            if len(step_ret) == 5:
                next_obs, reward, terminated, truncated, _ = step_ret
                done = terminated or truncated
            else:
                next_obs, reward, done, _ = step_ret

            next_img = next_obs["image"]
            next_vec = np.concatenate([next_obs["agent_position"], next_obs["target_position"]])

            images.append(img)
            vectors.append(vec)
            actions.append(action)
            rewards.append(reward)
            next_images.append(next_img)
            next_vectors.append(next_vec)
            dones.append(float(done))

            env_collected += 1
            if done:
                ret = env.reset()
                obs = ret[0] if isinstance(ret, tuple) else ret
            else:
                obs = next_obs

        env.close()
        print(f"  [OK] Collected {env_collected} expert steps for environment: {env_name.upper()}")

    print("=" * 70)
    print(f"  Compressing and saving dataset ({len(images)} total transitions)...")
    np.savez_compressed(
        DATASET_PATH,
        images=np.array(images, dtype=np.uint8),
        vectors=np.array(vectors, dtype=np.float32),
        actions=np.array(actions, dtype=np.int64),
        rewards=np.array(rewards, dtype=np.float32),
        next_images=np.array(next_images, dtype=np.uint8),
        next_vectors=np.array(next_vectors, dtype=np.float32),
        dones=np.array(dones, dtype=np.float32)
    )
    print(f"[SAVE SUCCESS] Dataset saved to: {DATASET_PATH}")
    print("=" * 70)

if __name__ == "__main__":
    collect_expert_data(total_steps=100000)

import sys
import os
import torch
import numpy as np
import cv2
import gymnasium as gym
import imageio
from stable_baselines3 import PPO
from torch.utils.tensorboard import SummaryWriter

# --- MONKEY PATCH FOR HEADLESS WSL ---
def dummy_imshow(window_name, image): pass
cv2.imshow = dummy_imshow

# Add the project root directory to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environments import MicrorobotEnvGameByTheWall
from environments.costum_wrappers.RateTargetReached import RateTargetReachedWrapper

# --- 1. MAPPING ENVIRONMENTS ---
TEST_SUITE = {
    "vascular": "default_segmentation_vascular.png",
    "capillary_test": "capillary.png",
    "aml_final_video": "veins.png",
    "4squares": "default_segmentation_vascular_large.png",
    "racetrack": "simple.png",
    "maze_3_test": "maze_3.png"
}

MODEL_PATH = "results/ppo_veins_baseline/ppo_veins_final"

# --- 2. COMPATIBILITY WRAPPER ---
class GymAdapterWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs_data = self.env.reset(**kwargs)
        if isinstance(obs_data, tuple) and len(obs_data) == 2:
            return obs_data
        return obs_data, {}

    def step(self, action):
        step_data = self.env.step(action)
        if len(step_data) == 5:
            return step_data
        obs, reward, done, info = step_data
        return obs, reward, done, False, info

def run_ppo_test(env_name, img_file):
    print(f"\n🚀 Running Baseline 2 (PPO) | Env: {env_name}")
    
    img_path = os.path.join('binary_images', img_file)
    output_dir = f"results/play_logs_{env_name}/ppo_baseline"
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=output_dir)

    env_config = {
        "image_string": img_path,
        "config": "results/logs/vascular_large_collision_smaller_bubbles_game/env_config_default_segmentation_vascular_large.yaml",
        "reduce_radius": -0.01,
        "render_mode": "rgb_array",
        "max_envs": 1
    }

    # Setup Environment
    env = MicrorobotEnvGameByTheWall(**env_config)
    env = RateTargetReachedWrapper(env, 100)
    env = GymAdapterWrapper(env)

    # Load Model
    model = PPO.load(MODEL_PATH, device="cpu")

    obs, info = env.reset()
    frames = []
    total_reward = 0

    for i in range(500):
        # Determine the action
        action, _ = model.predict(obs, deterministic=True)
        
        # Take the step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Capture Frame for GIF/TB
        frame = None
        if isinstance(obs, dict) and 'image' in obs:
            frame = obs['image']
        elif isinstance(obs, np.ndarray) and obs.ndim == 3:
            frame = obs

        if frame is not None:
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
            frames.append(frame)
            
            if i % 25 == 0:
                img_tensor = torch.from_numpy(frame).permute(2, 0, 1)
                writer.add_image(f'Visuals/PPO_{env_name}', img_tensor, i)

        writer.add_scalar("Rollout/Reward", reward, i)
        writer.add_scalar("Rollout/Cumulative_Reward", total_reward, i)

        if done: break

    if frames:
        imageio.mimsave(os.path.join(output_dir, f"ppo_{env_name}.gif"), frames, fps=30)
        print(f"✅ GIF/TB Images saved for {env_name}")

    print(f"✅ Finished {env_name}. Total Reward: {total_reward:.2f}")
    writer.close()
    env.close()

def main():
    if not os.path.exists(MODEL_PATH + ".zip"):
        print(f"❌ Error: PPO model not found at {MODEL_PATH}")
        return

    for env_label, img_filename in TEST_SUITE.items():
        if os.path.exists(os.path.join('binary_images', img_filename)):
            run_ppo_test(env_label, img_filename)

if __name__ == '__main__':
    main()

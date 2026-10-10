import sys
import os
import torch
import numpy as np
import cv2

# --- MONKEY PATCH FOR HEADLESS WSL ---
def dummy_imshow(window_name, image): pass
cv2.imshow = dummy_imshow

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
from environments import MicrorobotEnvGameByTheWall
from environments.costum_wrappers.RateTargetReached import RateTargetReachedWrapper
import imageio
from torch.utils.tensorboard import SummaryWriter

# --- 1. MAPPING ENVIRONMENTS ---
TEST_SUITE = {
    "vascular": "default_segmentation_vascular.png",
    "capillary_test": "capillary.png",
    "aml_final_video": "veins.png",
    "4squares": "default_segmentation_vascular_large.png",
    "racetrack": "simple.png",
    "maze_3_test": "maze_3.png"
}

# --- 2. MPC SETTINGS ---
HORIZON = 5
NUM_SAMPLES = 50
STEP_SIZE = 0.02 

def get_mpc_action(env):
    best_action = None
    best_score = -float('inf')
    try:
        start_pos = env.unwrapped.agent_location.copy()
        target_pos = env.unwrapped.target_location
    except AttributeError:
        start_pos = env.unwrapped.agent_position.copy()
        target_pos = env.unwrapped.target_position
    
    for _ in range(NUM_SAMPLES):
        action_seq = np.random.randint(0, 8, size=HORIZON) 
        sim_pos = start_pos.copy()
        cumulative_reward = 0
        for action_idx in action_seq:
            if hasattr(env.unwrapped, '_action_to_direction'):
                direction = env.unwrapped._action_to_direction[action_idx]
            else:
                angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
                direction = np.array([np.cos(angles[action_idx]), np.sin(angles[action_idx])])

            next_pos = sim_pos + (direction * STEP_SIZE)
            hit = env.unwrapped._is_collided(next_pos) if hasattr(env.unwrapped, '_is_collided') else False

            if hit:
                cumulative_reward -= 1000 
                break
            else:
                dist_old = np.linalg.norm(sim_pos - target_pos)
                dist_new = np.linalg.norm(next_pos - target_pos)
                cumulative_reward += (dist_old - dist_new) * 100
            sim_pos = next_pos

        if cumulative_reward > best_score:
            best_score = cumulative_reward
            best_action = action_seq[0]
    return best_action if best_action is not None else env.action_space.sample()

def run_test(env_name, img_file):
    print(f"\n🚀 Running Baseline 1 (MPC) | Env: {env_name}")
    img_path = os.path.join('binary_images', img_file)
    output_dir = f"results/play_logs_{env_name}/mpc_baseline"
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=output_dir)

    env_config = {
        "image_string": img_path,
        "config": "results/logs/vascular_large_collision_smaller_bubbles_game/env_config_default_segmentation_vascular_large.yaml",
        "reduce_radius": -0.01,
        "render_mode": "rgb_array",
        "max_envs": 1
    }

    env = MicrorobotEnvGameByTheWall(**env_config)
    env = RateTargetReachedWrapper(env, 100)
    
    reset_return = env.reset()
    obs = reset_return[0] if isinstance(reset_return, tuple) else reset_return

    frames = []
    total_reward = 0

    for i in range(500):
        # --- CAPTURE IMAGE FROM OBSERVATION ---
        frame = None
        if isinstance(obs, dict) and 'image' in obs:
            frame = obs['image']
        elif isinstance(obs, np.ndarray) and obs.ndim == 3:
            frame = obs
        
        if frame is not None:
            # Ensure frame is in uint8 [0-255] for GIF/TB
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
            
            frames.append(frame)
            if i % 50 == 0:
                img_tensor = torch.from_numpy(frame).permute(2, 0, 1)
                writer.add_image(f'Visuals/{env_name}', img_tensor, i)

        action = get_mpc_action(env)
        step_result = env.step(action)
        
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result

        total_reward += reward
        writer.add_scalar("Rollout/Reward", reward, i)
        writer.add_scalar("Rollout/Cumulative_Reward", total_reward, i)
        
        if done: break

    if frames:
        imageio.mimsave(os.path.join(output_dir, f"mpc_{env_name}.gif"), frames, fps=30)
        print(f"✅ GIF and TB Images generated for {env_name}")
    else:
        print(f"⚠️ Warning: Still no frames found in 'obs'.")

    print(f"✅ Finished {env_name}. Total Reward: {total_reward:.2f}")
    writer.close()
    env.close()

def main():
    for env_label, img_filename in TEST_SUITE.items():
        if os.path.exists(os.path.join('binary_images', img_filename)):
            run_test(env_label, img_filename)

if __name__ == '__main__':
    main()

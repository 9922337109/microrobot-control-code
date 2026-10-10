import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from environments import MicrorobotEnvGameByTheWall
from environments.costum_wrappers.RateTargetReached import RateTargetReachedWrapper
import os

# --- 1. CONFIGURATION ---
LOGDIR = "results/ppo_veins_baseline"
os.makedirs(LOGDIR, exist_ok=True)

env_config = {
    "image_string": 'binary_images/default_segmentation_vascular.png',
    "config": "results/logs/vascular_large_collision_smaller_bubbles_game/env_config_default_segmentation_vascular_large.yaml",
    "reduce_radius": -0.01,
    "name": (0, 0),
    "render_mode": "rgb_array",
    "max_envs": 1,
    "subepisode_sampling": True,
    "subepisode_length": 2
}

# --- 2. THE FIX: COMPATIBILITY WRAPPER ---
class GymAdapterWrapper(gym.Wrapper):
    """
    Translates Old Gym API (env returns just obs) 
    to New Gymnasium API (env returns obs, info)
    Required for Stable Baselines 3.
    """
    def __init__(self, env):
        super().__init__(env)
    
    def reset(self, **kwargs):
        # Call the old reset
        obs_data = self.env.reset(**kwargs)
        
        # If it returns a tuple of 2, it's already correct. 
        # If it returns a list/array (image), we must wrap it.
        if isinstance(obs_data, tuple) and len(obs_data) == 2:
            return obs_data
        else:
            # Old API -> New API: Return (obs, empty_info_dict)
            return obs_data, {}

    def step(self, action):
        # Call the old step
        step_data = self.env.step(action)
        
        # Check if we got 4 values (Old) or 5 values (New)
        if len(step_data) == 5:
            return step_data
        else:
            # Old API: obs, reward, done, info
            obs, reward, done, info = step_data
            # New API: obs, reward, terminated, truncated, info
            # We map 'done' to 'terminated' and set 'truncated' to False
            return obs, reward, done, False, info

# --- 3. ENVIRONMENT FACTORY ---
def make_env():
    # 1. Load your custom environment
    env = MicrorobotEnvGameByTheWall(**env_config)
    # 2. Add your custom stats wrapper
    env = RateTargetReachedWrapper(env, 100)
    # 3. Apply the API Adapter (Crucial Fix)
    env = GymAdapterWrapper(env)
    return env

# --- 4. MAIN TRAINING LOOP ---
def main():
    print("------------------------------------------------")
    print("   STARTING PPO TRAINING (Model-Free Baseline)  ")
    print("------------------------------------------------")

    # Create 1 Vectorized Environment
    env = make_vec_env(make_env, n_envs=1)

    # Define PPO Agent
    model = PPO(
        "MultiInputPolicy",  # Handles Dictionary Observations (Image + Pos)
        env, 
        verbose=1, 
        tensorboard_log=LOGDIR,
        learning_rate=3e-4,
        batch_size=64,
        n_steps=2048,
        policy_kwargs={
            "net_arch": [256, 256], 
        }
    )

    # Setup Checkpointing
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"{LOGDIR}/checkpoints/",
        name_prefix="ppo_veins"
    )

    # Train for 100k steps
    model.learn(total_timesteps=100000, callback=checkpoint_callback)

    # Save
    model.save(f"{LOGDIR}/ppo_veins_final")
    print("Training Finished! Model saved.")

if __name__ == '__main__':
    main()
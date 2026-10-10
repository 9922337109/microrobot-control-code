import sys
import os
import warnings
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dreamerv3
from dreamerv3 import embodied
from environments import MicrorobotEnvGameByTheWall
from environments.costum_wrappers.MaxandSkip import MaxAndSkipEnv
from environments.costum_wrappers.RateTargetReached import RateTargetReachedWrapper
from dreamerv3.embodied.envs import from_gym

# --- CONFIGURATION ---
CKP_PATH = "results/logs/aml_finetuning/checkpoint.ckpt"
BASE_LOGDIR = 'results/play_logs'
NUM_EPISODES = 5  # Reduced to 5 for faster verification

TEST_SUITE = {
    "vascular": "default_segmentation_vascular.png",
    "capillary_test": "capillary.png",
    "aml_final_video": "veins.png",
    "4squares": "default_segmentation_vascular_large.png",
    "racetrack": "simple.png",
    "maze_3_test": "maze_3.png"
}

def main():
    if len(sys.argv) < 2:
        print("Error: No environment name provided.")
        return
    
    env_name = sys.argv.pop(1)
    if env_name not in TEST_SUITE:
        print(f"Error: {env_name} not in TEST_SUITE.")
        return
    
    img_file = TEST_SUITE[env_name]
    log_path = f"{BASE_LOGDIR}_{env_name}/dreamer_world_model"
    os.makedirs(log_path, exist_ok=True)

    print(f"\n🚀 STARTING EVALUATION: {env_name}")
    
    # 1. Setup Environment
    env_config = {
        "name": (0,0),
        "config": "results/logs/vascular_large_collision_smaller_bubbles_game/env_config_default_segmentation_vascular_large.yaml",
        "max_envs": 1, "render_mode": "rgb_array",
        "image_string": f'binary_images/{img_file}',
        "subepisode_sampling": True, "subepisode_length": 2, "reduce_radius": -0.01
    }
    
    env = MicrorobotEnvGameByTheWall(**env_config)
    env = MaxAndSkipEnv(env, 4)
    env = RateTargetReachedWrapper(env, 100, dreamer=True, verbose=0, logdir=f"{log_path}/tg_eval.csv")
    env = from_gym.FromGym(env)
    
    # 2. Setup Agent
    config = embodied.Config(dreamerv3.configs['defaults'])
    config = config.update(dreamerv3.configs['large'])
    config = config.update({
        'jax.platform': 'gpu', 
        'run.actor_batch': 1,
        'encoder.mlp_keys': 'agent_position|target_position',
        'decoder.mlp_keys': 'agent_position|target_position',
        'encoder.cnn_keys': 'image',
        'decoder.cnn_keys': 'image',
    })
    config = embodied.Flags(config).parse([])

    step = embodied.Counter()
    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
    
    # 3. Load Weights
    checkpoint = embodied.Checkpoint(CKP_PATH)
    checkpoint.agent = agent
    checkpoint.load(CKP_PATH)
    print(f"✅ Weights Loaded.")

    # 4. Custom Loop
    policy = lambda *args: agent.policy(*args, mode='eval')
    
    for ep in range(NUM_EPISODES):
        obs = env.reset()
        state = None 
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            act, state = policy(obs, state)
            obs, reward, done, info = env.step(act)
            total_reward += reward
            steps += 1
            if steps > 500: break 

        print(f"Episode {ep+1}: Steps={steps}, Return={total_reward:.2f}")

    env.close()
    print(f"✅ Finished {env_name}\n")

if __name__ == '__main__':
    main()

import sys
import os
import warnings
import dreamerv3
from dreamerv3 import embodied
import glob

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environments import MicrorobotEnvGameByTheWall
from environments.costum_wrappers.MaxandSkip import MaxAndSkipEnv
from environments.costum_wrappers.RateTargetReached import RateTargetReachedWrapper, RewardWrapper
from dreamerv3.embodied.envs import from_gym

# --- 1. SETTINGS ---
CKP_PATH = "results/logs/aml_finetuning"
BASE_LOGDIR = 'results/play_logs'

TEST_SUITE = {
    "vascular": "default_segmentation_vascular.png",
    "capillary_test": "capillary.png",
    "aml_final_video": "veins.png",
    "4squares": "default_segmentation_vascular_large.png",
    "racetrack": "simple.png",
    "maze_3_test": "maze_3.png"
}

def make_env(env_name, img_file, dreamer_config):
    log_path = f"{BASE_LOGDIR}_{env_name}/dreamer_world_model"
    os.makedirs(log_path, exist_ok=True)
    
    config = {
        "name": (0,0),
        "config": "results/logs/vascular_large_collision_smaller_bubbles_game/env_config_default_segmentation_vascular_large.yaml",
        "max_envs": 1,
        "render_mode": "rgb_array",
        "image_string": f'binary_images/{img_file}',
        "subepisode_sampling": True,
        "subepisode_length": 2,
        "reduce_radius": -0.01
    }
    
    env = MicrorobotEnvGameByTheWall(**config)
    env = MaxAndSkipEnv(env, 4)
    env = RateTargetReachedWrapper(env, 100, dreamer=True, verbose=0, logdir=f"{log_path}/tg_eval.csv")
    env = RewardWrapper(env, logdir=f"{log_path}/reward.csv")
    env = from_gym.FromGym(env)
    env = dreamerv3.wrap_env(env, dreamer_config)
    return env

def run_suite():
    warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')
    
    config = embodied.Config(dreamerv3.configs['defaults'])
    config = config.update(dreamerv3.configs['large'])
    config = config.update({
        'jax.platform': 'gpu',
        'run.actor_batch': 1,
        'encoder.mlp_keys': '$^',
        'decoder.mlp_keys': '$^',
        'encoder.cnn_keys': 'image',
        'decoder.cnn_keys': 'image',
    })
    config = embodied.Flags(config).parse()

    for env_name, img_file in TEST_SUITE.items():
        print(f"\n🚀 STARTING ADVANCED MODEL (WORLD MODELS): {env_name}")
        
        env_logdir = f"{BASE_LOGDIR}_{env_name}/dreamer_world_model"
        step = embodied.Counter()
        
        logger = embodied.Logger(step, [
            embodied.logger.TerminalOutput(),
            embodied.logger.TensorBoardOutput(env_logdir),
        ])
        
        env = embodied.BatchEnv([make_env(env_name, img_file, config)], parallel=False)
        agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
        
        checkpoint_file = f"{CKP_PATH}/checkpoint.ckpt"
        checkpoint = embodied.Checkpoint(checkpoint_file)
        checkpoint.step = step
        checkpoint.agent = agent
        checkpoint.load(checkpoint_file)
        
        # --- CRITICAL FIX: Reset step to 0 for evaluation ---
        step.set(0)
        
        args = embodied.Config(config.run, logdir=env_logdir)
        args = args.update({'eval_eps': 10})
        
        try:
            embodied.run.eval_only(agent, env, logger, args)
        except Exception as e:
            print(f"Evaluation of {env_name} finished.")
        
        env.close()

if __name__ == '__main__':
    run_suite()

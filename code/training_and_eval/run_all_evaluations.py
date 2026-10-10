try:
    import tf_keras
except ImportError:
    pass
import sys
import optax

class OptaxAliasFinder:
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith('optax.transforms'):
            import importlib.util
            return importlib.util.spec_from_loader(fullname, self)
        return None
    def create_module(self, spec):
        return optax
    def exec_module(self, module):
        pass

sys.meta_path.insert(0, OptaxAliasFinder())
sys.modules['optax.transforms'] = optax

import warnings
import io
import os
import numpy as np
import cv2
import heapq
import gymnasium as gym
from PIL import Image as PILImage
from torch.utils.tensorboard import SummaryWriter

import dreamerv3
from dreamerv3 import embodied
from dreamerv3.embodied.envs import from_gym

from environments import MicrorobotEnvGameByTheWall
from environments.costum_wrappers.MaxandSkip import MaxAndSkipEnv
from environments.costum_wrappers.RateTargetReached import RateTargetReachedWrapper, RewardWrapper

from scripts.play_guided_hybrid import WaypointGuidedWrapper, MPCSafetyShield, frames_to_gif_bytes

# ── List of all 6 environments to benchmark ──────────────────────────────────
ALL_ENVS = ["vascular", "capillary", "veins", "vascular_large", "racetrack", "maze_3"]

_LOG_DIRS = {
    "vascular":       "results/play_logs_eval_final/vascular",
    "capillary":      "results/play_logs_eval_final/capillary",
    "veins":          "results/play_logs_eval_final/veins",
    "vascular_large": "results/play_logs_eval_final/vascular_large",
    "racetrack":      "results/play_logs_eval_final/racetrack",
    "maze_3":         "results/play_logs_eval_final/maze_3",
}

_ENV_YAML = "results/play_logs/env_config_default_segmentation_vascular_large.yaml"

_env_base = {
    "name": (0, 0), "config": _ENV_YAML, "max_envs": 1,
    "render_mode": "rgb_array",
    "subepisode_sampling": True, "subepisode_length": 2, "reduce_radius": -0.01,
}

ENV_CONFIGS = {
    "vascular":       {**_env_base, "image_string": "binary_images/default_segmentation_vascular.png"},
    "capillary":      {**_env_base, "image_string": "binary_images/capillary.png"},
    "veins":          {**_env_base, "image_string": "binary_images/veins.png"},
    "vascular_large": {**_env_base, "image_string": "binary_images/default_segmentation_vascular_large.png"},
    "racetrack":      {**_env_base, "image_string": "binary_images/default_mask_racetrack_segmented.png", "subepisode_sampling": False},
    "maze_3":         {**_env_base, "image_string": "binary_images/maze_3.png", "subepisode_sampling": False},
}

CHECKPOINT_DIR = "results/logs/finetuned_paper_upgrades"

def evaluate_environment(env_name, agent, config):
    log_base = _LOG_DIRS[env_name]
    tb_dir   = f"{log_base}/tensorboard"
    os.makedirs(log_base, exist_ok=True)
    os.makedirs(tb_dir,   exist_ok=True)

    env_cfg = ENV_CONFIGS[env_name]
    raw_env = MicrorobotEnvGameByTheWall(**env_cfg)
    env     = MaxAndSkipEnv(raw_env, 4)

    mpc_shield = MPCSafetyShield(env)
    env        = mpc_shield

    env = WaypointGuidedWrapper(env, env_cfg['image_string'], env_name=env_name)
    env = RateTargetReachedWrapper(env, 100, dreamer=True, verbose=0, logdir=f"{log_base}/tg_eval.csv")
    env = RewardWrapper(env, logdir=f"{log_base}/reward.csv")
    env = from_gym.FromGym(env)
    env = dreamerv3.wrap_env(env, config)
    env = embodied.BatchEnv([env], parallel=False)

    writer     = SummaryWriter(log_dir=tb_dir)
    driver     = embodied.Driver(env)
    ep_counter = [1]
    results    = []

    def report(episode_data, env_index):
        ep            = ep_counter[0]
        rewards       = episode_data["reward"]
        total_reward  = float(sum(rewards))
        ep_length     = len(rewards)
        tg_rate       = float(np.mean(episode_data.get("target_reached", [0])))
        interventions = mpc_shield.interventions

        success = (ep_length < 1500 and total_reward > 0) or (tg_rate > 0.5)
        results.append({
            "episode": ep, "reward": total_reward, "steps": ep_length,
            "interventions": interventions, "success": success
        })

        status = "SUCCESS" if success else "TIMEOUT"
        print(f"  [{env_name.upper()}] Ep {ep:2d}/{10} [{status:7s}] | Reward: {total_reward:7.2f} | "
              f"Steps: {ep_length:4d} | Interventions: {interventions:3d}")

        writer.add_scalar("eval/episode_reward",       total_reward,  ep)
        writer.add_scalar("eval/episode_length",       ep_length,     ep)
        writer.add_scalar("eval/target_reached_rate",  tg_rate,       ep)
        writer.add_scalar("eval/shield_interventions", interventions, ep)

        if "image" in episode_data:
            frames = np.array(episode_data["image"])
            _, pil_frames = frames_to_gif_bytes(frames, fps=20)
            gif_path = os.path.join(log_base, f"guided_episode_{ep}.gif")
            pil_frames[0].save(
                gif_path, format="GIF", save_all=True,
                append_images=pil_frames[1:], loop=0, duration=50,
            )

        writer.flush()
        ep_counter[0] += 1

    driver.on_episode(report)
    policy = lambda *args: agent.policy(*args, mode="eval")
    driver(policy, episodes=10)
    env.close()
    writer.close()

    successes = sum(1 for r in results if r["success"])
    avg_steps = np.mean([r["steps"] for r in results])
    avg_inter = np.mean([r["interventions"] for r in results])
    sr = (successes / len(results)) * 100.0 if results else 0.0

    print(f"  --> {env_name.upper()} Complete | Success Rate: {sr:.1f}% | Avg Steps: {avg_steps:.0f} | Avg Interventions: {avg_inter:.1f}\n")
    return sr, avg_steps, avg_inter

def main():
    warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

    print("=" * 70)
    print("  FULL 6-ENVIRONMENT BENCHMARK EVALUATION")
    print("     Papers: 3 (Vascular Parkour), 4 (TQC+SuANN), 1 (GAP_SAC)")
    print("=" * 70)

    config = embodied.Config(dreamerv3.configs["defaults"])
    config = config.update(dreamerv3.configs["large"])
    config = config.update({
        "logdir":            CHECKPOINT_DIR,
        "run.train_ratio":   1,
        "jax.platform":      "cpu",   # JAX CUDA not installed — using cpu (PyTorch GPU separate)
        "encoder.mlp_keys":  "$^",
        "decoder.mlp_keys":  "$^",
        "encoder.cnn_keys":  "image",
        "decoder.cnn_keys":  "image",
    })
    config = embodied.Flags(config).parse([])
    step   = embodied.Counter()

    # Dummy env for agent initialization
    dummy_cfg = ENV_CONFIGS["vascular"]
    dummy_raw = MicrorobotEnvGameByTheWall(**dummy_cfg)
    dummy_env = MaxAndSkipEnv(dummy_raw, 4)
    dummy_env = MPCSafetyShield(dummy_env)
    dummy_env = WaypointGuidedWrapper(dummy_env, dummy_cfg['image_string'])
    dummy_env = RateTargetReachedWrapper(dummy_env, 100, dreamer=True, verbose=0)
    dummy_env = RewardWrapper(dummy_env)
    dummy_env = from_gym.FromGym(dummy_env)
    dummy_env = dreamerv3.wrap_env(dummy_env, config)
    dummy_env = embodied.BatchEnv([dummy_env], parallel=False)

    agent      = dreamerv3.Agent(dummy_env.obs_space, dummy_env.act_space, step, config)
    checkpoint = embodied.Checkpoint(f"{CHECKPOINT_DIR}/checkpoint.ckpt")
    checkpoint.agent = agent

    if os.path.exists(f"{CHECKPOINT_DIR}/checkpoint.ckpt"):
        print(f"[LOAD] Loading fine-tuned checkpoint: {CHECKPOINT_DIR}/checkpoint.ckpt")
        checkpoint.load(f"{CHECKPOINT_DIR}/checkpoint.ckpt", keys=["agent"])
    else:
        print(f"[INFO] Fine-tuned checkpoint not found at {CHECKPOINT_DIR}/checkpoint.ckpt. Using fresh weights.")

    dummy_env.close()

    summary_table = []

    for env_name in ALL_ENVS:
        print(f"\n[{ALL_ENVS.index(env_name)+1}/{len(ALL_ENVS)}] Benchmarking Environment: {env_name.upper()}")
        sr, steps, inter = evaluate_environment(env_name, agent, config)
        summary_table.append((env_name, sr, steps, inter))

    print("\n" + "=" * 70)
    print("  CONSOLIDATED 6-ENVIRONMENT BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Environment':<18} | {'Success Rate':<14} | {'Avg Steps':<12} | {'Avg Shield Interventions':<22}")
    print("-" * 70)

    baselines = {
        "vascular": "60%", "capillary": "30%", "veins": "30%",
        "vascular_large": "60%", "racetrack": "70%", "maze_3": "50%"
    }

    for env_name, sr, steps, inter in summary_table:
        base = baselines.get(env_name, "N/A")
        print(f"{env_name:<18} | {sr:5.1f}% (was {base}) | {steps:8.0f}     | {inter:14.1f}")

    overall_sr = np.mean([sr for _, sr, _, _ in summary_table])
    print("-" * 70)
    print(f"{'OVERALL AVERAGE':<18} | {overall_sr:5.1f}% (was 53%) |")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()

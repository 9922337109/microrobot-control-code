import warnings
import io
import os
import sys
import numpy as np
import cv2
import heapq
import gymnasium as gym

import dreamerv3
from dreamerv3 import embodied
from dreamerv3.embodied.envs import from_gym

from environments import MicrorobotEnvGameByTheWall
from environments.costum_wrappers.MaxandSkip import MaxAndSkipEnv
from environments.costum_wrappers.RateTargetReached import RateTargetReachedWrapper, RewardWrapper

# ── Import wrappers from play_guided_hybrid ─────────────────────────────────
from scripts.play_guided_hybrid import WaypointGuidedWrapper, MPCSafetyShield

# ── LOG AND CONFIGURATION SETTINGS ───────────────────────────────────────────
LOG_BASE = "results/logs/finetuned_paper_upgrades"
TB_DIR   = f"{LOG_BASE}/tensorboard"
ENV_YAML = "results/play_logs/env_config_default_segmentation_vascular_large.yaml"
IMAGE_STRING = "binary_images/default_segmentation_vascular_large.png"

def main():
    warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")
    os.makedirs(LOG_BASE, exist_ok=True)
    os.makedirs(TB_DIR,   exist_ok=True)

    print("=" * 60)
    print("  DREAMERV3 GPU FINE-TUNING -- 2025 RESEARCH PAPERS")
    print("     Phase 1: Dynamic Replanning + Stuck Escape + Adaptive MPC")
    print("     Phase 2: Yaw Alignment Reward + Wall Proximity Penalty")
    print("     Phase 3: Gated Attention Encoder Layer (nets.py)")
    print("=" * 60)

    # ── JAX GPU Configuration with CPU fallback ──────────────────────────────
    platform = "gpu"
    try:
        import jax
        available_devices = jax.devices("gpu")
        print(f"[GPU] Found {len(available_devices)} GPU device(s): {available_devices}")
    except Exception as e:
        platform = "cpu"
        print(f"[CPU] GPU unavailable ({e}), using CPU for fine-tuning.")

    config = embodied.Config(dreamerv3.configs["defaults"])
    config = config.update(dreamerv3.configs["large"])
    config = config.update({
        "logdir":            LOG_BASE,
        "run.train_ratio":   32,              # 32 environment steps per train step
        "run.steps":         50000,           # 50,000 fine-tuning steps
        "jax.platform":      platform,
        "encoder.mlp_keys":  "$^",
        "decoder.mlp_keys":  "$^",
        "encoder.cnn_keys":  "image",
        "decoder.cnn_keys":  "image",
        # Enable disagreement exploration (Paper 4 curiosity ensemble)
        "expl_behavior":     "Explore",
        "expl_rewards.disag": 0.1,
    })
    config = embodied.Flags(config).parse([])

    step = embodied.Counter()
    logdir = embodied.Path(LOG_BASE)

    # ── Logger Setup ─────────────────────────────────────────────────────────
    logger = embodied.Logger(step, [
        embodied.logger.TerminalOutput('.*'),
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        embodied.logger.JSONLOutput(logdir, 'scores.jsonl', 'episode/score'),
        embodied.logger.TensorBoardOutput(logdir),
    ])

    # ── Environment Pipeline ─────────────────────────────────────────────────
    env_cfg = {
        "name": (0, 0),
        "config": ENV_YAML,
        "image_string": IMAGE_STRING,
        "max_envs": 1,
        "render_mode": "rgb_array",
        "subepisode_sampling": True,
        "subepisode_length": 2,
        "reduce_radius": -0.01,
    }

    raw_env = MicrorobotEnvGameByTheWall(**env_cfg)   # includes Phase 2 reward
    env     = MaxAndSkipEnv(raw_env, 4)
    env     = MPCSafetyShield(env)                   # Phase 1 adaptive MPC
    env     = WaypointGuidedWrapper(env, IMAGE_STRING) # Phase 1 replanning + escape
    env     = RateTargetReachedWrapper(env, 100, dreamer=True, verbose=0, logdir=f"{LOG_BASE}/tg_eval.csv")
    env     = RewardWrapper(env, logdir=f"{LOG_BASE}/reward.csv")
    env     = from_gym.FromGym(env)
    env     = dreamerv3.wrap_env(env, config)
    env     = embodied.BatchEnv([env], parallel=False)

    # ── Replay Buffer & Agent ────────────────────────────────────────────────
    replay = embodied.replay.Uniform(
        config.batch_length, config.replay_size, logdir / 'replay'
    )
    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)

    # ── Checkpoint Management ────────────────────────────────────────────────
    checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
    checkpoint.step = step
    checkpoint.agent = agent
    checkpoint.replay = replay

    if (logdir / 'checkpoint.ckpt').exists():
        print(f"[LOAD] Resuming fine-tuning from {logdir / 'checkpoint.ckpt'}")
        checkpoint.load()
    else:
        print(f"[INIT] Starting fresh fine-tuning run at {LOG_BASE}")

    print("\n[TRAIN] Launching fine-tuning loop...")
    args = embodied.Config(
        **config.run, logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length
    )
    embodied.run.train(agent, env, replay, logger, args)

    print(f"\n{'='*60}")
    print(f"[COMPLETE] Fine-tuning finished. Model saved to {LOG_BASE}")
    print(f"[LOGS] View metrics: tensorboard --logdir {LOG_BASE}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

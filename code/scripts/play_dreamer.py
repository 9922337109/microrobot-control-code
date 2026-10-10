import warnings
import io
import os
import numpy as np
import imageio
import dreamerv3
from dreamerv3 import embodied
from dreamerv3.embodied.envs import from_gym
from PIL import Image as PILImage
from torch.utils.tensorboard import SummaryWriter
from environments import MicrorobotEnvGameByTheWall
from environments.costum_wrappers.MaxandSkip import MaxAndSkipEnv
from environments.costum_wrappers.RateTargetReached import RateTargetReachedWrapper, RewardWrapper

# ── SET YOUR CHOICE HERE: 'vascular' | 'capillary' | 'veins' | 'vascular_large' | 'racetrack' | 'maze_3' ──
CURRENT_ENV_NAME = "maze_3"

# ── Log directory map ─────────────────────────────────────────────────────────
_LOG_DIRS = {
    "vascular":       "results/play_logs_vascular/dreamer_world_model",
    "capillary":      "results/play_logs_capillary_test/dreamer_world_model",
    "veins":          "results/play_logs_aml_final_video/dreamer_world_model",
    "vascular_large": "results/play_logs/dreamer_world_model",
    "racetrack":      "results/play_logs_racetrack/dreamer_world_model",
    "maze_3":         "results/play_logs_maze_3_test/dreamer_world_model",
}
LOG_BASE = _LOG_DIRS[CURRENT_ENV_NAME]
TB_DIR   = f"{LOG_BASE}/tensorboard"
ckp      = "results/logs/aml_finetuning"

# ── Environment configs ───────────────────────────────────────────────────────
_ENV_YAML = "results/logs/vascular_large_collision_smaller_bubbles_game/env_config_default_segmentation_vascular_large.yaml"

env_config_vascular = {
    "name": (0, 0),
    "config": _ENV_YAML,
    "max_envs": 1,
    "render_mode": "rgb_array",
    "image_string": "binary_images/default_segmentation_vascular.png",
    "subepisode_sampling": True,
    "subepisode_length": 2,
}

env_config_capillary = {
    "name": (0, 0),
    "config": _ENV_YAML,
    "max_envs": 1,
    "render_mode": "rgb_array",
    "image_string": "binary_images/capillary.png",
    "subepisode_sampling": True,
    "subepisode_length": 2,
}

env_config_veins = {
    "name": (0, 0),
    "config": _ENV_YAML,
    "max_envs": 1,
    "render_mode": "rgb_array",
    "image_string": "binary_images/veins.png",
    "subepisode_sampling": True,
    "subepisode_length": 2,
}

# Vascular Large — default_segmentation_vascular_large.png → play_logs
env_config_vascular_large = {
    "name": (0, 0),
    "config": _ENV_YAML,
    "max_envs": 1,
    "render_mode": "rgb_array",
    "image_string": "binary_images/default_segmentation_vascular_large.png",
    "subepisode_sampling": True,
    "subepisode_length": 2,
}

# Racetrack — default_mask_racetrack_segmented_inv_dil.png → play_logs_racetrack
# subepisode_sampling=False: racetrack geometry can cause infinite search for valid positions
env_config_racetrack = {
    "name": (0, 0),
    "config": _ENV_YAML,
    "max_envs": 1,
    "render_mode": "rgb_array",
    "image_string": "binary_images/default_mask_racetrack_segmented.png",
    "subepisode_sampling": False,
    "subepisode_length": 2,
}

# Maze 3 — maze_3.png → play_logs_maze_3_test
env_config_maze_3 = {
    "name": (0, 0),
    "config": _ENV_YAML,
    "max_envs": 1,
    "render_mode": "rgb_array",
    "image_string": "binary_images/maze_3.png",
    "subepisode_sampling": False,
    "subepisode_length": 2,
}

ENV_CONFIGS = {
    "vascular":       env_config_vascular,
    "capillary":      env_config_capillary,
    "veins":          env_config_veins,
    "vascular_large": env_config_vascular_large,
    "racetrack":      env_config_racetrack,
    "maze_3":         env_config_maze_3,
}


def frames_to_gif_bytes(frames, fps=20):
    """Convert (T, H, W, C) uint8 array to an in-memory GIF (bytes)."""
    pil_frames = [PILImage.fromarray(f) for f in frames]
    buf = io.BytesIO()
    pil_frames[0].save(
        buf, format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=int(1000 / fps),
    )
    buf.seek(0)
    return buf, pil_frames


def main(env_cfg):
    warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")
    os.makedirs(LOG_BASE, exist_ok=True)
    os.makedirs(TB_DIR,   exist_ok=True)

    # ── DreamerV3 config ──────────────────────────────────────────────────────
    config = embodied.Config(dreamerv3.configs["defaults"])
    config = config.update(dreamerv3.configs["large"])
    config = config.update({
        "logdir": LOG_BASE,
        "run.train_ratio": 1,
        "jax.platform": "gpu",
        # Must match checkpoint (trained with CNN-only encoder, mlp_keys=$^)
        # deter=2048 + CNN_embed=8192 = 10240  ← matches saved obs_out kernel
        "encoder.mlp_keys": "$^",
        "decoder.mlp_keys": "$^",
        "encoder.cnn_keys": "image",
        "decoder.cnn_keys": "image",
    })
    config = embodied.Flags(config).parse([])

    step = embodied.Counter()

    # ── Environment ───────────────────────────────────────────────────────────
    raw_env = MicrorobotEnvGameByTheWall(**env_cfg)
    env = MaxAndSkipEnv(raw_env, 4)
    env = RateTargetReachedWrapper(env, 100, dreamer=True, verbose=1,
                                   logdir=f"{LOG_BASE}/tg_eval.csv")
    env = RewardWrapper(env, logdir=f"{LOG_BASE}/reward.csv")
    env = from_gym.FromGym(env)
    env = dreamerv3.wrap_env(env, config)
    env = embodied.BatchEnv([env], parallel=False)

    # ── Agent + checkpoint ────────────────────────────────────────────────────
    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
    checkpoint_path = f"{ckp}/checkpoint.ckpt"
    checkpoint = embodied.Checkpoint(checkpoint_path)
    checkpoint.agent = agent
    checkpoint.load(checkpoint_path, keys=["agent"])
    print("✅ Brain architecture aligned (deter=2048 + CNN_embed=8192 = 10240) and loaded.")

    # ── TensorBoard writer ────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=TB_DIR)
    print(f"📊 TensorBoard logs → {TB_DIR}")
    print(f"   Run:  tensorboard --logdir {TB_DIR}")

    # ── Evaluation loop ───────────────────────────────────────────────────────
    driver     = embodied.Driver(env)
    ep_counter = [1]

    def report(episode_data, env_index):
        ep           = ep_counter[0]
        rewards      = episode_data["reward"]
        total_reward = float(sum(rewards))
        ep_length    = len(rewards)
        mean_reward  = total_reward / max(ep_length, 1)
        tg_rate      = float(np.mean(episode_data.get("target_reached", [0])))

        print(f"✅ Episode {ep} | Reward: {total_reward:.2f} | "
              f"Steps: {ep_length} | TG rate: {tg_rate:.2%}")

        # Scalars
        writer.add_scalar("eval/episode_reward",      total_reward, ep)
        writer.add_scalar("eval/episode_length",      ep_length,    ep)
        writer.add_scalar("eval/mean_step_reward",    mean_reward,  ep)
        writer.add_scalar("eval/target_reached_rate", tg_rate,      ep)

        if "image" in episode_data:
            frames = np.array(episode_data["image"])   # (T, H, W, C) uint8

            # Image grid: 8 evenly-spaced frames → TensorBoard Images tab
            n        = min(8, len(frames))
            idx      = np.linspace(0, len(frames) - 1, n, dtype=int)
            grid     = frames[idx].transpose(0, 3, 1, 2)   # (N, C, H, W)
            writer.add_images(
                "eval/episode_frames",
                grid.astype(np.float32) / 255.0,
                global_step=ep,
                dataformats="NCHW",
            )

            # Save animated GIF to disk (no moviepy needed)
            gif_buf, pil_frames = frames_to_gif_bytes(frames, fps=20)
            gif_path = os.path.join(LOG_BASE, f"episode_{ep}.gif")
            pil_frames[0].save(
                gif_path, format="GIF",
                save_all=True,
                append_images=pil_frames[1:],
                loop=0,
                duration=int(1000 / 20),
            )
            print(f"💾 Saved GIF: {gif_path}")

        writer.flush()
        ep_counter[0] += 1

    driver.on_episode(report)
    policy = lambda *args: agent.policy(*args, mode="eval")

    print(f"▶️  Starting Evaluation for {CURRENT_ENV_NAME}...")
    driver(policy, episodes=10)
    env.close()
    writer.close()
    print(f"✅ Done. View results with:  tensorboard --logdir {TB_DIR}")


if __name__ == "__main__":
    main(ENV_CONFIGS[CURRENT_ENV_NAME])
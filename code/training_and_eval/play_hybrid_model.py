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
import gymnasium as gym
from PIL import Image as PILImage
from torch.utils.tensorboard import SummaryWriter

import dreamerv3
from dreamerv3 import embodied
from dreamerv3.embodied.envs import from_gym

from environments import MicrorobotEnvGameByTheWall
from environments.costum_wrappers.MaxandSkip import MaxAndSkipEnv
from environments.costum_wrappers.RateTargetReached import RateTargetReachedWrapper, RewardWrapper

# ── SET YOUR CHOICE HERE: 'vascular' | 'capillary' | 'veins' | 'vascular_large' | 'racetrack' | 'maze_3' ──
CURRENT_ENV_NAME = "maze_3"

# ── Log directory map (Saved in separate 'hybrid' folders to compare with baseline) ──
_LOG_DIRS = {
    "vascular":       "results/play_logs_vascular/hybrid_model",
    "capillary":      "results/play_logs_capillary_test/hybrid_model",
    "veins":          "results/play_logs_aml_final_video/hybrid_model",
    "vascular_large": "results/play_logs/hybrid_model",
    "racetrack":      "results/play_logs_racetrack/hybrid_model",
    "maze_3":         "results/play_logs_maze_3_test/hybrid_model",
}

LOG_BASE = _LOG_DIRS[CURRENT_ENV_NAME]
TB_DIR   = f"{LOG_BASE}/tensorboard"
ckp      = "results/logs/finetuned_paper_upgrades"

# ── Environment configs ───────────────────────────────────────────────────────
_ENV_YAML = "results/logs/vascular_large_collision_smaller_bubbles_game/env_config_default_segmentation_vascular_large.yaml"

env_config_vascular = {
    "name": (0, 0), "config": _ENV_YAML, "max_envs": 1, "render_mode": "rgb_array",
    "image_string": "binary_images/default_segmentation_vascular.png",
    "subepisode_sampling": True, "subepisode_length": 2,
}
env_config_capillary = {
    "name": (0, 0), "config": _ENV_YAML, "max_envs": 1, "render_mode": "rgb_array",
    "image_string": "binary_images/capillary.png",
    "subepisode_sampling": True, "subepisode_length": 2,
}
env_config_veins = {
    "name": (0, 0), "config": _ENV_YAML, "max_envs": 1, "render_mode": "rgb_array",
    "image_string": "binary_images/veins.png",
    "subepisode_sampling": True, "subepisode_length": 2,
}
env_config_vascular_large = {
    "name": (0, 0), "config": _ENV_YAML, "max_envs": 1, "render_mode": "rgb_array",
    "image_string": "binary_images/default_segmentation_vascular_large.png",
    "subepisode_sampling": True, "subepisode_length": 2,
}
env_config_racetrack = {
    "name": (0, 0), "config": _ENV_YAML, "max_envs": 1, "render_mode": "rgb_array",
    "image_string": "binary_images/default_mask_racetrack_segmented.png",
    "subepisode_sampling": False, "subepisode_length": 2,
}
env_config_maze_3 = {
    "name": (0, 0), "config": _ENV_YAML, "max_envs": 1, "render_mode": "rgb_array",
    "image_string": "binary_images/maze_3.png",
    "subepisode_sampling": False, "subepisode_length": 2,
}

ENV_CONFIGS = {
    "vascular":       env_config_vascular,
    "capillary":      env_config_capillary,
    "veins":          env_config_veins,
    "vascular_large": env_config_vascular_large,
    "racetrack":      env_config_racetrack,
    "maze_3":         env_config_maze_3,
}

# ── THE HYBRID SAFETY LAYER (MPC) ─────────────────────────────────────────────
class MPCSafetyShield(gym.Wrapper):
    """
    Intercepts the Action from Dreamer V3.
    If the action causes a collision, it overrides it with a safe one.
    """
    def __init__(self, env, max_steps=2000):
        super().__init__(env)
        self.interventions = 0
        self.ep_steps      = 0
        self.max_steps     = max_steps  # hard per-episode cutoff

    def reset(self, **kwargs):
        self.interventions = 0
        self.ep_steps      = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        # DreamerV3 outputs 0-indexed action (int or one-hot ndarray)
        if isinstance(action, np.ndarray) and action.shape == (8,):
            action_idx_0 = int(np.argmax(action))
        elif isinstance(action, (int, np.integer)):
            action_idx_0 = int(action)
        else:
            return self.env.step(action)

        safe_action = action
        u = self.env.unwrapped   # MicrorobotEnvGameByTheWall

        _has_mpc = (hasattr(u, '_action_to_direction') and
                    hasattr(u, 'agent_location') and
                    hasattr(u, '_fast_obstacles') and
                    hasattr(u, '_eval_pixel_collision'))
        if not _has_mpc:
            if not getattr(self, '_warned_no_mpc', False):
                print("⚠️  MPCSafetyShield: required env attributes missing — shield INACTIVE")
                self._warned_no_mpc = True
            return self.env.step(action)

        try:
            n_piezos  = u.config['Action_space_settings']['NUMBER_PIEZOS']
            act_num   = action_idx_0 % n_piezos
            env_key   = act_num + 1                       # env dict is 1-indexed

            direction = u._action_to_direction[env_key]

            # Use MINIMUM amplitude for the safety check so we only block truly
            # immediate wall hits (not just nearby-wall proximity at average step).
            ratio     = u.img_size / 64.0                 # = 8.0
            amplitude = u.min_ampl * ratio               # = 8px in 512-space

            # Predict next agent position
            predicted = u.agent_location + direction * amplitude
            px, py    = int(predicted[0]), int(predicted[1])

            # Safety radius = bubble radius (smallest valid check)
            r  = int(np.ceil(u.get_radius() + u.tolerance_collision))
            xs = np.arange(max(0, px - r), min(u.img_size, px + r + 1))
            ys = np.arange(max(0, py - r), min(u.img_size, py + r + 1))
            xx, yy = np.meshgrid(xs, ys)

            would_collide = u._eval_pixel_collision(xx, yy)

            if would_collide:
                # Sort alternative keys by how close they bring the agent to the target location
                alt_candidates = []
                for alt_key, alt_dir in u._action_to_direction.items():
                    if alt_key == env_key:
                        continue
                    alt_pred = u.agent_location + alt_dir * amplitude
                    dist_to_tgt = np.linalg.norm(alt_pred - u.target_location)
                    alt_candidates.append((dist_to_tgt, alt_key, alt_dir))
                
                # Sort by distance (closest first)
                alt_candidates.sort(key=lambda x: x[0])

                # Search for the first safe alternative direction (closest first)
                for dist_to_tgt, alt_key, alt_dir in alt_candidates:
                    alt_pred = u.agent_location + alt_dir * amplitude
                    apx, apy = int(alt_pred[0]), int(alt_pred[1])
                    axs = np.arange(max(0, apx - r), min(u.img_size, apx + r + 1))
                    ays = np.arange(max(0, apy - r), min(u.img_size, apy + r + 1))
                    axx, ayy = np.meshgrid(axs, ays)
                    if not u._eval_pixel_collision(axx, ayy):
                        alt_idx_0 = alt_key - 1
                        if isinstance(action, np.ndarray):
                            safe_action            = np.zeros_like(action)
                            safe_action[alt_idx_0] = 1.0
                        else:
                            safe_action = alt_idx_0
                        self.interventions += 1   # only count when action actually changes
                        break

        except Exception as ex:
            if not getattr(self, '_warned_ex', False):
                print(f"⚠️  MPCSafetyShield exception (once): {ex}")
                self._warned_ex = True

        ret = self.env.step(safe_action)

        # ── Hard per-episode step limit ────────────────────────────────────────
        self.ep_steps += 1
        if self.ep_steps >= self.max_steps:
            # Force episode termination regardless of env's own done signal
            if len(ret) == 5:
                obs, reward, terminated, truncated, info = ret
                return obs, reward, True, True, info
            else:
                obs, reward, done, info = ret
                return obs, reward, True, info

        return ret




# ── Robust wrapper-chain traversal ───────────────────────────────────────────
def find_shield(root_env):
    """Walk the wrapper chain and return the first MPCSafetyShield found."""
    env = root_env
    while env is not None:
        if isinstance(env, MPCSafetyShield):
            return env
        env = getattr(env, 'env', None)
    return None


def frames_to_gif_bytes(frames, fps=20):
    pil_frames = [PILImage.fromarray(f) for f in frames]
    buf = io.BytesIO()
    pil_frames[0].save(
        buf, format="GIF", save_all=True, append_images=pil_frames[1:],
        loop=0, duration=int(1000 / fps),
    )
    buf.seek(0)
    return buf, pil_frames

# ── MAIN EXECUTION ────────────────────────────────────────────────────────────
def main(env_cfg):
    warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")
    os.makedirs(LOG_BASE, exist_ok=True)
    os.makedirs(TB_DIR,   exist_ok=True)

    config = embodied.Config(dreamerv3.configs["defaults"])
    config = config.update(dreamerv3.configs["large"])
    config = config.update({
        "logdir": LOG_BASE,
        "run.train_ratio": 1,
        "jax.platform": "gpu",
        # Correct Architecture Alignment
        "encoder.mlp_keys": "$^",
        "decoder.mlp_keys": "$^",
        "encoder.cnn_keys": "image",
        "decoder.cnn_keys": "image",
    })
    config = embodied.Flags(config).parse([])

    step = embodied.Counter()

    # ── Environment & Hybrid Wrapper ──────────────────────────────────────────
    raw_env = MicrorobotEnvGameByTheWall(**env_cfg)
    env = MaxAndSkipEnv(raw_env, 4)
    
    # *** INJECTING THE SHIELD HERE — keep reference for reporting ***
    mpc_shield = MPCSafetyShield(env)
    env = mpc_shield
    
    env = RateTargetReachedWrapper(env, 100, dreamer=True, verbose=0, logdir=f"{LOG_BASE}/tg_eval.csv")
    env = RewardWrapper(env, logdir=f"{LOG_BASE}/reward.csv")
    env = from_gym.FromGym(env)
    env = dreamerv3.wrap_env(env, config)
    env = embodied.BatchEnv([env], parallel=False)

    # ── Agent + Checkpoint ────────────────────────────────────────────────────
    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
    checkpoint_path = f"{ckp}/checkpoint.ckpt"
    checkpoint = embodied.Checkpoint(checkpoint_path)
    checkpoint.agent = agent
    checkpoint.load(checkpoint_path, keys=["agent"])
    
    print("\n" + "="*50)
    print(" 🛡️ STARTING HYBRID SHIELDED MODEL (MPC + Dreamer) 🛡️ ")
    print("="*50)

    writer = SummaryWriter(log_dir=TB_DIR)
    
    driver     = embodied.Driver(env)
    ep_counter = [1]

    def report(episode_data, env_index):
        ep           = ep_counter[0]
        rewards      = episode_data["reward"]
        total_reward = float(sum(rewards))
        ep_length    = len(rewards)
        tg_rate      = float(np.mean(episode_data.get("target_reached", [0])))
        
        # Read interventions directly from the captured shield reference
        interventions = mpc_shield.interventions

        print(f"✅ Shielded Ep {ep} | Reward: {total_reward:7.2f} | Steps: {ep_length:3d} | Shield Interventions: {interventions}")

        # Scalars
        writer.add_scalar("eval/episode_reward",      total_reward, ep)
        writer.add_scalar("eval/episode_length",      ep_length,    ep)
        writer.add_scalar("eval/target_reached_rate", tg_rate,      ep)
        writer.add_scalar("eval/shield_interventions", interventions, ep)

        if "image" in episode_data:
            frames = np.array(episode_data["image"])
            _, pil_frames = frames_to_gif_bytes(frames, fps=20)
            gif_path = os.path.join(LOG_BASE, f"shielded_episode_{ep}.gif")
            pil_frames[0].save(
                gif_path, format="GIF", save_all=True, append_images=pil_frames[1:],
                loop=0, duration=int(1000 / 20),
            )
            print(f"💾 Saved GIF: {gif_path}")

        writer.flush()
        ep_counter[0] += 1

    driver.on_episode(report)
    policy = lambda *args: agent.policy(*args, mode="eval")

    driver(policy, episodes=10)
    env.close()
    writer.close()
    
    print("\n" + "="*50)
    print(f"📊 Compare results: tensorboard --logdir results/play_logs_{CURRENT_ENV_NAME}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main(ENV_CONFIGS[CURRENT_ENV_NAME])
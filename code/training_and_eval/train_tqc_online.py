"""
train_tqc_online.py
===================
Online TQC training from scratch — fair comparison baseline against DreamerV3.

Training protocol (matches DreamerV3 budget):
  • 3,000,000 total environment steps
  • Round-robin across 6 vascular/navigation environments
  • Replay buffer (300K capacity) — off-policy RL
  • 1 gradient update every TRAIN_FREQ env steps (on GPU)
  • SuANN surrogate dynamics model updated concurrently
  • Checkpoint saved every 100K steps
  • Evaluation (10 eps × 6 envs) every 200K steps

Usage:
    cd d:\\redemption-eq\\microrobot-rl-main
    python scripts/train_tqc_online.py
"""

import os
import sys
import csv
import time
import warnings
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# Force line-buffered stdout so progress is visible in task logs
sys.stdout.reconfigure(line_buffering=True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environments import MicrorobotEnvGameByTheWall
from utils.tqc_agent import TQCAgent
from utils.suann_surrogate import SuANNSurrogate, SuANNDynamicsTrainer
from utils.replay_buffer import ReplayBuffer

# ══════════════════════════════════════════════════════════════════════════════
#  Hyper-parameters
# ══════════════════════════════════════════════════════════════════════════════

TOTAL_STEPS       = 3_000_000   # Match DreamerV3 training budget
REPLAY_CAPACITY   = 300_000     # Rolling replay window
WARMUP_STEPS      = 10_000      # Random exploration before first gradient update
BATCH_SIZE        = 256         # Mini-batch size for each gradient update
TRAIN_FREQ        = 4           # Gradient update every N env steps
EVAL_FREQ         = 200_000     # Full evaluation every N steps
SAVE_FREQ         = 100_000     # Checkpoint every N steps
SUANN_TRAIN_FREQ  = 100         # SuANN gradient update every N TQC gradient steps
LOG_PRINT_FREQ    = 500         # Console progress print every N steps

# TQC agent hyper-parameters
N_CRITICS         = 5
N_QUANTILES       = 25
TOP_QUANT_DROP    = 5
LR_TQC            = 3e-4
GAMMA             = 0.99
TAU               = 0.005       # Soft target update

# SuANN hyper-parameters
LR_SUANN          = 1e-3

# Paths
LOG_DIR   = "results/logs/tqc_v3_online"
_ENV_YAML = "results/play_logs/env_config_default_segmentation_vascular_large.yaml"

# Six environments — same as DreamerV3 training
ENV_CONFIGS = {
    "vascular":       {"image_string": "binary_images/default_segmentation_vascular.png"},
    "capillary":      {"image_string": "binary_images/capillary.png"},
    "veins":          {"image_string": "binary_images/veins.png"},
    "vascular_large": {"image_string": "binary_images/default_segmentation_vascular_large.png"},
    "racetrack":      {"image_string": "binary_images/default_mask_racetrack_segmented.png"},
    "maze_3":         {"image_string": "binary_images/maze_3.png"},
}
ALL_ENVS = list(ENV_CONFIGS.keys())

# ══════════════════════════════════════════════════════════════════════════════
#  Environment helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_env(env_name: str):
    """Instantiate a single MicrorobotEnvGameByTheWall for the given map."""
    cfg = ENV_CONFIGS[env_name]
    return MicrorobotEnvGameByTheWall(
        name=(0, 0),
        config=_ENV_YAML,
        max_envs=1,
        render_mode="rgb_array",
        subepisode_sampling=True,
        subepisode_length=2,
        reduce_radius=-0.01,
        image_string=cfg["image_string"],
    )


def parse_obs(raw_obs: dict):
    """Extract (image uint8 HWC, vector float32 (4,)) from env observation dict."""
    img = raw_obs["image"]                                        # (64, 64, 3) uint8
    vec = np.concatenate([
        raw_obs["agent_position"],
        raw_obs["target_position"],
    ]).astype(np.float32)                                         # (4,)
    return img, vec


def env_step(env, action):
    """Unified env.step() wrapper that handles both 4-tuple and 5-tuple APIs."""
    ret = env.step(action)
    if len(ret) == 5:
        next_obs, reward, terminated, truncated, info = ret
        done = terminated or truncated
    else:
        next_obs, reward, done, info = ret
    return next_obs, float(reward), bool(done), info


def env_reset(env):
    """Unified env.reset() wrapper."""
    ret = env.reset()
    return ret[0] if isinstance(ret, tuple) else ret


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(
    agent:        TQCAgent,
    step:         int,
    writer:       SummaryWriter,
    eval_writer:  csv.writer,
    eval_csv_file,
    device:       str,
) -> float:
    """
    Run 10 deterministic episodes per environment (60 episodes total).
    Returns overall mean success rate across all 6 environments.
    """
    print(f"\n{'-'*65}")
    print(f"  EVALUATION @ step {step:,}")
    print(f"{'-'*65}")

    all_sr = []

    for env_name in ALL_ENVS:
        env = make_env(env_name)
        ep_rewards, ep_steps, ep_ok = [], [], []

        for _ in range(10):
            obs  = env_reset(env)
            done = False
            ep_r = 0.0
            ep_s = 0

            while not done and ep_s < 400:
                img, vec     = parse_obs(obs)
                action       = agent.select_action(img, vec, deterministic=True)
                obs, r, done, _ = env_step(env, action)
                ep_r += r
                ep_s += 1

            ep_rewards.append(ep_r)
            ep_steps.append(ep_s)
            ep_ok.append(ep_r > 0)

        env.close()

        sr    = float(np.mean(ep_ok)) * 100.0
        avg_r = float(np.mean(ep_rewards))
        avg_s = float(np.mean(ep_steps))
        all_sr.append(sr)

        print(f"  [{env_name:14s}] SR: {sr:5.1f}% | "
              f"Avg Reward: {avg_r:7.2f} | Avg Steps: {avg_s:5.0f}")

        writer.add_scalar(f"eval/{env_name}/success_rate", sr, step)
        writer.add_scalar(f"eval/{env_name}/avg_reward",   avg_r, step)
        writer.add_scalar(f"eval/{env_name}/avg_steps",    avg_s, step)
        eval_writer.writerow([step, env_name, f"{sr:.1f}", f"{avg_r:.4f}", f"{avg_s:.1f}"])

    overall_sr = float(np.mean(all_sr))
    print(f"  {'-'*61}")
    print(f"  {'OVERALL':14s}  SR: {overall_sr:5.1f}%")
    print(f"{'-'*65}\n")

    writer.add_scalar("eval/overall_success_rate", overall_sr, step)
    eval_csv_file.flush()
    return overall_sr


# ══════════════════════════════════════════════════════════════════════════════
#  Main training loop
# ══════════════════════════════════════════════════════════════════════════════

def main():
    warnings.filterwarnings("ignore")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(LOG_DIR, exist_ok=True)

    print("=" * 70, flush=True)
    print(f"  TQC + SuANN  ONLINE TRAINING  --  FROM SCRATCH", flush=True)
    print(f"  Device : {device.upper()}", flush=True)
    print(f"  Steps  : {TOTAL_STEPS:,}", flush=True)
    print(f"  Buffer : {REPLAY_CAPACITY:,}  |  Batch: {BATCH_SIZE}  |  LR: {LR_TQC}", flush=True)
    print(f"  Envs   : {ALL_ENVS}", flush=True)
    print(f"  LogDir : {LOG_DIR}", flush=True)
    print("=" * 70, flush=True)

    # ── Models ─────────────────────────────────────────────────────────────────
    agent = TQCAgent(
        img_channels    = 3,
        vector_dim      = 4,
        action_dim      = 8,
        n_critics       = N_CRITICS,
        n_quantiles     = N_QUANTILES,
        top_quantile_drop = TOP_QUANT_DROP,
        gamma           = GAMMA,
        tau             = TAU,
        lr              = LR_TQC,
        device          = device,
    )

    suann_model   = SuANNSurrogate(state_dim=4, action_dim=8, hidden_dim=256).to(device)
    suann_trainer = SuANNDynamicsTrainer(suann_model, lr=LR_SUANN, device=device)

    # ── Checkpoint Auto-Resume ──────────────────────────────────────────────────
    resume_step = 0
    tqc_ckps = [f for f in os.listdir(LOG_DIR) if f.startswith("tqc_checkpoint_") and f.endswith(".pth")]
    if tqc_ckps:
        steps_found = []
        for ckp_name in tqc_ckps:
            try:
                s_str = ckp_name.replace("tqc_checkpoint_", "").replace("k.pth", "")
                steps_found.append((int(s_str) * 1000, ckp_name))
            except ValueError:
                pass
        if steps_found:
            steps_found.sort(key=lambda x: x[0], reverse=True)
            resume_step, latest_tqc_ckp = steps_found[0]
            latest_suann_ckp = latest_tqc_ckp.replace("tqc_checkpoint_", "suann_checkpoint_")
            tag_str = latest_tqc_ckp.replace("tqc_checkpoint_", "").replace(".pth", "")
            replay_path = os.path.join(LOG_DIR, f"replay_buffer_{tag_str}.npz")
            
            tqc_path = os.path.join(LOG_DIR, latest_tqc_ckp)
            suann_path = os.path.join(LOG_DIR, latest_suann_ckp)
            
            if os.path.exists(tqc_path):
                agent.load(tqc_path)
                print(f"  [RESUME] Loaded TQC agent checkpoint: {tqc_path}", flush=True)
            if os.path.exists(suann_path):
                suann_model.load_state_dict(torch.load(suann_path, map_location=device))
                print(f"  [RESUME] Loaded SuANN surrogate checkpoint: {suann_path}", flush=True)
            if os.path.exists(replay_path):
                replay.load(replay_path)
                print(f"  [RESUME] Loaded Replay Buffer memory: {replay_path}", flush=True)

    # ── Logging ────────────────────────────────────────────────────────────────
    writer    = SummaryWriter(log_dir=f"{LOG_DIR}/tensorboard")
    mode      = "a" if resume_step > 0 else "w"
    train_f   = open(f"{LOG_DIR}/training_log.csv", mode, newline="", encoding="utf-8")
    eval_f    = open(f"{LOG_DIR}/eval_log.csv",     mode, newline="", encoding="utf-8")
    train_csv = csv.writer(train_f)
    eval_csv  = csv.writer(eval_f)
    if resume_step == 0:
        train_csv.writerow(["step", "env", "episode_reward", "ep_steps",
                            "success", "critic_loss", "actor_loss", "suann_loss"])
        eval_csv.writerow( ["step", "env", "success_rate_%", "avg_reward", "avg_steps"])

    # ── Replay buffer ──────────────────────────────────────────────────────────
    replay = ReplayBuffer(capacity=REPLAY_CAPACITY, device=device)

    # ── Counters ───────────────────────────────────────────────────────────────
    total_steps     = resume_step
    gradient_steps  = max(0, (resume_step - WARMUP_STEPS) // TRAIN_FREQ)
    last_save_step  = (resume_step // SAVE_FREQ) * SAVE_FREQ
    last_eval_step  = (resume_step // EVAL_FREQ) * EVAL_FREQ
    best_sr         = 0.0
    t_start         = time.time()

    # Episode counters
    ep_reward  = 0.0
    ep_steps_n = 0
    last_c_loss = 0.0
    last_a_loss = 0.0
    last_s_loss = 0.0

    # ── Environment (round-robin) ──────────────────────────────────────────────
    env_idx  = 0
    env_name = ALL_ENVS[env_idx]
    env      = make_env(env_name)
    obs      = env_reset(env)

    if total_steps < WARMUP_STEPS:
        print(f"\n[Warmup] Collecting {WARMUP_STEPS:,} random steps before first gradient update ...\n", flush=True)
    else:
        print(f"\n[Resuming] Continuing training from step {total_steps:,} (Gradient Steps: {gradient_steps:,}) ...\n", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  Main loop
    # ══════════════════════════════════════════════════════════════════════════
    while total_steps < TOTAL_STEPS:

        # ── Action selection ──────────────────────────────────────────────────
        img, vec = parse_obs(obs)
        if total_steps < WARMUP_STEPS:
            action = env.action_space.sample()          # random warmup
        else:
            action = agent.select_action(img, vec, deterministic=False)

        # ── Environment step ──────────────────────────────────────────────────
        next_obs, reward, done, _ = env_step(env, action)
        next_img, next_vec        = parse_obs(next_obs)

        # Store transition
        replay.add(img, vec, action, reward, next_img, next_vec, done)

        ep_reward  += reward
        ep_steps_n += 1
        total_steps += 1

        obs = next_obs

        # ── Episode end ───────────────────────────────────────────────────────
        if done:
            success = ep_reward > 0

            writer.add_scalar("train/episode_reward",          ep_reward,  total_steps)
            writer.add_scalar(f"train/{env_name}/ep_reward",   ep_reward,  total_steps)
            writer.add_scalar("train/episode_steps",           ep_steps_n, total_steps)

            train_csv.writerow([total_steps, env_name,
                                f"{ep_reward:.4f}", ep_steps_n, int(success),
                                f"{last_c_loss:.6f}", f"{last_a_loss:.6f}",
                                f"{last_s_loss:.6f}"])

            # Console log every LOG_PRINT_FREQ steps (approx)
            if total_steps % LOG_PRINT_FREQ < ep_steps_n + 5:
                elapsed = time.time() - t_start
                fps     = total_steps / max(1.0, elapsed)
                eta_h   = (TOTAL_STEPS - total_steps) / max(1.0, fps) / 3600.0
                print(
                    f"  [{total_steps:>9,} / {TOTAL_STEPS:,}] "
                    f"Env: {env_name:14s} | "
                    f"Rwd: {ep_reward:7.2f} | "
                    f"{'OK    ' if success else 'FAIL  ':6s} | "
                    f"Buf: {len(replay):>7,} | "
                    f"GradSteps: {gradient_steps:>6,} | "
                    f"FPS: {fps:5.1f} | ETA: {eta_h:.1f}h",
                    flush=True
                )

            # Cycle to next environment (round-robin)
            env.close()
            env_idx  = (env_idx + 1) % len(ALL_ENVS)
            env_name = ALL_ENVS[env_idx]
            env      = make_env(env_name)
            obs      = env_reset(env)
            ep_reward  = 0.0
            ep_steps_n = 0

        # ── Gradient update ───────────────────────────────────────────────────
        if (total_steps >= WARMUP_STEPS
                and len(replay) >= BATCH_SIZE
                and total_steps % TRAIN_FREQ == 0):

            if total_steps == WARMUP_STEPS:
                print(f"[Train] Warmup done — starting online RL training at step {total_steps:,}\n")

            imgs, vecs, acts, rews, n_imgs, n_vecs, dones_t = replay.sample(BATCH_SIZE)

            # TQC gradient step (GPU)
            last_c_loss, last_a_loss = agent.train_step(
                imgs, vecs, acts, rews, n_imgs, n_vecs, dones_t
            )
            gradient_steps += 1

            writer.add_scalar("train/critic_loss", last_c_loss, total_steps)
            writer.add_scalar("train/actor_loss",  last_a_loss, total_steps)

            # SuANN concurrent update (every SUANN_TRAIN_FREQ gradient steps)
            if gradient_steps % SUANN_TRAIN_FREQ == 0:
                # Convert discrete action indices → one-hot on GPU
                acts_oh = torch.zeros(BATCH_SIZE, 8, device=device)
                acts_oh.scatter_(1, acts.view(-1, 1), 1.0)

                # Single pass SuANN update
                suann_model.train()
                suann_trainer.optimizer.zero_grad()
                pred_ns, pred_r = suann_model(vecs, acts_oh)
                loss_s  = torch.nn.functional.mse_loss(pred_ns[:, :2], n_vecs[:, :2])
                loss_r  = torch.nn.functional.mse_loss(pred_r.squeeze(-1), rews)
                s_loss  = loss_s + loss_r
                s_loss.backward()
                suann_trainer.optimizer.step()
                last_s_loss = s_loss.item()

                writer.add_scalar("train/suann_loss", last_s_loss, total_steps)

        # ── Checkpoint ────────────────────────────────────────────────────────
        if total_steps - last_save_step >= SAVE_FREQ:
            last_save_step = (total_steps // SAVE_FREQ) * SAVE_FREQ
            tag = f"{last_save_step // 1_000}k"
            agent.save(os.path.join(LOG_DIR, f"tqc_checkpoint_{tag}.pth"))
            torch.save(suann_model.state_dict(),
                       os.path.join(LOG_DIR, f"suann_checkpoint_{tag}.pth"))
            replay.save(os.path.join(LOG_DIR, f"replay_buffer_{tag}.npz"))
            train_f.flush()
            print(f"  [SAVE] Checkpoint & Replay Buffer @ step {last_save_step:,}  ({tag})", flush=True)

        # ── Evaluation ────────────────────────────────────────────────────────
        if total_steps - last_eval_step >= EVAL_FREQ:
            last_eval_step = (total_steps // EVAL_FREQ) * EVAL_FREQ
            # Close training env temporarily
            env.close()

            sr = evaluate(agent, last_eval_step, writer, eval_csv, eval_f, device)

            if sr > best_sr:
                best_sr = sr
                agent.save(os.path.join(LOG_DIR, "tqc_best.pth"))
                torch.save(suann_model.state_dict(),
                           os.path.join(LOG_DIR, "suann_best.pth"))
                print(f"  [BEST] New best success rate: {best_sr:.1f}%  -- model saved.\n", flush=True)

            # Re-open training env after eval
            env = make_env(env_name)
            obs = env_reset(env)
            ep_reward  = 0.0
            ep_steps_n = 0

    # ══════════════════════════════════════════════════════════════════════════
    #  Training complete
    # ══════════════════════════════════════════════════════════════════════════
    env.close()

    agent.save(os.path.join(LOG_DIR, "tqc_final.pth"))
    torch.save(suann_model.state_dict(), os.path.join(LOG_DIR, "suann_final.pth"))

    train_f.close()
    eval_f.close()
    writer.close()

    elapsed_h = (time.time() - t_start) / 3600.0
    print("\n" + "=" * 70)
    print(f"  TQC ONLINE TRAINING COMPLETE")
    print(f"  Total env steps  : {total_steps:,}")
    print(f"  Gradient updates : {gradient_steps:,}")
    print(f"  Best success rate: {best_sr:.1f}%")
    print(f"  Wall-clock time  : {elapsed_h:.2f} h")
    print(f"  Final model      : {LOG_DIR}/tqc_final.pth")
    print("=" * 70)


if __name__ == "__main__":
    main()

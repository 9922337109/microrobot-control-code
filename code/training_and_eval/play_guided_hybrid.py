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

# ── SET YOUR CHOICE HERE ─────────────────────────────────────────────────────
CURRENT_ENV_NAME = "maze_3"

# ── Log directory map ─────────────────────────────────────────────────────────
_LOG_DIRS = {
    "vascular":       "results/play_logs_vascular/guided_hybrid",
    "capillary":      "results/play_logs_capillary_test/guided_hybrid",
    "veins":          "results/play_logs_aml_final_video/guided_hybrid",
    "vascular_large": "results/play_logs/guided_hybrid",
    "racetrack":      "results/play_logs_racetrack/guided_hybrid",
    "maze_3":         "results/play_logs_maze_3_test/guided_hybrid",
}
LOG_BASE = _LOG_DIRS[CURRENT_ENV_NAME]
TB_DIR   = f"{LOG_BASE}/tensorboard"
ckp      = "results/logs/finetuned_paper_upgrades"

# ── Environment configs ───────────────────────────────────────────────────────
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

# ── 1. A* PLANNER ─────────────────────────────────────────────────────────────
def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def astar(array, start, goal):
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    close_set  = set()
    came_from  = {}
    gscore     = {start: 0}
    fscore     = {start: heuristic(start, goal)}
    oheap      = []
    heapq.heappush(oheap, (fscore[start], start))
    max_x, max_y = array.shape

    while oheap:
        current = heapq.heappop(oheap)[1]
        if np.linalg.norm(np.array(current) - np.array(goal)) < 3.0:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data[::-1]

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            if 0 <= neighbor[0] < max_x and 0 <= neighbor[1] < max_y:
                if array[neighbor[0]][neighbor[1]] == 1:
                    continue
            else:
                continue
            tentative_g = gscore[current] + heuristic(current, neighbor)
            if neighbor in close_set and tentative_g >= gscore.get(neighbor, 0):
                continue
            if tentative_g < gscore.get(neighbor, 0) or neighbor not in [x[1] for x in oheap]:
                came_from[neighbor] = current
                gscore[neighbor]    = tentative_g
                fscore[neighbor]    = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    return None

# ── 2. WAYPOINT-GUIDED WRAPPER ────────────────────────────────────────────────
class WaypointGuidedWrapper(gym.Wrapper):
    """
    Replaces env's target_position with the next A* waypoint each step.

    [Paper 3 — Vascular Parkour, arXiv 2025] Upgrades:
      - Dynamic A* Replanning: path recomputed every step from current agent
        position instead of only at episode reset. Eliminates stale-path deadlocks.
      - Stuck Detection + Escape Waypoint: detects when robot makes no progress
        for STUCK_THRESH consecutive steps and injects a reflected escape waypoint
        to break the deadlock before resuming normal planning.
    """

    # ── Stuck detection hyperparameters (Paper 3 Section 3.3) ────────────────
    STUCK_THRESH  = 15    # steps without sufficient progress → stuck
    PROGRESS_EPS  = 2.0   # minimum pixel movement per step to count as progress
    ESCAPE_STEPS  = 8     # how many steps the escape waypoint stays active

    # Per-environment max steps — vascular is the largest branched map and needs more steps
    _MAX_STEPS = {
        "vascular":       2001,   # Complex branched topology — longest paths
        "vascular_large": 2001,   # Larger map — paths can be very long
        "capillary":      1800,   # Narrow channels need more time to navigate carefully
        "racetrack":      1500,   # Closed loop — shorter effective paths
        "veins":          1500,   # Medium complexity
        "maze_3":         1800,   # Dead-end maze — needs backtracking time
    }

    def __init__(self, env, map_path, env_name=None):
        super().__init__(env)
        raw_img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        self.grid_size = (64, 64)
        if raw_img.shape != self.grid_size:
            raw_img = cv2.resize(raw_img, (self.grid_size[1], self.grid_size[0]))

        _, binary_map = cv2.threshold(raw_img, 127, 1, cv2.THRESH_BINARY)
        # Inflate safe region (3 iterations) so edge-of-vessel positions
        # map to free A* cells rather than wall cells
        kernel     = np.ones((3, 3), np.uint8)
        binary_map = cv2.dilate(binary_map, kernel, iterations=3)
        self.grid  = 1 - binary_map          # A* expects 1=wall, 0=free

        self.current_path   = []
        self.waypoint_index = 0
        self.ep_steps       = 0

        # Per-environment step budget — larger/more complex maps get more time
        if env_name and env_name in self._MAX_STEPS:
            self.max_steps = self._MAX_STEPS[env_name]
        else:
            self.max_steps = 1500   # safe default for unknown environments

        # [Paper 3] Goal coordinates stored for per-step replanning
        self._goal_rc       = None   # (row, col) of episode goal, set in reset()

        # [Paper 3] Stuck detection state
        self._stuck_counter  = 0
        self._last_grid_pos  = None  # (row, col) of agent at last check
        self._escape_steps   = 0     # countdown for escape waypoint
        self._escape_wp      = None  # temporary escape waypoint (row, col)

    def _is_free_cell(self, row, col, radius=1):
        """Check if a (row, col) cell and its neighbourhood are all free in the grid.

        BUG FIX: a single-pixel free cell may still be physically blocked when
        surrounded by dilated wall pixels. Checking a (2*radius+1)^2 neighbourhood
        ensures the escape waypoint lands in genuinely open space.
        """
        gs = self.grid_size
        r0 = max(0, row - radius)
        r1 = min(gs[0] - 1, row + radius)
        c0 = max(0, col - radius)
        c1 = min(gs[1] - 1, col + radius)
        return np.all(self.grid[r0:r1+1, c0:c1+1] == 0)

    def reset(self, **kwargs):
        ret = self.env.reset(**kwargs)
        obs  = ret[0] if isinstance(ret, tuple) else ret
        info = ret[1] if isinstance(ret, tuple) else {}

        # Reset all episode state
        self.waypoint_index  = 0
        self.ep_steps        = 0
        self._stuck_counter  = 0
        self._last_grid_pos  = None
        self._escape_steps   = 0
        self._escape_wp      = None
        self._goal_rc        = None

        try:
            start_norm  = obs['agent_position']   # [x_norm, y_norm]
            target_norm = obs['target_position']   # [x_norm, y_norm]

            # agent_position[0]=x (horizontal), agent_position[1]=y (vertical)
            # A* grid uses [row, col] = [y_idx, x_idx]  ← must swap!
            gs = self.grid_size
            sx = np.clip(int(start_norm[0]  * gs[1]), 0, gs[1]-1)  # x → col
            sy = np.clip(int(start_norm[1]  * gs[0]), 0, gs[0]-1)  # y → row
            gx = np.clip(int(target_norm[0] * gs[1]), 0, gs[1]-1)
            gy = np.clip(int(target_norm[1] * gs[0]), 0, gs[0]-1)

            # Clear 3×3 neighbourhood around start AND goal in grid (row, col) coords
            r0, r1 = max(0, sy-1), min(gs[0]-1, sy+1)
            c0, c1 = max(0, sx-1), min(gs[1]-1, sx+1)
            self.grid[r0:r1+1, c0:c1+1] = 0
            r0, r1 = max(0, gy-1), min(gs[0]-1, gy+1)
            c0, c1 = max(0, gx-1), min(gs[1]-1, gx+1)
            self.grid[r0:r1+1, c0:c1+1] = 0

            # Store goal for per-step replanning [Paper 3]
            self._goal_rc = (gy, gx)

            # Initial A* path at episode start
            self.current_path   = astar(self.grid, (sy, sx), (gy, gx))
            self.waypoint_index = 0

            if not self.current_path:
                print(f"[WARNING] A*: No path found (start=({sy},{sx}), goal=({gy},{gx}))")
        except Exception as e:
            print(f"[WARNING] A* planner error in reset: {e}")
            self.current_path = []

        obs = self._update_obs(obs)
        return (obs, info) if isinstance(ret, tuple) else obs

    def step(self, action):
        ret = self.env.step(action)
        if len(ret) == 5:
            obs, reward, terminated, truncated, info = ret
        else:
            obs, reward, done, info = ret
            terminated, truncated = done, False

        self.ep_steps += 1
        if self.ep_steps >= self.max_steps:
            terminated = True   # force episode end after timeout
            truncated  = True

        # ── Compute current grid position ─────────────────────────────────────
        gs = self.grid_size
        try:
            ax = int(obs['agent_position'][0] * gs[1])  # col
            ay = int(obs['agent_position'][1] * gs[0])  # row
            ax = np.clip(ax, 0, gs[1]-1)
            ay = np.clip(ay, 0, gs[0]-1)
            curr_rc = np.array([ay, ax])               # (row, col)
        except Exception:
            curr_rc = None

        # ── [Paper 3] Dynamic A* Replanning ──────────────────────────────────
        # Recompute path every step from current agent position.
        # This eliminates stale-path deadlocks where the stored path
        # diverges from the agent's actual position mid-episode.
        if curr_rc is not None and self._goal_rc is not None and not terminated:
            try:
                new_path = astar(
                    self.grid,
                    (int(curr_rc[0]), int(curr_rc[1])),
                    self._goal_rc
                )
                if new_path and len(new_path) > 0:
                    self.current_path   = new_path
                    self.waypoint_index = 0   # always start from nearest waypoint
            except Exception:
                pass  # keep existing path on A* failure

        # ── Advance waypoint if robot is close enough ─────────────────────────
        if self.current_path and self.waypoint_index < len(self.current_path):
            try:
                target_wp = np.array(self.current_path[self.waypoint_index])
                if np.linalg.norm(curr_rc - target_wp) < 4.0:
                    self.waypoint_index = min(
                        self.waypoint_index + 1, len(self.current_path) - 1
                    )
            except Exception:
                pass

        # ── [Paper 3] Stuck Detection + Escape Waypoint ───────────────────────
        # Detects when agent makes no progress for STUCK_THRESH steps.
        # Injects a symmetric escape waypoint to break the deadlock.
        if curr_rc is not None and not terminated:
            if self._last_grid_pos is not None:
                moved = np.linalg.norm(curr_rc - self._last_grid_pos)
                if moved < self.PROGRESS_EPS:
                    self._stuck_counter += 1
                else:
                    self._stuck_counter = 0
            self._last_grid_pos = curr_rc.copy()

            # Countdown escape waypoint (Paper 3 Section 3.3)
            if self._escape_steps > 0:
                self._escape_steps -= 1
                if self._escape_steps == 0:
                    self._escape_wp = None  # escape phase over, resume normal planning

            # Trigger escape if stuck threshold exceeded
            if (self._stuck_counter > self.STUCK_THRESH
                    and self.current_path
                    and self._escape_wp is None):
                try:
                    # Direction vector from waypoint to current position
                    wp_row, wp_col = self.current_path[self.waypoint_index]
                    d_row = curr_rc[0] - wp_row
                    d_col = curr_rc[1] - wp_col

                    # Candidate escape cells: direct opposite (180 deg) and perpendicular directions (90 deg CCW/CW)
                    candidates = [
                        (curr_rc[0] + d_row, curr_rc[1] + d_col),
                        (curr_rc[0] + d_col, curr_rc[1] - d_row),
                        (curr_rc[0] - d_col, curr_rc[1] + d_row),
                    ]

                    success = False
                    for r_cand, c_cand in candidates:
                        esc_row = int(np.clip(r_cand, 0, gs[0]-1))
                        esc_col = int(np.clip(c_cand, 0, gs[1]-1))
                        if self._is_free_cell(esc_row, esc_col, radius=1):
                            self._escape_wp    = (esc_row, esc_col)
                            self._escape_steps = self.ESCAPE_STEPS
                            success = True
                            break

                    if not success:
                        # Neighborhood search if primary reflections are blocked
                        for dr, dc in [(-2,0),(2,0),(0,-2),(0,2),(-2,-2),(2,2)]:
                            esc_row = int(np.clip(curr_rc[0] + dr, 0, gs[0]-1))
                            esc_col = int(np.clip(curr_rc[1] + dc, 0, gs[1]-1))
                            if self._is_free_cell(esc_row, esc_col, radius=1):
                                self._escape_wp    = (esc_row, esc_col)
                                self._escape_steps = self.ESCAPE_STEPS
                                success = True
                                break

                    self._stuck_counter = 0
                except Exception:
                    self._stuck_counter = 0

        obs = self._update_obs(obs)
        if len(ret) == 5:
            return obs, reward, terminated, truncated, info
        return obs, reward, (terminated or truncated), info

    def _update_obs(self, obs):
        """Override target_position with the look-ahead A* waypoint.

        If an escape waypoint is active (Paper 3 stuck escape), that takes
        priority over the normal A* waypoint until the escape phase ends.
        """
        gs = self.grid_size

        # [Paper 3] Active escape waypoint takes priority
        if self._escape_wp is not None:
            row, col = self._escape_wp
            x_norm = col / gs[1]
            y_norm = row / gs[0]
            if isinstance(obs, dict):
                obs = obs.copy()
                obs['target_position'] = np.array([x_norm, y_norm], dtype=np.float32)
            return obs

        # Normal A* look-ahead waypoint
        if self.current_path and self.waypoint_index < len(self.current_path):
            lookahead = min(self.waypoint_index + 4, len(self.current_path) - 1)
            row, col  = self.current_path[lookahead]   # A* returns (row, col) = (y_idx, x_idx)
            # Convert back to [x_norm, y_norm] matching agent_position convention
            x_norm = col / gs[1]   # col → x
            y_norm = row / gs[0]   # row → y
            if isinstance(obs, dict):
                obs = obs.copy()
                obs['target_position'] = np.array([x_norm, y_norm], dtype=np.float32)
        return obs

# ── 3. MPC SAFETY SHIELD ──────────────────────────────────────────────────────
class MPCSafetyShield(gym.Wrapper):
    """
    Pixel-level wall-collision shield using the env's own _fast_obstacles map.
    Uses minimum amplitude to check only truly immediate collisions.
    Intervention counter increments only when the action is actually changed.

    [Paper 4 — TQC+SuANN, Wiley AIS 2025] Upgrade:
      Adaptive safety threshold: the collision-check radius scales DOWN
      automatically when the intervention rate is high (indicating a narrow
      channel). This prevents over-blocking in tight vessels (Capillary, Veins)
      while maintaining full safety in open environments.

      Base radius: 8px (as before)
      Adaptive range: 4px (50% of base) ← 8px (100% of base)
      Scaling: linear in intervention_rate ∈ [0, 1]
    """

    def __init__(self, env, window_size=30):
        super().__init__(env)
        self.interventions   = 0
        self._step_count     = 0     # [Paper 4] total steps for rate calculation
        self._warned_no_mpc  = False
        self._warned_ex      = False
        self._window_size    = window_size
        self._history        = []    # tracks 1 for intervention, 0 for normal in last N steps

    def reset(self, **kwargs):
        self.interventions  = 0
        self._step_count    = 0
        self._history       = []
        return self.env.reset(**kwargs)

    def step(self, action):
        # Decode action index (DreamerV3 passes one-hot ndarray or int)
        if isinstance(action, np.ndarray) and action.shape == (8,):
            action_idx_0 = int(np.argmax(action))
        elif isinstance(action, (int, np.integer)):
            action_idx_0 = int(action)
        else:
            return self.env.step(action)

        self._step_count += 1       # [Paper 4] track total steps
        safe_action = action
        u = self.env.unwrapped          # MicrorobotEnvGameByTheWall

        _has_mpc = (hasattr(u, '_action_to_direction') and
                    hasattr(u, 'agent_location') and
                    hasattr(u, '_fast_obstacles') and
                    hasattr(u, '_eval_pixel_collision'))
        if not _has_mpc:
            if not self._warned_no_mpc:
                print("[WARNING] MPCSafetyShield: required env attributes missing -- shield INACTIVE")
                self._warned_no_mpc = True
            return self.env.step(action)

        try:
            n_piezos  = u.config['Action_space_settings']['NUMBER_PIEZOS']
            act_num   = action_idx_0 % n_piezos
            env_key   = act_num + 1             # env dict is 1-indexed

            direction = u._action_to_direction[env_key]

            # ── [Paper 4] Adaptive safety threshold ──────────────────────────
            # Base amplitude from physics (minimum step size = 8px in 512-space)
            ratio          = u.img_size / 64.0       # = 8.0  (512px / 64px)
            base_amplitude = u.min_ampl * ratio       # = 8px baseline

            # Intervention rate: fraction of steps where shield overrode action
            # High rate → we are in a tight, narrow channel → scale threshold down
            # Low rate  → open environment → keep full threshold
            if len(self._history) > 0:
                intervention_rate = sum(self._history) / len(self._history)
            else:
                intervention_rate = 0.0
            # Clamp to [0, 1] and map to scale [0.5, 1.0]:
            #   rate=0.0 → scale=1.0 → amplitude=8px   (open space)
            #   rate=1.0 → scale=0.5 → amplitude=4px   (very tight channel)
            adaptive_scale = max(0.5, 1.0 - 0.5 * min(intervention_rate, 1.0))
            amplitude      = base_amplitude * adaptive_scale

            predicted = u.agent_location + direction * amplitude
            px, py    = int(predicted[0]), int(predicted[1])

            # Safety radius = bubble radius + tolerance (same geometry as is_valid)
            r  = int(np.ceil(u.get_radius() + u.tolerance_collision))
            xs = np.arange(max(0, px - r), min(u.img_size, px + r + 1))
            ys = np.arange(max(0, py - r), min(u.img_size, py + r + 1))
            xx, yy = np.meshgrid(xs, ys)

            would_collide = u._eval_pixel_collision(xx, yy)

            overrode = False
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
                        self.interventions += 1   # count only actual action changes
                        overrode = True
                        break

            # Record outcome in history and prune to window size
            self._history.append(1 if overrode else 0)
            if len(self._history) > self._window_size:
                self._history.pop(0)

        except Exception as ex:
            if not self._warned_ex:
                print(f"⚠️  MPCSafetyShield exception (once): {ex}")
                self._warned_ex = True

        return self.env.step(safe_action)

# ── 4. GIF UTILITIES ──────────────────────────────────────────────────────────
def frames_to_gif_bytes(frames, fps=20):
    pil_frames = [PILImage.fromarray(f) for f in frames]
    buf = io.BytesIO()
    pil_frames[0].save(
        buf, format="GIF", save_all=True, append_images=pil_frames[1:],
        loop=0, duration=int(1000 / fps),
    )
    buf.seek(0)
    return buf, pil_frames

# ── 5. MAIN EXECUTION ─────────────────────────────────────────────────────────
def main(env_cfg):
    warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")
    os.makedirs(LOG_BASE, exist_ok=True)
    os.makedirs(TB_DIR,   exist_ok=True)

    config = embodied.Config(dreamerv3.configs["defaults"])
    config = config.update(dreamerv3.configs["large"])
    config = config.update({
        "logdir":            LOG_BASE,
        "run.train_ratio":   1,
        "jax.platform":      "cpu",   # JAX CUDA not installed — using cpu (PyTorch GPU separate)
        "encoder.mlp_keys":  "$^",
        "decoder.mlp_keys":  "$^",
        "encoder.cnn_keys":  "image",
        "decoder.cnn_keys":  "image",
    })
    config = embodied.Flags(config).parse([])

    step = embodied.Counter()

    # ── Environment pipeline ──────────────────────────────────────────────────
    raw_env    = MicrorobotEnvGameByTheWall(**env_cfg)
    env        = MaxAndSkipEnv(raw_env, 4)

    # Capture shield reference BEFORE further wrapping (find_shield can't cross from_gym)
    mpc_shield = MPCSafetyShield(env)
    env        = mpc_shield

    env = WaypointGuidedWrapper(env, env_cfg['image_string'], env_name=CURRENT_ENV_NAME)
    env = RateTargetReachedWrapper(env, 100, dreamer=True, verbose=0, logdir=f"{LOG_BASE}/tg_eval.csv")
    env = RewardWrapper(env, logdir=f"{LOG_BASE}/reward.csv")
    env = from_gym.FromGym(env)
    env = dreamerv3.wrap_env(env, config)
    env = embodied.BatchEnv([env], parallel=False)

    # ── Agent + checkpoint ────────────────────────────────────────────────────
    agent          = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
    checkpoint     = embodied.Checkpoint(f"{ckp}/checkpoint.ckpt")
    checkpoint.agent = agent
    if os.path.exists(f"{ckp}/checkpoint.ckpt"):
        print(f"[LOAD] Loading checkpoint: {ckp}/checkpoint.ckpt")
        checkpoint.load(f"{ckp}/checkpoint.ckpt", keys=["agent"])
    else:
        print(f"[INFO] Checkpoint file '{ckp}/checkpoint.ckpt' not found. Evaluating with fresh agent initialization.")

    print("\n" + "=" * 60)
    print(f"  HIERARCHICAL HYBRID -- {CURRENT_ENV_NAME.upper()}")
    print("     A* Global + DreamerV3 Local + MPC Safety Shield")
    print("  Papers: Vascular Parkour (arXiv 2025) + TQC+SuANN (Wiley AIS 2025)")
    print("  [OK] Dynamic A* Replanning: ON")
    print("  [OK] Stuck Escape Waypoint: ON")
    print("  [OK] Adaptive MPC Threshold: ON")
    print("=" * 60)

    writer     = SummaryWriter(log_dir=TB_DIR)
    driver     = embodied.Driver(env)
    ep_counter = [1]

    def report(episode_data, env_index):
        ep           = ep_counter[0]
        rewards      = episode_data["reward"]
        total_reward = float(sum(rewards))
        ep_length    = len(rewards)
        tg_rate      = float(np.mean(episode_data.get("target_reached", [0])))
        interventions = mpc_shield.interventions    # direct captured reference

        print(f"[EPISODE {ep:2d}] Reward: {total_reward:8.2f} | "
              f"Steps: {ep_length:4d} | Interventions: {interventions}")

        writer.add_scalar("eval/episode_reward",       total_reward,  ep)
        writer.add_scalar("eval/episode_length",       ep_length,     ep)
        writer.add_scalar("eval/target_reached_rate",  tg_rate,       ep)
        writer.add_scalar("eval/shield_interventions", interventions, ep)

        if "image" in episode_data:
            frames = np.array(episode_data["image"])
            _, pil_frames = frames_to_gif_bytes(frames, fps=20)
            gif_path = os.path.join(LOG_BASE, f"guided_episode_{ep}.gif")
            pil_frames[0].save(
                gif_path, format="GIF", save_all=True,
                append_images=pil_frames[1:], loop=0, duration=50,
            )
            print(f"[SAVE] Saved GIF: {gif_path}")

        writer.flush()
        ep_counter[0] += 1

    driver.on_episode(report)
    policy = lambda *args: agent.policy(*args, mode="eval")

    driver(policy, episodes=10)
    env.close()
    writer.close()

    print(f"\n{'='*60}")
    print(f"[LOGS] TensorBoard: tensorboard --logdir {LOG_BASE}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main(ENV_CONFIGS[CURRENT_ENV_NAME])
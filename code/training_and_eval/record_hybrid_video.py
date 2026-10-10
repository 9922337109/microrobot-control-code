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
import numpy as np
import cv2
import gymnasium as gym
import dreamerv3
from dreamerv3 import embodied
from dreamerv3.embodied.envs import from_gym
from environments import MicrorobotEnvGameByTheWall
from environments.costum_wrappers.MaxandSkip import MaxAndSkipEnv
import imageio
import os

# --- CONFIGURATION ---
MODEL_PATH = "results/logs/finetuned_paper_upgrades/checkpoint.ckpt" 
VIDEO_PATH_MP4 = "results/play_logs_guided_hybrid/hybrid_demonstration.mp4"
VIDEO_PATH_GIF = "results/play_logs_guided_hybrid/hybrid_demonstration.gif"
IMAGE_PATH = 'binary_images/default_segmentation_vascular.png'
NUM_EPISODES = 3

# --- 1. A* PLANNER ---
def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def astar(array, start, goal):
    neighbors = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []
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
                if array[neighbor[0]][neighbor[1]] == 1: continue
            else: continue
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0): continue
            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    return None

# --- 2. GUIDED WRAPPER ---
class WaypointGuidedWrapper(gym.Wrapper):
    def __init__(self, env, map_path):
        super().__init__(env)
        raw_img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        self.grid_size = (64, 64)
        if raw_img.shape != self.grid_size:
            raw_img = cv2.resize(raw_img, (self.grid_size[1], self.grid_size[0]))
        _, binary_map = cv2.threshold(raw_img, 127, 1, cv2.THRESH_BINARY)
        kernel = np.ones((3,3), np.uint8)
        binary_map = cv2.dilate(binary_map, kernel, iterations=1)
        self.grid = 1 - binary_map 
        self.current_path = []
        self.waypoint_index = 0
        
    def reset(self, **kwargs):
        ret = self.env.reset(**kwargs)
        if isinstance(ret, tuple): obs, info = ret
        else: obs, info = ret, {}
        try:
            start_grid = (obs['agent_position'] * np.array(self.grid_size)).astype(int)
            target_grid = (obs['target_position'] * np.array(self.grid_size)).astype(int)
            start_grid = np.clip(start_grid, 0, 63)
            target_grid = np.clip(target_grid, 0, 63)
            if self.grid[start_grid[0], start_grid[1]] == 1: self.grid[start_grid[0], start_grid[1]] = 0 
            self.current_path = astar(self.grid, tuple(start_grid), tuple(target_grid))
            self.waypoint_index = 0
        except: self.current_path = []
        if isinstance(ret, tuple): return self._update_obs(obs), info
        return self._update_obs(obs)

    def step(self, action):
        ret = self.env.step(action)
        if len(ret) == 5: obs, reward, terminated, truncated, info = ret
        else: obs, reward, done, info = ret; terminated, truncated = done, False
        if self.current_path and self.waypoint_index < len(self.current_path):
            try:
                curr_grid = (obs['agent_position'] * np.array(self.grid_size)).astype(int)
                if np.linalg.norm(curr_grid - np.array(self.current_path[self.waypoint_index])) < 4.0:
                    self.waypoint_index = min(self.waypoint_index + 1, len(self.current_path) - 1)
            except: pass
        updated_obs = self._update_obs(obs)
        if len(ret) == 5: return updated_obs, reward, terminated, truncated, info
        return updated_obs, reward, done, info

    def _update_obs(self, obs):
        if self.current_path and self.waypoint_index < len(self.current_path):
            lookahead = min(self.waypoint_index + 4, len(self.current_path) - 1)
            waypoint_norm = np.array(self.current_path[lookahead]) / np.array(self.grid_size)
            if isinstance(obs, dict):
                obs = obs.copy()
                obs['target_position'] = waypoint_norm.astype(np.float32)
        return obs

# --- 3. MPC SHIELD ---
class MPCSafetyShield(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.interventions = 0

    def step(self, action):
        if isinstance(action, np.ndarray) and action.shape == (8,): action_idx_0 = int(np.argmax(action))
        elif isinstance(action, (int, np.integer)): action_idx_0 = int(action)
        else: return self.env.step(action)

        safe_action = action
        u = self.env.unwrapped

        _has_mpc = (hasattr(u, '_action_to_direction') and
                    hasattr(u, 'agent_location') and
                    hasattr(u, '_fast_obstacles') and
                    hasattr(u, '_eval_pixel_collision'))
        if not _has_mpc:
            return self.env.step(action)

        try:
            n_piezos  = u.config['Action_space_settings']['NUMBER_PIEZOS']
            act_num   = action_idx_0 % n_piezos
            env_key   = act_num + 1

            direction = u._action_to_direction[env_key]
            ratio     = u.img_size / 64.0
            amplitude = u.min_ampl * ratio

            predicted = u.agent_location + direction * amplitude
            px, py    = int(predicted[0]), int(predicted[1])

            r  = int(np.ceil(u.get_radius() + u.tolerance_collision))
            xs = np.arange(max(0, px - r), min(u.img_size, px + r + 1))
            ys = np.arange(max(0, py - r), min(u.img_size, py + r + 1))
            xx, yy = np.meshgrid(xs, ys)

            would_collide = u._eval_pixel_collision(xx, yy)

            if would_collide:
                alt_candidates = []
                for alt_key, alt_dir in u._action_to_direction.items():
                    if alt_key == env_key:
                        continue
                    alt_pred = u.agent_location + alt_dir * amplitude
                    dist_to_tgt = np.linalg.norm(alt_pred - u.target_location)
                    alt_candidates.append((dist_to_tgt, alt_key, alt_dir))
                
                alt_candidates.sort(key=lambda x: x[0])

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
                        self.interventions += 1
                        break
        except Exception:
            pass

        return self.env.step(safe_action)

# --- MAIN RECORDING LOOP ---
def main():
    warnings.filterwarnings('ignore')
    
    config = embodied.Config(dreamerv3.configs['defaults'])
    config = config.update(dreamerv3.configs['large'])
    config = config.update({
        'batch_size': 1,
        'run.actor_batch': 1,
        'jax.platform': 'gpu', 
        'encoder.mlp_keys': '$^', 
        'decoder.mlp_keys': '$^',
        'encoder.cnn_keys': 'image',
        'decoder.cnn_keys': 'image',
    })
    config = embodied.Flags(config).parse()
    
    def make_env():
        env_config = {"image_string": IMAGE_PATH, "config": "results/logs/vascular_large_collision_smaller_bubbles_game/env_config_default_segmentation_vascular_large.yaml", "reduce_radius": -0.01, "name": (0, 0), "render_mode": "rgb_array"}
        env = MicrorobotEnvGameByTheWall(**env_config)
        env = MaxAndSkipEnv(env, 4)
        env = MPCSafetyShield(env)
        env = WaypointGuidedWrapper(env, IMAGE_PATH)
        env = from_gym.FromGym(env)
        env = dreamerv3.wrap_env(env, config)
        return env

    env = embodied.BatchEnv([make_env()], parallel=False)
    
    step = embodied.Counter()
    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
    
    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    print(f"Loading checkpoint: {MODEL_PATH}")
    checkpoint.load(MODEL_PATH, keys=['agent'])
    
    print(f"Recording {NUM_EPISODES} episodes...")
    frames = []
    
    for episode in range(NUM_EPISODES):
        print(f"Starting Episode {episode + 1}...")
        
        act = {k: np.zeros((1,) + v.shape, v.dtype) for k, v in env.act_space.items()}
        act['reset'] = np.array([True])
        
        obs = env.step(act)
        state = None 
        done = False
        step_count = 0
        
        while not done:
            outs, state = agent.policy(obs, state, mode='eval')
            
            act = {k: v for k, v in outs.items()}
            if 'reset' not in act: act['reset'] = np.array([False])

            obs = env.step(act)
            done = obs['is_last'][0]
            
            if 'image' in obs:
                frame = obs['image'][0] 
                if frame.dtype != np.uint8: frame = (frame * 255).astype(np.uint8)
                frames.append(frame)
            
            step_count += 1
            if step_count > 1000: break
        
        print(f"Episode {episode + 1} finished in {step_count} steps.")

    env.close()

    if frames:
        os.makedirs(os.path.dirname(VIDEO_PATH_MP4), exist_ok=True)
        print(f"Saving {len(frames)} frames...")
        
        # --- ROBUST SAVE LOGIC ---
        try:
            # Try Saving as MP4 (Preferred)
            print(f"Attempting to save MP4 to {VIDEO_PATH_MP4}...")
            imageio.mimsave(VIDEO_PATH_MP4, frames, fps=30, plugin='ffmpeg')
            print("SUCCESS: MP4 Video Saved!")
        except Exception as e:
            print(f"WARNING: MP4 save failed ({e}). Falling back to GIF...")
            try:
                # Fallback to GIF (Universal)
                imageio.mimsave(VIDEO_PATH_GIF, frames, fps=30)
                print(f"SUCCESS: GIF Saved to {VIDEO_PATH_GIF}")
            except Exception as e2:
                print(f"CRITICAL: Could not save video. {e2}")
    else:
        print("Error: No frames captured.")

if __name__ == '__main__':
    main()
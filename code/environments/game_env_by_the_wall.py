import pygame
import cv2
import numpy as np
from .game_env_no_collision import MicrorobotEnvGameNoCollision
from utils.path_planning_v2 import RRTStar
from PIL import Image


class MicrorobotEnvGameByTheWall(MicrorobotEnvGameNoCollision):
    """
    Microrobot navigation environment with wall-collision awareness.

    [Paper 1 — GAP_SAC, Springer 2025] Heuristic Composite Reward Upgrade:
      Adds two auxiliary reward terms to the existing inverse-distance base reward:

      1. Yaw Alignment Reward (Section 3.4 of Paper 1):
         Rewards the agent for moving TOWARD the current target waypoint.
         Penalises lateral/perpendicular motion.
         Formula: r_yaw = yaw_weight * cos(angle between velocity and to-target)
         Effect:  Reduces oscillatory, wandering behaviour in open vessel regions.

      2. Wall Proximity Penalty (Section 3.4 of Paper 1):
         Applies a small negative reward when the agent enters a region
         within wall_penalty_dist pixels of any vessel wall.
         Effect:  Encourages the agent to stay in vessel centres,
                  reducing MPC shield interventions.

    These terms are ADDITIVE — the original reward_function(distance) is kept.
    Requires DreamerV3 fine-tuning to learn the updated reward signal.
    """

    # ── [Paper 1] Heuristic reward hyperparameters ────────────────────────────
    YAW_WEIGHT         = 0.05    # scale for yaw alignment bonus/penalty
    WALL_PENALTY       = -0.02   # fixed penalty for being near a wall
    WALL_PENALTY_DIST  = 12      # pixel distance threshold for wall proximity

    def __init__(self, inv_img_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if inv_img_path is None:
            img = cv2.imread(kwargs["image_string"], cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            inv_dil_obs = RRTStar.inverse_dilation((np.asarray(img, dtype=np.uint8)/255).astype(np.uint8),
                                                            self.config["General_environment_settings"]["DILATION"],
                                                            True)
            img = Image.fromarray((inv_dil_obs*255).astype(np.uint8)).convert('RGB')
            img.save(f"{kwargs['image_string'][:-4]}_inv_dil.png")
            self.inv_dil_obs = pygame.image.load(f"{kwargs['image_string'][:-4]}_inv_dil.png")

        else:
            self.inv_dil_obs = pygame.image.load(inv_img_path)

        self.inv_dil_obs = pygame.transform.scale(self.inv_dil_obs, (self.img_size, self.img_size))
        self.inv_dil_obs = pygame.surfarray.pixels3d(self.inv_dil_obs)
        self.reward_center = self.config['Reward Settings']['reward_center']
        self.flow_direction = np.array(self.config['General_environment_settings']['FLOW_DIRECTION'])
        self._flow = any(self.flow_direction) or self.reward_center != 0

        # [Paper 1] Track previous agent position for velocity-based yaw reward
        self._prev_agent_location = None

    def reset(self, **kwargs):
        """Reset the environment and clear previous position tracker."""
        result = super().reset(**kwargs)
        self._prev_agent_location = None   # [Paper 1] clear on episode reset
        return result

    def _post_step(self, direction, amplitude):
        reward = 0
        if self.is_valid(direction, amplitude, radius=self.get_radius()+self.tolerance_collision):
            self.agent_location = self.move_agent(direction, amplitude)
        elif self.check_collision(*self.agent_location, self.get_radius()):
            if self.is_valid(direction, amplitude/2, radius=self.get_radius()/4):
                self.agent_location = self.move_agent(direction, amplitude/2)
            else:
                reward += self.reward_collision
                self._update_radius(self.reduce_radius/2)
        else:
            self.agent_location = self.move_agent(direction, self.get_radius()/1)

        if self._flow:
            if self._eval_pixel_collision(self.agent_location[0].astype(int),
                                        self.agent_location[1].astype(int), self.inv_dil_obs):
                reward += self.reward_center
                if self.is_valid(self.flow_direction, amplitude/1.5, radius=self.get_radius()/1.5):
                    self.agent_location = self.move_agent(self.flow_direction, amplitude)
                else:
                    self._update_radius(self.reduce_radius)

        self.terminated = False
        self.truncated = False
        self._elapsed_steps += 1
        tol_target = self.get_radius() + self.tolerance_target_reached

        if np.allclose(self.agent_location, self.target_location, atol=tol_target):
            reward = self.reward_target_reached
            self.terminated = True
            self.target_reached = True
            self.count_target += 1

        elif self.get_radius() == self.min_bubble_radius:
            self.terminated = True
            reward = self.reward_termination
            self._reset_radius()

        else:
            # ── Base distance reward (unchanged) ─────────────────────────────
            distance = self._get_norm_dist(self.agent_location, self.target_location)
            reward  += self.reward_function(distance)

            # ── [Paper 1] Yaw Alignment Reward ───────────────────────────────
            # Measures angle between actual movement direction and direction to target.
            # Positive when moving toward target, negative when moving away/sideways.
            # Encourages directional efficiency; reduces oscillatory wandering.
            if self._prev_agent_location is not None:
                vel     = self.agent_location - self._prev_agent_location
                to_tgt  = self.target_location - self.agent_location
                norm_v  = np.linalg.norm(vel)
                norm_t  = np.linalg.norm(to_tgt)
                if norm_v > 0.5 and norm_t > 0.5:
                    cos_angle = np.dot(vel, to_tgt) / (norm_v * norm_t)
                    reward   += self.YAW_WEIGHT * float(cos_angle)

            # ── [Paper 1] Wall Proximity Penalty ─────────────────────────────
            # Applies a small negative reward when agent enters a region near
            # vessel walls. Uses the existing _fast_obstacles map (no new data).
            # Encourages paths that stay in vessel centres → fewer MPC interventions.
            #
            # BUG FIX: _eval_pixel_collision(x, y) treats x as the first axis
            # of the pygame surfarray (obstacles[x, y]) where x=column-index and
            # y=row-index. np.meshgrid(xs, ys) produces xx of shape
            # (len_ys, len_xs) — i.e. xx[i,j]=xs[j] and yy[i,j]=ys[i].
            # So we must pass xx (x/col values) as first arg and yy (y/row) as
            # second. Clipping both to [0, img_size-1] prevents out-of-bounds.
            if hasattr(self, '_fast_obstacles') and self._fast_obstacles is not None:
                try:
                    ax = int(np.clip(self.agent_location[0], 0, self.img_size - 1))
                    ay = int(np.clip(self.agent_location[1], 0, self.img_size - 1))
                    d  = self.WALL_PENALTY_DIST
                    # xs = x-coordinates (columns), ys = y-coordinates (rows)
                    xs = np.arange(max(0, ax - d), min(self.img_size, ax + d + 1))
                    ys = np.arange(max(0, ay - d), min(self.img_size, ay + d + 1))
                    # meshgrid: xx[i,j]=xs[j] (col), yy[i,j]=ys[i] (row)
                    xx, yy = np.meshgrid(xs, ys)
                    # _eval_pixel_collision expects (x_array, y_array)
                    if self._eval_pixel_collision(xx.ravel(), yy.ravel()):
                        reward += self.WALL_PENALTY
                except Exception:
                    pass   # skip if obstacle map shape is incompatible

            if self.verbose == 2:
                print("distance: ", distance)
                print("reward: ", reward)

        # ── [Paper 1] Update previous position for next step ─────────────────
        self._prev_agent_location = self.agent_location.copy()

        done = self.terminated
        return self._get_obs(), reward, done, self._get_info()
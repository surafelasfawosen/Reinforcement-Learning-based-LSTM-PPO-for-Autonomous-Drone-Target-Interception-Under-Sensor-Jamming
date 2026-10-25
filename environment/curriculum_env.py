"""
Curriculum Drone Anti-Jamming Gymnasium Environment
Implements 4-Stage Progressive Curriculum Learning:
  Stage 1: Close-proximity Hover & Intercept (1.0m - 2.5m, stationary, clean)
  Stage 2: Full-Arena Stationary Intercept (2.0m - 6.0m, mild noise)
  Stage 3: Dynamic Moving Target Pursuit (linear motion, boundary bounces)
  Stage 4: Full Electronic Warfare (all 4 adversarial jamming modes + evasive targets)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from environment.drone_dynamics import DroneDynamics
from environment.jamming_model import JammingModel, JammingMode
from environment.target_manager import TargetManager, TargetMode


class CurriculumDroneEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        render_mode: str = None,
        max_episode_steps: int = 400,
        stage: int = 1,
        domain_randomization: bool = True
    ):
        super().__init__()

        # --- Observation Space (13-D) ---
        # [dx, dy, dz, vx, vy, vz, wr, wp, wy, gx, gy, gz, J_flag]
        obs_low = np.array([-15.0, -15.0, -15.0, -10.0, -10.0, -10.0, -np.pi, -np.pi, -np.pi, -1.0, -1.0, -1.0, 0.0], dtype=np.float32)
        obs_high = np.array([15.0,  15.0,  15.0,  10.0,  10.0,  10.0,  np.pi,  np.pi,  np.pi,  1.0,  1.0,  1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # --- Action Space (4-D Continuous) ---
        # [roll_rate, pitch_rate, yaw_rate, collective_thrust] in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.stage = stage
        self.domain_randomization = domain_randomization
        self.dt = 0.02

        # Physical arena limits
        self.arena_x = (-7.0, 7.0)
        self.arena_y = (-7.0, 7.0)
        self.arena_z = (0.1, 6.0)

        # Reward Weights
        self.w_closure = 3.0
        self.w_dist_penalty = 0.05
        self.w_hit = 120.0
        self.w_jerk = 0.05
        self.w_jam_drift = 0.05
        self.w_crash = 50.0
        self.intercept_threshold = 0.35  # meters

        # Subsystems
        self.drone = DroneDynamics(mass=1.0, arm_length=0.17, dt=self.dt)
        self.jamming = JammingModel()
        self.target = TargetManager()

        self.step_count = 0
        self.prev_distance = 0.0
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.episode_history = []

    def set_stage(self, stage: int):
        """Update curriculum difficulty stage (1 to 4)."""
        self.stage = max(1, min(4, stage))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.episode_history = []

        if options and "stage" in options:
            self.set_stage(options["stage"])

        # Domain Randomization (scaled with curriculum)
        mass_var = np.random.uniform(-0.10, 0.10) if (self.domain_randomization and self.stage >= 2) else 0.0
        wind = np.random.normal(0, 0.3, size=3).astype(np.float32) if (self.domain_randomization and self.stage >= 3) else np.zeros(3)

        drone_start = np.array([0.0, 0.0, 1.5], dtype=np.float32)
        self.drone.reset(start_pos=drone_start, mass_variation=mass_var, wind_gust=wind)

        # Target placement according to curriculum stage
        if self.stage == 1:
            # Stage 1: Close target (1.2m - 2.5m away)
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(1.2, 2.5)
            tgt_pos = np.array([
                drone_start[0] + dist * np.cos(angle),
                drone_start[1] + dist * np.sin(angle),
                np.random.uniform(1.2, 2.2)
            ], dtype=np.float32)
            self.target.reset(specific_mode=TargetMode.STATIONARY, initial_pos=tgt_pos)
            self.jamming.reset(force_jammed=False)

        elif self.stage == 2:
            # Stage 2: Medium/Far target (2.5m - 5.0m away), stationary
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(2.5, 5.0)
            tgt_pos = np.array([
                drone_start[0] + dist * np.cos(angle),
                drone_start[1] + dist * np.sin(angle),
                np.random.uniform(1.0, 3.5)
            ], dtype=np.float32)
            self.target.reset(specific_mode=TargetMode.STATIONARY, initial_pos=tgt_pos)
            self.jamming.reset(force_jammed=False)

        elif self.stage == 3:
            # Stage 3: Moving target with linear velocity
            self.target.reset(specific_mode=TargetMode.LINEAR)
            self.jamming.reset(force_jammed=(np.random.random() < 0.3), specific_mode=JammingMode.GAUSSIAN)

        else:
            # Stage 4: Full Electronic Warfare (Random attack mode, evasive target)
            tgt_mode = TargetMode.SINUSOIDAL if np.random.random() < 0.5 else TargetMode.LINEAR
            self.target.reset(specific_mode=tgt_mode)
            attack_mode = np.random.choice([
                JammingMode.GAUSSIAN,
                JammingMode.SPIKE,
                JammingMode.BLACKOUT,
                JammingMode.DRIFT
            ])
            force_jam = options.get("force_jamming", False) if options else False
            self.jamming.reset(force_jammed=force_jam, specific_mode=attack_mode)

        self.prev_distance = float(np.linalg.norm(self.drone.pos - self.target.position))
        obs = self._get_observation()
        info = self._get_info(self.prev_distance, hit=False, crash=False)
        return obs, info

    def step(self, action: np.ndarray):
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        target_pos = self.target.step(self.dt)
        self.drone.step(action)
        drone_pos = self.drone.pos

        # Update jamming status
        if self.stage >= 3:
            self.jamming.step(current_step=self.step_count)
        else:
            self.jamming.is_jammed = False

        current_distance = float(np.linalg.norm(drone_pos - target_pos))
        hit = current_distance < self.intercept_threshold

        out_of_bounds = (
            drone_pos[0] < self.arena_x[0] or drone_pos[0] > self.arena_x[1] or
            drone_pos[1] < self.arena_y[0] or drone_pos[1] > self.arena_y[1] or
            drone_pos[2] < self.arena_z[0] or drone_pos[2] > self.arena_z[1]
        )
        crash = bool(out_of_bounds or drone_pos[2] < 0.15)

        terminated = bool(hit or crash)
        truncated = bool(self.step_count >= self.max_episode_steps)

        reward = self._compute_reward(current_distance, action, hit, crash)
        self.prev_distance = current_distance
        self.prev_action = action.copy()

        obs = self._get_observation()
        info = self._get_info(current_distance, hit, crash)
        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        true_p_rel = self.target.position - self.drone.pos
        obs_p_rel = self.jamming.corrupt_relative_position(true_p_rel, clip_range=15.0)

        v_drone, w_drone = self.drone.get_imu_readings()
        g_body = self.drone.get_gravity_vector()

        v_drone = np.clip(v_drone, -10.0, 10.0)
        w_drone = np.clip(w_drone, -np.pi, np.pi)
        j_flag_val = 1.0 if self.jamming.is_jammed else 0.0

        return np.concatenate([
            obs_p_rel,
            v_drone,
            w_drone,
            g_body,
            np.array([j_flag_val], dtype=np.float32)
        ]).astype(np.float32)

    def _compute_reward(self, current_distance: float, action: np.ndarray, hit: bool, crash: bool) -> float:
        # Distance closure gradient
        closure = self.w_closure * (self.prev_distance - current_distance)

        # Line-of-sight closing velocity
        true_p_rel = self.target.position - self.drone.pos
        p_rel_norm = max(current_distance, 0.1)
        unit_los = true_p_rel / p_rel_norm
        rel_vel = self.drone.vel - self.target.velocity
        closing_vel = float(-np.dot(rel_vel, unit_los))
        closing_reward = 0.4 * np.clip(closing_vel, -2.0, 4.0)

        # Upright attitude shaping
        g_body = self.drone.get_gravity_vector()
        upright_reward = 0.15 * float(g_body[2])

        dist_cost = -self.w_dist_penalty * current_distance
        hit_bonus = 150.0 if hit else 0.0
        jerk_cost = -self.w_jerk * float(np.sum((action - self.prev_action) ** 2))

        jam_penalty = 0.0
        if self.jamming.is_jammed:
            jam_penalty = -self.w_jam_drift * current_distance

        crash_cost = -25.0 if crash else 0.0

        return float(closure + closing_reward + upright_reward + dist_cost + hit_bonus + jerk_cost + jam_penalty + crash_cost)

    def _get_info(self, current_distance: float, hit: bool, crash: bool) -> dict:
        return {
            "distance": current_distance,
            "hit": hit,
            "crash": crash,
            "stage": self.stage,
            "J_flag": int(self.jamming.is_jammed),
            "jamming_mode": self.jamming.active_mode.name,
            "drone_pos": self.drone.pos.copy(),
            "target_pos": self.target.position.copy(),
            "step": self.step_count
        }

    def close(self):
        pass

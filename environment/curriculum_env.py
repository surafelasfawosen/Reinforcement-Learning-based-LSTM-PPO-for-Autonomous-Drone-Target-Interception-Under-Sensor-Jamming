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
        self.w_hit = 150.0
        self.w_jerk = 0.05
        self.w_jam_drift = 0.05
        self.w_crash = 25.0
        # Intercept threshold scales from generous (1.0m) in Stage 1 to tight (0.35m) in Stage 4.
        # This ensures the agent gets early positive experiences before the problem gets hard.
        self.stage_thresholds = {1: 1.0, 2: 0.6, 3: 0.45, 4: 0.35}
        self.intercept_threshold = self.stage_thresholds.get(self.stage, 0.35)

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
        self.intercept_threshold = self.stage_thresholds.get(self.stage, 0.35)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.episode_history = []

        if options and "stage" in options:
            self.set_stage(options["stage"])

        # Domain Randomization (scaled with curriculum — disabled in Stage 1 to prevent distraction)
        mass_var = np.random.uniform(-0.10, 0.10) if (self.domain_randomization and self.stage >= 3) else 0.0
        wind = np.random.normal(0, 0.3, size=3).astype(np.float32) if (self.domain_randomization and self.stage >= 4) else np.zeros(3)

        drone_start = np.array([0.0, 0.0, 1.5], dtype=np.float32)
        self.drone.reset(start_pos=drone_start, mass_variation=mass_var, wind_gust=wind)

        # Target placement according to curriculum stage
        if self.stage == 1:
            # Stage 1: Very close target (0.4m - 1.2m).  Threshold=1.0m so the agent can
            # physically reach it and get reward in its FIRST episodes, bootstrapping learning.
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(0.4, 1.2)
            tgt_pos = np.array([
                drone_start[0] + dist * np.cos(angle),
                drone_start[1] + dist * np.sin(angle),
                np.clip(drone_start[2] + np.random.uniform(-0.3, 0.3), 0.5, 2.5)
            ], dtype=np.float32)
            self.target.reset(specific_mode=TargetMode.STATIONARY, initial_pos=tgt_pos)
            self.jamming.reset(force_jammed=False)

        elif self.stage == 2:
            # Stage 2: Medium target (1.5m - 3.0m away), stationary, no jamming
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(1.5, 3.0)
            tgt_pos = np.array([
                drone_start[0] + dist * np.cos(angle),
                drone_start[1] + dist * np.sin(angle),
                np.random.uniform(0.8, 3.0)
            ], dtype=np.float32)
            self.target.reset(specific_mode=TargetMode.STATIONARY, initial_pos=tgt_pos)
            self.jamming.reset(force_jammed=False)

        elif self.stage == 3:
            # Stage 3: Moving target (standoff 2.5m - 4.5m), mild jamming
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(2.5, 4.5)
            tgt_pos = np.array([
                drone_start[0] + dist * np.cos(angle),
                drone_start[1] + dist * np.sin(angle),
                np.random.uniform(1.0, 3.5)
            ], dtype=np.float32)
            self.target.reset(specific_mode=TargetMode.LINEAR, initial_pos=tgt_pos)
            self.jamming.reset(force_jammed=(np.random.random() < 0.3), specific_mode=JammingMode.GAUSSIAN)

        else:
            # Stage 4: Full Electronic Warfare (Random attack mode, evasive target)
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(2.5, 5.0)
            tgt_pos = np.array([
                drone_start[0] + dist * np.cos(angle),
                drone_start[1] + dist * np.sin(angle),
                np.random.uniform(1.2, 3.5)
            ], dtype=np.float32)
            tgt_mode = TargetMode.SINUSOIDAL if np.random.random() < 0.5 else TargetMode.LINEAR
            self.target.reset(specific_mode=tgt_mode, initial_pos=tgt_pos)
            attack_mode = np.random.choice([
                JammingMode.GAUSSIAN,
                JammingMode.SPIKE,
                JammingMode.BLACKOUT,
                JammingMode.DRIFT
            ])
            force_jam = options.get("force_jamming", (np.random.random() < 0.6)) if options else (np.random.random() < 0.6)
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

        # --- Soft boundary handling (Stage 1 & 2) vs hard termination (Stage 3+) ---
        # In early stages, clamp position to arena instead of terminating, so the agent
        # gets more steps to learn — prevents 90%+ crash rate killing all learning.
        ground_crash = bool(drone_pos[2] < 0.08)
        out_of_bounds = (
            drone_pos[0] < self.arena_x[0] or drone_pos[0] > self.arena_x[1] or
            drone_pos[1] < self.arena_y[0] or drone_pos[1] > self.arena_y[1] or
            drone_pos[2] < self.arena_z[0] or drone_pos[2] > self.arena_z[1]
        )

        if self.stage <= 2 and out_of_bounds and not ground_crash:
            # Soft wall: reflect velocity, clamp position, apply small penalty — don't terminate
            self.drone.pos[0] = np.clip(self.drone.pos[0], self.arena_x[0] + 0.1, self.arena_x[1] - 0.1)
            self.drone.pos[1] = np.clip(self.drone.pos[1], self.arena_y[0] + 0.1, self.arena_y[1] - 0.1)
            self.drone.pos[2] = np.clip(self.drone.pos[2], self.arena_z[0] + 0.1, self.arena_z[1] - 0.1)
            self.drone.vel *= -0.3  # Reflect and damp velocity
            crash = False
            out_of_bounds = False  # Reset after clamping
        else:
            crash = bool(out_of_bounds or ground_crash)

        terminated = bool(hit or crash)
        truncated = bool(self.step_count >= self.max_episode_steps)

        reward = self._compute_reward(current_distance, action, hit, crash)
        self.prev_distance = current_distance
        self.prev_action = action.copy()

        obs = self._get_observation()
        info = self._get_info(current_distance, hit, crash)
        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        # Rotation matrix from body to world (R.T transforms world -> body)
        R = self.drone._euler_to_rotation_matrix(self.drone.rpy)
        R_inv = R.T

        # Target relative position in drone BODY frame (resolves heading/yaw ambiguity)
        true_p_rel_world = self.target.position - self.drone.pos
        true_p_rel_body = R_inv @ true_p_rel_world

        # Corrupt relative position in sensor frame
        obs_p_rel = self.jamming.corrupt_relative_position(true_p_rel_body, clip_range=15.0)

        # Onboard IMU velocity projected in body frame
        v_drone_world, w_drone = self.drone.get_imu_readings()
        v_drone_body = R_inv @ v_drone_world
        g_body = self.drone.get_gravity_vector()

        v_drone_body = np.clip(v_drone_body, -10.0, 10.0)
        w_drone = np.clip(w_drone, -np.pi, np.pi)
        j_flag_val = 1.0 if self.jamming.is_jammed else 0.0

        return np.concatenate([
            obs_p_rel,
            v_drone_body,
            w_drone,
            g_body,
            np.array([j_flag_val], dtype=np.float32)
        ]).astype(np.float32)

    def _compute_reward(self, current_distance: float, action: np.ndarray, hit: bool, crash: bool) -> float:
        # Distance closure gradient (primary learning signal)
        closure = self.w_closure * (self.prev_distance - current_distance)

        # Line-of-sight closing velocity: v_rel . unit_los (positive when approaching target)
        true_p_rel = self.target.position - self.drone.pos
        p_rel_norm = max(current_distance, 0.1)
        unit_los = true_p_rel / p_rel_norm
        rel_vel = self.drone.vel - self.target.velocity
        closing_vel = float(np.dot(rel_vel, unit_los))
        closing_reward = 0.5 * np.clip(closing_vel, -1.0, 4.0)

        # Upright attitude shaping (penalize excessive tilt, prevents upside-down hovering)
        g_body = self.drone.get_gravity_vector()
        upright_penalty = -0.3 * float(max(0.0, 1.0 - g_body[2]))

        # Proximity shaping: dense gradient that increases strongly near the target
        proximity_bonus = 0.5 / (current_distance + 0.1)  # grows as distance shrinks

        dist_cost = -self.w_dist_penalty * current_distance
        hit_bonus = self.w_hit if hit else 0.0
        jerk_cost = -self.w_jerk * float(np.sum((action - self.prev_action) ** 2))

        jam_penalty = 0.0
        if self.jamming.is_jammed:
            jam_penalty = -self.w_jam_drift * current_distance

        crash_cost = -self.w_crash if crash else 0.0

        return float(closure + closing_reward + upright_penalty + proximity_bonus +
                     dist_cost + hit_bonus + jerk_cost + jam_penalty + crash_cost)

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

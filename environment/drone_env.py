"""
Drone Anti-Jamming Gymnasium Environment (POMDP)
Formulates the adversarial electronic warfare interception problem:
  - State: 3D Quadrotor dynamics + Target kinematics
  - Observation (10D): Corrupted p_rel under jamming, clean IMU v and w, and J_flag
  - Action (4D): [roll_rate, pitch_rate, yaw_rate, collective_thrust] in [-1, 1]
  - Reward: Distance closure, terminal hit bonus, jerk penalty, jammed tracking penalty, crash penalty
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from environment.drone_dynamics import DroneDynamics
from environment.jamming_model import JammingModel, JammingMode
from environment.target_manager import TargetManager, TargetMode


class DroneAntiJammingEnv(gym.Env):
    """
    POMDP Environment for Autonomous Drone Interception under Sensor Spoofing.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        render_mode: str = None,
        max_episode_steps: int = 500,
        target_mode: TargetMode = TargetMode.STATIONARY,
        jamming_mode: JammingMode = JammingMode.GAUSSIAN,
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

        # Configuration
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.default_target_mode = target_mode
        self.default_jamming_mode = jamming_mode
        self.domain_randomization = domain_randomization
        self.dt = 0.02  # 50 Hz control loop

        # Arena bounds (meters)
        self.arena_x = (-8.0, 8.0)
        self.arena_y = (-8.0, 8.0)
        self.arena_z = (0.1, 7.0)

        # Reward weights
        self.w_closure = 2.0
        self.w_hit = 100.0
        self.w_jerk = 0.08
        self.w_jam_drift = 0.04
        self.w_crash = 60.0
        self.intercept_threshold = 1.2  # meters (aerial interception acquisition basket)

        # Subsystems
        self.drone = DroneDynamics(mass=1.0, arm_length=0.17, dt=self.dt)
        self.jamming = JammingModel(default_mode=jamming_mode)
        self.target = TargetManager(mode=target_mode)

        # Episode state variables
        self.step_count = 0
        self.prev_distance = 0.0
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.trajectory_history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.trajectory_history = []

        # Domain Randomization
        mass_var = np.random.uniform(-0.15, 0.15) if self.domain_randomization else 0.0
        wind = np.random.normal(0, 0.5, size=3).astype(np.float32) if self.domain_randomization else np.zeros(3)

        # Drone start position: near origin with random jitter
        drone_start = np.array([
            np.random.uniform(-1.0, 1.0),
            np.random.uniform(-1.0, 1.0),
            np.random.uniform(1.0, 2.0)
        ], dtype=np.float32)
        self.drone.reset(start_pos=drone_start, mass_variation=mass_var, wind_gust=wind)

        # Target reset
        target_pos = self.target.reset(specific_mode=self.default_target_mode)

        # Jamming reset
        force_jam = False
        if options and "force_jamming" in options:
            force_jam = options["force_jamming"]
        self.jamming.reset(force_jammed=force_jam, specific_mode=self.default_jamming_mode)

        self.prev_distance = float(np.linalg.norm(self.drone.pos - target_pos))

        obs = self._get_observation()
        info = self._get_info(current_distance=self.prev_distance, hit=False, crash=False)
        return obs, info

    def step(self, action: np.ndarray):
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # 1. Step Target kinematics
        target_pos = self.target.step(self.dt)

        # 2. Step Drone 6-DOF physics
        self.drone.step(action)
        drone_pos = self.drone.pos

        # 3. Step Jamming state machine
        self.jamming.step(current_step=self.step_count)

        # 4. Compute Distance & Termination Conditions
        current_distance = float(np.linalg.norm(drone_pos - target_pos))
        hit = current_distance < self.intercept_threshold

        # Crash if below ground or out of arena bounds
        out_of_bounds = (
            drone_pos[0] < self.arena_x[0] or drone_pos[0] > self.arena_x[1] or
            drone_pos[1] < self.arena_y[0] or drone_pos[1] > self.arena_y[1] or
            drone_pos[2] < self.arena_z[0] or drone_pos[2] > self.arena_z[1]
        )
        crash = bool(out_of_bounds or drone_pos[2] < 0.15)

        terminated = bool(hit or crash)
        truncated = bool(self.step_count >= self.max_episode_steps)

        # 5. Compute Reward
        reward = self._compute_reward(current_distance, action, hit, crash)
        self.prev_distance = current_distance
        self.prev_action = action.copy()

        # Log trajectory step
        self.trajectory_history.append({
            "step": self.step_count,
            "drone_pos": drone_pos.copy(),
            "target_pos": target_pos.copy(),
            "J_flag": int(self.jamming.is_jammed),
            "distance": current_distance
        })

        obs = self._get_observation()
        info = self._get_info(current_distance, hit, crash)
        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Extracts the 13-D observation vector for the agent."""
        # True relative vector from drone to target
        true_p_rel = self.target.position - self.drone.pos

        # Jammed observation (corrupts p_rel if J_flag == 1)
        obs_p_rel = self.jamming.corrupt_relative_position(true_p_rel, clip_range=15.0)

        # Clean onboard IMU readings (hardware local — never jammed)
        v_drone, w_drone = self.drone.get_imu_readings()
        g_body = self.drone.get_gravity_vector()

        v_drone = np.clip(v_drone, -10.0, 10.0)
        w_drone = np.clip(w_drone, -np.pi, np.pi)
        j_flag_val = 1.0 if self.jamming.is_jammed else 0.0

        obs = np.concatenate([
            obs_p_rel,
            v_drone,
            w_drone,
            g_body,
            np.array([j_flag_val], dtype=np.float32)
        ]).astype(np.float32)

        return obs

    def _compute_reward(self, current_distance: float, action: np.ndarray, hit: bool, crash: bool) -> float:
        """Computes the enhanced multi-objective reward signal."""
        # 1. Distance closure reward (potential-based dense gradient)
        closure_reward = self.w_closure * (self.prev_distance - current_distance)

        # 2. Line-of-Sight closing velocity: - (v_rel . unit_los)
        true_p_rel = self.target.position - self.drone.pos
        p_rel_norm = max(current_distance, 0.1)
        unit_los = true_p_rel / p_rel_norm
        rel_vel = self.drone.vel - self.target.velocity
        closing_vel = float(-np.dot(rel_vel, unit_los))
        closing_reward = 0.4 * np.clip(closing_vel, -2.0, 4.0)

        # 3. Upright attitude shaping (encourage keeping vertical thrust aligned with gravity)
        g_body = self.drone.get_gravity_vector()
        upright_reward = 0.15 * float(g_body[2])  # g_body[2] = cos(pitch)*cos(roll)

        # 4. Stand-off proximity gradient
        dist_cost = -0.05 * current_distance

        # 5. Terminal interception bonus
        hit_bonus = 150.0 if hit else 0.0

        # 6. Action jerk penalty (actuator rate smoothing)
        action_diff = action - self.prev_action
        jerk_penalty = -self.w_jerk * float(np.sum(action_diff ** 2))

        # 7. Jammed tracking error penalty (forces LSTM dead-reckoning)
        jam_penalty = 0.0
        if self.jamming.is_jammed:
            jam_penalty = -self.w_jam_drift * current_distance

        # 8. Controlled crash penalty (avoids policy paralysis)
        crash_penalty = -25.0 if crash else 0.0

        total_reward = (closure_reward + closing_reward + upright_reward + dist_cost +
                        hit_bonus + jerk_penalty + jam_penalty + crash_penalty)
        return float(total_reward)

    def _get_info(self, current_distance: float, hit: bool, crash: bool) -> dict:
        """Returns metadata for metrics and evaluation logging."""
        return {
            "distance": current_distance,
            "hit": hit,
            "crash": crash,
            "J_flag": int(self.jamming.is_jammed),
            "jamming_mode": self.jamming.active_mode.name,
            "drone_pos": self.drone.pos.copy(),
            "target_pos": self.target.position.copy(),
            "step": self.step_count
        }

    def close(self):
        pass

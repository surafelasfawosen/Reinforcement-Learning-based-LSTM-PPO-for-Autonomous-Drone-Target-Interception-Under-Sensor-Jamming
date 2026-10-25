"""
Drone Dynamics Module
Implements robust, high-fidelity 6-DOF Quadrotor Dynamics with first-order actuator lag
and Euler attitude kinematics. Guaranteed numerically stable under all control inputs.
"""

import numpy as np


class DroneDynamics:
    """
    6-DOF Quadrotor Dynamics Engine.
    Uses first-order body-rate response dynamics and translational Newton-Euler equations.
    """
    def __init__(
        self,
        mass: float = 1.0,
        arm_length: float = 0.17,
        dt: float = 0.02
    ):
        self.nominal_mass = mass
        self.mass = mass
        self.arm_length = arm_length
        self.dt = dt

        # Physical constants
        self.g = 9.81
        self.drag_coeff = 0.15
        self.max_speed = 8.0  # m/s

        # Actuator time constant (seconds for motor/body-rate response)
        self.tau_omega = 0.06  # Fast rate response (~60ms)
        self.max_rate_cmd = np.array([np.pi * 1.5, np.pi * 1.5, np.pi], dtype=np.float32)

        # Max tilt angle (rad) to prevent upside-down inverted flight
        self.max_tilt = 1.25  # ~72 degrees

        # Thrust limits (N)
        self.hover_thrust = self.nominal_mass * self.g
        self.max_thrust = 2.2 * self.hover_thrust
        self.min_thrust = 0.1 * self.hover_thrust

        # State Variables
        self.pos = np.zeros(3, dtype=np.float32)
        self.vel = np.zeros(3, dtype=np.float32)
        self.rpy = np.zeros(3, dtype=np.float32)       # [roll, pitch, yaw]
        self.omega = np.zeros(3, dtype=np.float32)     # [p, q, r] body rates

        # Environmental disturbances
        self.wind = np.zeros(3, dtype=np.float32)

    def reset(
        self,
        start_pos: np.ndarray = None,
        mass_variation: float = 0.0,
        wind_gust: np.ndarray = None
    ):
        """Reset drone state and apply domain randomization."""
        self.mass = self.nominal_mass * (1.0 + mass_variation)
        self.hover_thrust = self.mass * self.g
        self.max_thrust = 2.2 * self.hover_thrust

        if start_pos is not None:
            self.pos = np.array(start_pos, dtype=np.float32)
        else:
            self.pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        self.vel = np.zeros(3, dtype=np.float32)
        self.rpy = np.zeros(3, dtype=np.float32)
        self.omega = np.zeros(3, dtype=np.float32)

        if wind_gust is not None:
            self.wind = np.array(wind_gust, dtype=np.float32)
        else:
            self.wind = np.zeros(3, dtype=np.float32)

    def _euler_to_rotation_matrix(self, rpy: np.ndarray) -> np.ndarray:
        """Rotation matrix from body frame to world frame (Z-Y-X Euler)."""
        roll, pitch, yaw = rpy
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,                cp * cr]
        ], dtype=np.float32)
        return R

    def step(self, action: np.ndarray):
        """
        Integrates 6-DOF physics over timestep dt.
        Action: [roll_rate, pitch_rate, yaw_rate, thrust_delta] in [-1, 1].
        """
        # 1. Target body rates
        omega_cmd = action[:3] * self.max_rate_cmd

        # First-order rate response: d_omega/dt = (omega_cmd - omega) / tau
        alpha = self.dt / (self.tau_omega + self.dt)
        self.omega = (1.0 - alpha) * self.omega + alpha * omega_cmd

        # 2. Update attitude (Euler angles)
        roll, pitch, yaw = self.rpy
        cos_p = np.cos(pitch)
        if np.abs(cos_p) < 0.05:
            cos_p = 0.05 if cos_p >= 0 else -0.05
        tan_p = np.sin(pitch) / cos_p

        E = np.array([
            [1.0, np.sin(roll) * tan_p,  np.cos(roll) * tan_p],
            [0.0, np.cos(roll),         -np.sin(roll)],
            [0.0, np.sin(roll) / cos_p,  np.cos(roll) / cos_p]
        ], dtype=np.float32)

        rpy_dot = E @ self.omega
        self.rpy += rpy_dot * self.dt

        # Bound tilt angles to realistic envelope
        self.rpy[0] = np.clip(self.rpy[0], -self.max_tilt, self.max_tilt)
        self.rpy[1] = np.clip(self.rpy[1], -self.max_tilt, self.max_tilt)
        # Wrap yaw in [-pi, pi]
        self.rpy[2] = (self.rpy[2] + np.pi) % (2.0 * np.pi) - np.pi

        # 3. Collective thrust calculation
        # action[3] in [-1, 1] maps around hover thrust
        thrust_delta = action[3] * (self.hover_thrust * 0.9)
        thrust = np.clip(self.hover_thrust + thrust_delta, self.min_thrust, self.max_thrust)

        # 4. Translational acceleration
        R = self._euler_to_rotation_matrix(self.rpy)
        thrust_world = R @ np.array([0.0, 0.0, thrust], dtype=np.float32)
        gravity_force = np.array([0.0, 0.0, -self.mass * self.g], dtype=np.float32)
        drag_force = -self.drag_coeff * self.vel * np.abs(self.vel) + self.wind

        accel = (thrust_world + gravity_force + drag_force) / self.mass

        # 5. Integrate velocity & position
        self.vel += accel * self.dt
        self.vel = np.clip(self.vel, -self.max_speed, self.max_speed)
        self.pos += self.vel * self.dt

    def get_imu_readings(self, noise_std: float = 0.01) -> tuple:
        """
        Returns linear and angular velocities as perceived by the onboard IMU.
        Hardware local — unjammable.
        """
        v_noise = np.random.normal(0, noise_std, size=3).astype(np.float32)
        w_noise = np.random.normal(0, noise_std * 0.5, size=3).astype(np.float32)
        return (self.vel + v_noise), (self.omega + w_noise)

    def get_gravity_vector(self) -> np.ndarray:
        """
        Returns projected gravity unit vector in body frame: R^T * [0, 0, 1].
        Directly measurable by onboard 3-axis accelerometer (unfiltered low-frequency).
        Hardware local — unjammable.
        """
        roll, pitch, _ = self.rpy
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        return np.array([-sp, cp * sr, cp * cr], dtype=np.float32)

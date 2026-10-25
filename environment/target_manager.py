"""
Target Manager Module
Manages target position, velocity, and kinematics across curriculum stages:
  1. Stationary Target (fixed random 3D position)
  2. Linear Moving Target (constant velocity, boundary reflections)
  3. Sinusoidal / Evasive Target (harmonic figure-8 / 3D weave)
"""

import numpy as np
from enum import Enum


class TargetMode(Enum):
    STATIONARY = 0
    LINEAR = 1
    SINUSOIDAL = 2


class TargetManager:
    """
    Simulates target motion in 3D arena.
    """
    def __init__(
        self,
        mode: TargetMode = TargetMode.STATIONARY,
        arena_bounds: tuple = ((-6.0, 6.0), (-6.0, 6.0), (0.5, 4.5)),
        max_speed: float = 1.2
    ):
        self.mode = mode
        self.arena_bounds = arena_bounds
        self.max_speed = max_speed

        self.position = np.zeros(3, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.base_pos = np.zeros(3, dtype=np.float32)
        self.time = 0.0

        # Sinusoidal parameters
        self.amplitudes = np.array([1.5, 1.5, 0.8], dtype=np.float32)
        self.frequencies = np.array([0.5, 0.3, 0.4], dtype=np.float32)
        self.phases = np.zeros(3, dtype=np.float32)

    def reset(self, specific_mode: TargetMode = None, initial_pos: np.ndarray = None):
        """Reset target state at episode start."""
        if specific_mode is not None:
            self.mode = specific_mode

        self.time = 0.0

        if initial_pos is not None:
            self.position = np.array(initial_pos, dtype=np.float32)
        else:
            self.position = np.array([
                np.random.uniform(self.arena_bounds[0][0] + 1.0, self.arena_bounds[0][1] - 1.0),
                np.random.uniform(self.arena_bounds[1][0] + 1.0, self.arena_bounds[1][1] - 1.0),
                np.random.uniform(self.arena_bounds[2][0] + 0.5, self.arena_bounds[2][1] - 0.5)
            ], dtype=np.float32)

        self.base_pos = self.position.copy()

        if self.mode == TargetMode.STATIONARY:
            self.velocity = np.zeros(3, dtype=np.float32)

        elif self.mode == TargetMode.LINEAR:
            # Random velocity vector normalized to max_speed
            v_dir = np.random.randn(3).astype(np.float32)
            v_dir[2] *= 0.3  # Less vertical speed for realistic target
            norm = np.linalg.norm(v_dir)
            speed = np.random.uniform(0.5, self.max_speed)
            self.velocity = (v_dir / (norm + 1e-8)) * speed

        elif self.mode == TargetMode.SINUSOIDAL:
            self.phases = np.random.uniform(0, 2 * np.pi, size=3).astype(np.float32)
            self.velocity = np.zeros(3, dtype=np.float32)

        return self.position.copy()

    def step(self, dt: float) -> np.ndarray:
        """Advance target physics by dt seconds."""
        self.time += dt

        if self.mode == TargetMode.STATIONARY:
            # Position remains fixed
            return self.position.copy()

        elif self.mode == TargetMode.LINEAR:
            # Advance position
            self.position += self.velocity * dt

            # Soft boundary bounce
            for i in range(3):
                min_b, max_b = self.arena_bounds[i]
                if self.position[i] < min_b:
                    self.position[i] = min_b
                    self.velocity[i] *= -1.0
                elif self.position[i] > max_b:
                    self.position[i] = max_b
                    self.velocity[i] *= -1.0

            return self.position.copy()

        elif self.mode == TargetMode.SINUSOIDAL:
            # Harmonic weaving trajectory around base_pos
            offset = self.amplitudes * np.sin(self.frequencies * self.time + self.phases)
            new_pos = self.base_pos + offset

            # Compute finite-difference velocity
            self.velocity = (new_pos - self.position) / dt
            self.position = new_pos

            # Clip within arena
            for i in range(3):
                min_b, max_b = self.arena_bounds[i]
                self.position[i] = np.clip(self.position[i], min_b, max_b)

            return self.position.copy()

        return self.position.copy()

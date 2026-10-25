"""
Jamming Model Module
Simulates electronic warfare and adversarial sensor attacks on UAV relative positioning.
Implements 4 distinct attack modes:
  1. Gaussian Noise (multipath/sensor degradation)
  2. Coordinate Spike / Spoofing (active injection of false coordinates)
  3. Blackout (total loss of signal)
  4. Cumulative Drift (novel slow creeping error)
"""

import numpy as np
from enum import Enum


class JammingMode(Enum):
    NONE = 0
    GAUSSIAN = 1
    SPIKE = 2
    BLACKOUT = 3
    DRIFT = 4


class JammingModel:
    """
    Adversarial Jamming and Spoofing Engine.
    Controls dynamic state transitions and corrupts the relative position vector p_rel.
    """
    def __init__(
        self,
        noise_std: float = 2.5,
        spike_magnitude: float = 8.0,
        drift_rate: float = 0.05,
        p_jam_start: float = 0.015,
        p_jam_recover: float = 0.04,
        warmup_steps: int = 100,
        default_mode: JammingMode = JammingMode.GAUSSIAN
    ):
        self.noise_std = noise_std
        self.spike_magnitude = spike_magnitude
        self.drift_rate = drift_rate
        self.p_jam_start = p_jam_start
        self.p_jam_recover = p_jam_recover
        self.warmup_steps = warmup_steps
        self.default_mode = default_mode

        # Dynamic state
        self.is_jammed = False
        self.active_mode = JammingMode.NONE
        self.accumulated_drift = np.zeros(3, dtype=np.float32)
        self.drift_direction = np.zeros(3, dtype=np.float32)
        self.jamming_step_counter = 0

    def reset(self, force_jammed: bool = False, specific_mode: JammingMode = None):
        """Reset jamming state at the start of an episode."""
        self.is_jammed = force_jammed
        if specific_mode is not None:
            self.active_mode = specific_mode
        elif force_jammed:
            self.active_mode = self.default_mode
        else:
            self.active_mode = JammingMode.NONE

        self.accumulated_drift = np.zeros(3, dtype=np.float32)
        # Random initial drift direction
        d = np.random.randn(3).astype(np.float32)
        norm = np.linalg.norm(d)
        self.drift_direction = d / (norm + 1e-8)
        self.jamming_step_counter = 0

    def step(self, current_step: int, force_jammed: bool = None) -> bool:
        """
        Update the jamming status for the current timestep.
        Returns the current boolean jamming flag (J_flag).
        """
        if force_jammed is not None:
            self.is_jammed = force_jammed
            if self.is_jammed and self.active_mode == JammingMode.NONE:
                self.active_mode = self.default_mode
            elif not self.is_jammed:
                self.active_mode = JammingMode.NONE
            return self.is_jammed

        # If already forced or active from reset, preserve jamming
        if self.is_jammed:
            self.jamming_step_counter += 1
            # Sustained electronic attack (recovery only after sustained duration)
            if self.jamming_step_counter > 80 and np.random.random() < self.p_jam_recover:
                self.is_jammed = False
                self.active_mode = JammingMode.NONE
                self.accumulated_drift = np.zeros(3, dtype=np.float32)
            return self.is_jammed

        # Warmup period for stochastic transitions (only if not already jammed)
        if current_step < self.warmup_steps:
            return False

        # Markov transition logic for spontaneous jamming
        if np.random.random() < self.p_jam_start:
            self.is_jammed = True
            self.active_mode = self.default_mode
            self.jamming_step_counter = 0
            self.accumulated_drift = np.zeros(3, dtype=np.float32)

        return self.is_jammed

    def corrupt_relative_position(self, p_rel_true: np.ndarray, clip_range: float = 10.0) -> np.ndarray:
        """
        Applies adversarial corruption to the true relative position vector.
        IMU velocity/gyro are NOT corrupted (onboard hardware).
        """
        if not self.is_jammed or self.active_mode == JammingMode.NONE:
            return np.clip(p_rel_true.astype(np.float32), -clip_range, clip_range)

        corrupted = p_rel_true.copy().astype(np.float32)

        if self.active_mode == JammingMode.GAUSSIAN:
            # High-frequency Gaussian noise (multipath / noise jamming)
            noise = np.random.normal(0, self.noise_std, size=3).astype(np.float32)
            corrupted += noise

        elif self.active_mode == JammingMode.SPIKE:
            # Coordinate injection / spoofing (adversary broadcasts false coordinates)
            spike = np.random.uniform(-self.spike_magnitude, self.spike_magnitude, size=3).astype(np.float32)
            corrupted = spike

        elif self.active_mode == JammingMode.BLACKOUT:
            # Complete signal denial (sensor returns zeros)
            corrupted = np.zeros(3, dtype=np.float32)

        elif self.active_mode == JammingMode.DRIFT:
            # Slow cumulative drift (novel mode: creeps in over time)
            self.accumulated_drift += self.drift_direction * self.drift_rate
            corrupted += self.accumulated_drift

        return np.clip(corrupted, -clip_range, clip_range)

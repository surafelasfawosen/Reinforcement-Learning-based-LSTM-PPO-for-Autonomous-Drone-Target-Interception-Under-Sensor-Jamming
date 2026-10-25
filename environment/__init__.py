"""
Environment Package Registration
Registers DroneAntiJammingEnv with Gymnasium.
"""

from gymnasium.envs.registration import register
from environment.drone_env import DroneAntiJammingEnv
from environment.curriculum_env import CurriculumDroneEnv
from environment.jamming_model import JammingModel, JammingMode
from environment.target_manager import TargetManager, TargetMode
from environment.drone_dynamics import DroneDynamics

register(
    id="DroneAntiJammingEnv-v0",
    entry_point="environment.drone_env:DroneAntiJammingEnv",
    max_episode_steps=500,
)

register(
    id="CurriculumDroneEnv-v0",
    entry_point="environment.curriculum_env:CurriculumDroneEnv",
    max_episode_steps=400,
)

__all__ = [
    "DroneAntiJammingEnv",
    "CurriculumDroneEnv",
    "JammingModel",
    "JammingMode",
    "TargetManager",
    "TargetMode",
    "DroneDynamics"
]

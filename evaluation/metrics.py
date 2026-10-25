"""
Evaluation Metrics Module
Calculates rigorous academic benchmark metrics for UAV interception research.
"""

import numpy as np


def compute_episode_metrics(episode_trajectory: list, hit: bool, crash: bool) -> dict:
    """
    Computes statistical performance metrics from a single rollout trajectory.
    """
    if len(episode_trajectory) == 0:
        return {
            "success": 0,
            "crash": 1,
            "rmse_tracking_error": 0.0,
            "control_jerk": 0.0,
            "survival_steps": 0,
            "jammed_steps": 0,
            "final_distance": 0.0
        }

    distances = np.array([pt["distance"] for pt in episode_trajectory], dtype=np.float32)
    j_flags = np.array([pt["J_flag"] for pt in episode_trajectory], dtype=np.int32)
    actions = np.array([pt["action"] for pt in episode_trajectory if "action" in pt], dtype=np.float32)

    # 1. Root Mean Square Tracking Error (RMSE)
    rmse = float(np.sqrt(np.mean(distances ** 2)))

    # 2. Control Jerk / Smoothness: mean squared difference between consecutive actions
    if len(actions) > 1:
        jerk = float(np.mean(np.sum((actions[1:] - actions[:-1]) ** 2, axis=1)))
    else:
        jerk = 0.0

    # 3. Survival & Jamming stats
    total_steps = len(episode_trajectory)
    jammed_indices = np.where(j_flags == 1)[0]
    jammed_steps = len(jammed_indices)

    # Post-jamming survival: steps survived after first jamming occurrence
    if jammed_steps > 0:
        first_jam = jammed_indices[0]
        post_jam_survival = total_steps - first_jam
    else:
        post_jam_survival = total_steps

    return {
        "success": 1 if hit else 0,
        "crash": 1 if crash else 0,
        "rmse_tracking_error": rmse,
        "control_jerk": jerk,
        "survival_steps": total_steps,
        "jammed_steps": jammed_steps,
        "post_jam_survival": post_jam_survival,
        "final_distance": float(distances[-1])
    }


def aggregate_experiment_results(all_episodes_metrics: list) -> dict:
    """Aggregates metrics across multiple evaluation runs."""
    n = len(all_episodes_metrics)
    if n == 0:
        return {}

    return {
        "success_rate": float(np.mean([m["success"] for m in all_episodes_metrics]) * 100.0),
        "crash_rate": float(np.mean([m["crash"] for m in all_episodes_metrics]) * 100.0),
        "mean_rmse": float(np.mean([m["rmse_tracking_error"] for m in all_episodes_metrics])),
        "std_rmse": float(np.std([m["rmse_tracking_error"] for m in all_episodes_metrics])),
        "mean_jerk": float(np.mean([m["control_jerk"] for m in all_episodes_metrics])),
        "mean_survival": float(np.mean([m["survival_steps"] for m in all_episodes_metrics])),
        "mean_final_distance": float(np.mean([m["final_distance"] for m in all_episodes_metrics]))
    }

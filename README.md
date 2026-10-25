# 🚁 Autonomous UAV Interception Under Adversarial Sensor Spoofing

### Resilient Edge-Autonomy via Recurrent PPO (LSTM-PPO) and POMDP

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v1.0+-008000.svg)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-Contrib-blueviolet.svg)](https://sb3-contrib.readthedocs.io/)

---

## 📌 Overview

This repository provides a complete, PhD-level reinforcement learning framework for autonomous drone target interception under active electronic warfare (EW), GPS spoofing, and sensor blackouts.

Traditional autonomous UAV target trackers rely on continuous telemetry or assume clean GPS/vision feeds. When an adversary introduces active sensor spoofing, memoryless agents (standard PID controllers or feedforward Deep RL) violently diverge and crash.

**Our Core Contribution:**
* **Adversarial POMDP Environment:** Formulates a Partially Observable Markov Decision Process where the target relative position vector is subjected to 4 distinct attack modes: **Gaussian Noise**, **Coordinate Spike Spoofing**, **Blackout**, and **Cumulative Drift**.
* **Emergent Dead-Reckoning via LSTM:** Deploys a Recurrent PPO agent. While the external target tracker is corrupted, onboard IMU sensors remain unjammable. The LSTM hidden states maintain temporal kinematic momentum, enabling the drone to filter out false coordinates and smoothly intercept moving targets.

---

## 📁 Repository Structure

```
reforcement learning/
├── environment/
│   ├── __init__.py               # Registers "DroneAntiJammingEnv-v0" & "CurriculumDroneEnv-v0"
│   ├── drone_env.py              # Main Gym POMDP Environment (step, reset, reward)
│   ├── drone_dynamics.py         # 6-DOF Quadrotor physics & rate controller
│   ├── jamming_model.py          # 4-Mode Adversarial Electronic Warfare Engine
│   ├── target_manager.py         # Target trajectories (Stationary, Linear, Sinusoidal)
│   └── curriculum_env.py         # 4-Stage Progressive Curriculum Environment
├── evaluation/
│   ├── metrics.py                # Statistical performance metrics (RMSE, Jerk, Survival)
│   ├── evaluate.py               # Benchmark test harness (Clean vs Jammed)
│   └── visualize.py              # Publication-grade figures & trajectory visualizer
├── results/                      # Consolidated research deliverables & empirical data
│   ├── baseline_ppo.zip          # Pre-trained Baseline FFN-PPO model checkpoint
│   ├── drone_lstm_ppo.zip        # Pre-trained Proposed Recurrent LSTM-PPO model checkpoint
│   ├── eval_results.json         # Real empirical benchmark results across all scenarios
│   ├── sample_trajectories.pkl   # Full 50 Hz flight rollout coordinate telemetry
│   ├── figures/                  # 5 High-resolution publication figures (PNG)
│   ├── logs/                     # TensorBoard training progression telemetry
│   └── training_scripts/         # Offline training pipelines & hyperparameter configs
├── complete_pipeline.ipynb       # Publication-grade interactive Jupyter Notebook
├── requirements.txt              # Core Python dependencies
└── README.md                     # Academic documentation & quick-start guide
```

---

## ⚡ Quick Start

### 1. Verify Environment Physics & Jamming Modes
```bash
python test_env.py
```

### 2. Train the Baseline FFN-PPO (Control Group)
```bash
python training/train_baseline.py --steps 50000 --envs 4
```

### 3. Train the Proposed Recurrent LSTM-PPO (Memory-Augmented)
```bash
python training/train_lstm.py --steps 50000 --envs 4
```

### 4. Run Comparative Evaluation Benchmark
```bash
python evaluation/evaluate.py
```

### 5. Generate Publication Figures
```bash
python evaluation/visualize.py
```

---

## 📓 Interactive Jupyter Notebook

For an interactive, step-by-step walkthrough with LaTeX mathematical formulas, real-time code execution, and inline 3D visualization:

Open `complete_pipeline.ipynb` in VS Code or JupyterLab:
```bash
jupyter lab complete_pipeline.ipynb
```

---

## 📊 Benchmark Results

| Scenario | Agent | Success Rate | Tracking RMSE | Control Jerk | Survival Steps |
|---|:---:|:---:|:---:|:---:|:---:|
| Clean Sensors | Baseline FFN-PPO | 86.7% | 1.15 m | 0.045 | 380 |
| **Active Jamming** | **Baseline FFN-PPO** | **13.3%** | **6.82 m** | **0.182** | **142 (Crash)** |
| Clean Sensors | Proposed LSTM-PPO | 93.3% | 0.82 m | 0.038 | 420 |
| **Active Jamming** | **Proposed LSTM-PPO** | **76.7%** | **1.94 m** | **0.065** | **365 (Intercept)** |

> **Key Discovery:** The **+63.4%** success gap under jamming proves that the recurrent policy's latent memory autonomously bridges sensor corruption using onboard inertial momentum.

# AUV Trajectory Tracking: DLNNSMC Replication

This repository contains the complete, from-scratch implementation and replication of the control methodologies presented in the research paper: *"Double-Loop PID-Type Neural Network Sliding Mode Control of an Uncertain Autonomous Underwater Vehicle Model"* (Mathematics 2022, 10, 3332).

## Project Overview

The objective of this project is to model a 4-DOF spatial Autonomous Underwater Vehicle (AUV) and evaluate advanced control strategies for complex 3D trajectory tracking (Helix and Sinusoidal paths). The simulation rigorously tests the controllers against realistic physical constraints.

### Evaluated Controllers:
1. **RBFPDSMC:** A baseline single-loop Proportional-Derivative Sliding Mode Controller using an RBF neural network.
2. **RBFPIDSMC:** A baseline single-loop Proportional-Integral-Derivative Sliding Mode Controller using an RBF neural network.
3. **DLNNSMC (Proposed):** A Double-Loop Neural Network Sliding Mode Controller that decouples the kinematics and dynamics for superior tracking stability.

### Simulation Realism
Unlike idealized mathematical simulations, this codebase enforces strict physical realism to prove robustness:
* **15% Model Mismatch:** The controllers operate with an incorrect mass matrix, forcing the neural networks to adapt to the unknown dynamics.
* **Sensor Noise:** Gaussian noise is continuously injected into the position and velocity feedback to mimic real-world IMU and DVL sensor inaccuracies.
* **Ocean Currents:** Low-frequency harmonic wave drift and stochastic disturbances are applied to all degrees of freedom.
* **Actuator Saturation:** Thruster outputs are physically clamped to realistic bounds.

## Repository Structure

* `auv_environment.py`: Contains the 4-DOF non-linear rigid body dynamics and hydrodynamic matrix formulations.
* `controllers.py`: Contains the algorithms for RBFPDSMC, RBFPIDSMC, and the Double-Loop DLNNSMC.
* `run_experiments.py`: The main simulation engine. Generates trajectories, applies noise/disturbances, and renders the plots and animated GIFs.
* `report.tex`: The comprehensive IEEE-format project report.
* `presentation.tex`: The Beamer presentation slide deck summarizing the findings.
* `results/`: Directory containing all generated 3D tracking plots, position error subplots, and high-definition animated trajectory GIFs.

## Execution

To run the simulation and generate the comparative plots and GIFs, ensure you have the required dependencies (numpy, scipy, matplotlib, Pillow) and execute:

```bash
python run_experiments.py
```

The outputs will be systematically saved to the `results/` folder, ready for inclusion in reports and presentations.

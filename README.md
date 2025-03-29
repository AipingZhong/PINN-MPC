# A Physics-Informed Neural Network-Based Model Predictive Control Framework for SIR Models
## Abstract
This paper introduces a Physics-Informed Neural Network (PINN)-based Model Predictive Control (MPC) framework for the Susceptible-Infected-Removed ($SIR$) spreading model. Existing studies in epidemic control problems often assume either measurable states, in which case parameters are learned, or known parameters, in which case states are learned, to design the MPC framework. In this work, we address the joint real-time estimation of states and parameters within the MPC framework using only noisy infected state data. We propose two novel PINN algorithms that are integrated into the MPC framework. First, we introduce Log-Scaled PINNs, which incorporate a log-scaled loss function to improve robustness against data noise. Next, we present Split-Integral PINNs, which leverage integral operators and state coupling in the neural network training process to effectively reconstruct complete epidemic state information. By incorporating these algorithms into the MPC framework, we simultaneously estimate model parameters and epidemic states while generating optimal control strategies interactively. Simulation results  demonstrate the effectiveness of the proposed methods in different settings.

## Structure
- `SI-PINNs.py`: Implements the Split-Integral PINNs model.
- `LS-PINNs.py`: Implements the Log-Scaled PINNs model.
- `PINNs.py`: Implements the PINNs model.
- `Plot/`: Contains all visualization scripts and figures for experimental results.
- `requirements.txt`: Key Python dependencies for running the experiments.

## Experiments

### Experiment 1
- **Models Used**: `SI-PINNs`, `LS-PINNs`
- **Purpose**: Evaluate and compare performance between SI-PINNs and LS-PINNs under the same conditions using the SIR model.

### Experiment 2
- **Models Used**: `PINNs`, `SI-PINNs`, `LS-PINNs`
- **Purpose**: Conduct comprehensive comparison among standard PINNs and its two variants to understand their strengths and weaknesses in SIR modeling tasks.

## Visualization
All result plots and figure-generating scripts are located in the `Plot/` directory.

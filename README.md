# Spatio-Temporal Extreme Modeling Benchmark

## Overview

This repository investigates methods for modeling **out-of-sample spatio-temporal extremes**. The primary objective is to evaluate how well different statistical and machine learning approaches generalize beyond the support of observed data, particularly under distributional shift and tail extrapolation.

The project:

1. Reviews and formalizes established and emerging methods for extreme event forecasting.
2. Evaluates these methods on multiple simulated and real-world datasets.
3. Compares performance across distinct extreme regimes.
4. Provides a modular framework for testing new approaches.

The emphasis is on **magnitude extrapolation**, **tail behavior**, **spatio-temporal amplification**, and **robustness under regime shift**.

---

## Research Questions

- Which modeling approaches best extrapolate to unseen extremes?
- How do methods perform under different mechanisms of extreme formation?
- What is the trade-off between predictive performance, stability, and computational cost?
- How many spatial locations or time series are required to reliably detect improvements?

---

## Methods

### Established Methods

The repository includes implementations and evaluations of:

#### 1. KL Divergence Loss for Distribution Modeling  
Qi and Majda, 2019  
Models predictive distributions by minimizing KL divergence to capture tail behavior.

#### 2. Gradient Boosting for Extreme Quantile Regression  
Velthoen et al., 2022  
Boosting-based estimation targeting extreme conditional quantiles.

#### 3. Extreme Value Loss (EVL)  
Ding et al., 2019  
Loss function designed to prioritize tail performance during training.

#### 4. MODWT  
Maximal Overlap Discrete Wavelet Transform for multi-scale decomposition of extremes.

#### 5. Extremal Random Forests  
Tree-based models adapted for heavy-tailed regression and extreme quantiles.

#### 6. Probability-Enhanced ANN (NEC+)  
Neural networks incorporating probabilistic structure to improve extreme sensitivity.

#### 7. Extreme Quantile Regression Networks  
Neural networks trained specifically for high-quantile prediction.

#### 8. Resampling Strategies  
Tail-aware resampling approaches to improve rare-event learning.

---

### Proposed Methods

#### 1. Clustering-Based Extreme Feature Extraction  
Extracts event and transition structure via clustering in spatio-temporal feature space.  
Designed to identify amplification regimes and regime transitions.

#### 2. DeepRV for POTS GPD Estimation  
Deep learning framework for Peaks Over Threshold modeling using Generalized Pareto distributions.  
Targets explicit tail parameter estimation in high-dimensional settings.

---

## Datasets

The benchmark spans both controlled simulations and real-world systems. Each dataset isolates a different mechanism of extreme formation.

---

### Simulated Datasets

#### Lorenz-96
Discrete spatial lattice with nonlinear local coupling and chaotic dynamics.

- Tests: Chaotic amplification
- Control parameter: Forcing amplitude increase

#### Network Shock Propagation System
Graph-based nonlinear transport system with injected shocks.

- Tests: Network amplification
- Control parameter: Subcritical to near-critical regime shift

#### Nonlinear Driver Field
Spatial field driven by nonlinear, lagged exogenous forcing with threshold effects.

- Tests: Functional extrapolation
- Control parameter: Driver support expansion

#### Heavy-Tailed Spatial AR Process
Multivariate autoregressive spatial field with heavy-tailed innovations and explicit spatial tail dependence.

- Tests: Tail extrapolation
- Control parameter: Tail index reduction

---

### Real-World Datasets

#### WeatherBench2 (240 × 121)
Global ERA5 reanalysis data at 6-hourly, 25 km resolution.

Targets:
- 2m temperature extremes
- Total precipitation extremes
- Wind speed extremes

Tests:
- Magnitude extrapolation in high-dimensional chaotic systems
- Multivariate nonlinear driver interaction
- Stability under stronger atmospheric instability

---

#### DeepExtremeCubes
Extreme-event-centered atmospheric cubes derived from ERA5.

Targets:
- Extreme temperature anomalies
- Extreme precipitation anomalies

Tests:
- Tail-focused learning
- Event-centric spatio-temporal amplification
- Extreme detection versus magnitude prediction

---

#### USGS Discharge + ERA5 Forcing
River discharge with meteorological drivers over directed river networks.

Targets:
- Flood peak discharge

Tests:
- Driver-based extreme formation
- Network transport and downstream amplification
- Heavy-tailed hydrological response

---

#### GEFCom Load + Weather
Regional electricity demand with weather covariates and calendar effects.

Targets:
- Peak load demand

Tests:
- Nonlinear response to exogenous drivers
- Extrapolation under rare temperature conditions
- Functional magnitude scaling

---

## Evaluation Framework

### Metrics

- Extreme quantile error
- Tail index estimation error
- CRPS and distributional scores
- Exceedance detection metrics
- Spatial consistency metrics
- Calibration under regime shift

---

### Optional High-Value Analyses

For each method, the framework supports:

#### Power Analysis
Estimate the number of time series or spatial locations required to detect a target improvement in extreme performance.

#### Runtime and Cost Analysis
Quantify computational cost versus predictive benefit.

#### Case Studies
Identify conditions under which specific methods are preferable.

#### Public Benchmarks
Leaderboards and reproducible notebooks to encourage adoption and comparison.

---

## Repository Structure

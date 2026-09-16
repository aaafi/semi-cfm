# Semi-CFM: Semi-Analytical Closed-Form Model for QoT Estimation

A comprehensive semi-analytical and machine learning framework for Quality of Transmission (QoT) estimation in ultra-wideband optical communication systems.

Semi-CFM provides rapid, accurate evaluations of Generalized Signal-to-Noise Ratio (GSNR) by modeling Inter-Channel Stimulated Raman Scattering (ISRS), Amplified Spontaneous Emission (ASE), and Non-Linear Interference (NLI) across C, L, S, and E bands.

## Key Features

- **Multi-Band Spectrum Grid:** Dynamic channel frequency allocation and guard-band implementation across L, C, S, and E bands.
- **Dynamic Attenuation Modeling:** Continuous, frequency-dependent fiber attenuation ($\alpha$) generation via `build_alpha_for_band`.
- **Semi-Analytical Solvers:** Fast numerical evaluation of forward/backward pump power (FLP/FRP) and semi-analytical NLI integration.
- **Topology & Routing:** Automated extraction of $k$-shortest paths and connection profiles from arbitrary mesh network cost matrices.
- **Hybrid ML Integration:** Designed to interface seamlessly with machine learning architectures and SHAP-based explainability pipelines for enhanced GSNR prediction.

## Installation

Clone the repository and install the dependencies:

```bash
git clone [https://github.com/aaafi/semi-cfm.git](https://github.com/aaafi/semi-cfm.git)
cd semi-cfm
pip install -r requirements.txt
```

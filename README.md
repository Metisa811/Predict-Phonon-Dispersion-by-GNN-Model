# Predicting Phonon Dispersion of MAX Phases via Physics-Informed Graph Neural Networks

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)]()

> **Goal**: Predict full phonon band structures of MAX phase materials using E(3)-Equivariant Graph Neural Networks with physics-informed constraints.  
> **Target publication**: *npj Computational Materials* or *Physical Review B*

---

## Overview

MAX phases (M₂AX) are nano-layered ceramics with unique combinations of metallic and ceramic properties, making them promising for nuclear, aerospace, and electronic applications. Computing their phonon band structures via Density Functional Perturbation Theory (DFPT) is computationally expensive (days–weeks per material).

This project develops a machine learning model that predicts full phonon dispersion curves from crystal structure alone — in milliseconds.

```
Crystal Structure (POSCAR)
  + Force Constants (IFCs)        →   E(3)-Equivariant GNN   →   Phonon Bands ω(q)
  + Atomic Features (100-dim)
  + Elastic Constants (C_ij)
```

---

## Key Results

| Model | MAE (THz) | Notes |
|-------|-----------|-------|
| Residual Dense Network | — | R² = 0.9827 (best early result) |
| Geometric Equivariant GNN | 0.48 | PaiNN-inspired, multi-task |
| Physics-Informed GNN (current) | 0.479 | RBF encoding, 5-layer MPNN |
| **Target** | **< 0.10** | Physics-constrained IFC prediction |

### Feature Importance (from XGBoost Ablation)
Top features driving phonon prediction accuracy:
1. `en_allen` — Allen electronegativity (ΔMAE: +19.66)
2. `covalent_radius_bragg` — Bragg covalent radius (ΔMAE: +17.01)
3. `magp_valence_s/d` — s/d orbital valence electrons (ΔMAE: +15–18)

Harmful features to exclude: `heat_of_formation`, `covalent_radius_cordero/slater`

---

## Dataset

- **358 MAX phase materials** (M₂AX formula, hexagonal P6₃/mmc symmetry)
- **8 atoms per unit cell** (consistent across all materials)
- Sources: POSCAR + Force Constants (Phonopy format) + Phonon bands (YAML)
- Atomic features: 100-dimensional per atom (from periodic table properties)
- Elastic constants: C₁₁, C₁₂, C₁₃, C₃₃, C₄₄, C₆₆, B_Hill, G_Hill, Debye temperature

Split: 286 train / 35 validation / 37 test

---

## Model Architecture

```
Node Features (100-dim)
       ↓
Initial Embedding: Dense(256) + LayerNorm + SiLU
       ↓
RBF Message Passing × 5
  - Edge encoding: 64 Gaussian RBF kernels (0.5–6.0 Å)
  - Direction-aware: dot-product attention
  - Residual connections at every layer
       ↓
Attention Pooling → Graph Embedding (256-dim)
       ↓
Phonon Head: Dense(512→512→256→OUTPUT)
```

### Physics-Informed Loss
$$L = \alpha L_{\text{IFC}} + \beta L_{\text{sym}} + \gamma L_{\text{ASR}} + \delta L_{\text{elastic}} + \epsilon L_{\text{smooth}}$$

| Term | Formula | Purpose |
|------|---------|---------|
| $L_{\text{sym}}$ | $\|IFC_{ij} - IFC_{ji}^T\|^2$ | Force constant symmetry |
| $L_{\text{ASR}}$ | $\|\sum_j IFC_{ij}\|^2$ | Acoustic sum rule |
| $L_{\text{smooth}}$ | $\|(y_{i+1}-2y_i+y_{i-1})\|^2$ | Band smoothness |

---

## Repository Structure

```
├── paper/                    # LaTeX manuscript
│   ├── main.tex
│   ├── sections/
│   └── figures/
├── src/
│   ├── data/
│   │   ├── parse_poscar.py
│   │   ├── parse_force_constants.py
│   │   └── build_dataset.py
│   ├── models/
│   │   ├── gnn.py            # Main GNN architecture
│   │   ├── ifc_head.py       # Force constants prediction head
│   │   └── phonon_calc.py    # IFC → phonon bands calculator
│   ├── training/
│   │   ├── train.py
│   │   └── losses.py         # Physics-informed losses
│   └── evaluation/
│       └── evaluate.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_xgboost_ablation.ipynb
│   ├── 03_gnn_training.ipynb
│   └── 04_results_analysis.ipynb
├── results/
│   └── experiments/
├── requirements.txt
└── README.md
```

---

## Physics Background

### Dynamical Matrix
$$D_{\alpha\beta}(\mathbf{q}) = \frac{1}{\sqrt{M_i M_j}} \sum_j \Phi_{\alpha\beta}(0i,lj)\, e^{i\mathbf{q}\cdot\mathbf{R}_l}$$

### Phonon Frequencies
$$D(\mathbf{q})\,\mathbf{e} = \omega^2(\mathbf{q})\,\mathbf{e}$$

### Acoustic Sum Rule
$$\sum_j \Phi_{\alpha\beta}(i,j) = 0 \quad \forall\, i,\alpha,\beta$$

---

## Installation

```bash
git clone https://github.com/Metisa811/Predict-Phonon-Dispersion-by-GNN-Model.git
cd Predict-Phonon-Dispersion-by-GNN-Model
pip install -r requirements.txt
```

---

## Citation

> Sadeghi, M. M. (2025). Predicting Phonon Dispersion of MAX Phases via
> Physics-Informed Graph Neural Networks. *In preparation*.

---

## Supervisor
- Dr. Khazaei

## Author
- Mohammad Mehdi Sadeghi

---
name: tool-installation
description: How to install design tools and the BM2 evaluator
keywords: [install, setup, conda, pip, GPU, CUDA, environment, download, requirements]
---
# Installation Guide

## Prerequisites
- NVIDIA GPU >= 16 GB VRAM (A100 40/80GB recommended)
- CUDA 11.8+ or 12.x
- conda or mamba
- ~100 GB disk for all tools + weights

## BM2 Evaluator (Standalone)
```bash
pip install bm2-evaluator
bm2-eval score --designs ./designs/ --target target.pdb --chain A
```

## Design Tools

### BindCraft
```bash
git clone https://github.com/martinpacesa/BindCraft
cd BindCraft && bash install_bindcraft.sh
# Creates conda env, needs AF2 weights + PyRosetta license
```

### BoltzGen
```bash
conda create -n boltzgen python=3.12
conda activate boltzgen && pip install boltzgen
# Downloads ~6 GB weights on first run
```

### RFdiffusion + LigandMPNN
```bash
git clone https://github.com/RosettaCommons/RFdiffusion
git clone https://github.com/dauparas/LigandMPNN
# Follow their install instructions
```

### Mosaic
```bash
pip install escalante-mosaic
```

## Refolding Engines

### Boltz2: available through Mosaic install or `pip install boltz`
### AF2: `pip install colabdesign` in separate env

## Verification
```bash
bm2 tools check
```

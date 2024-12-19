# 🐸 Grok-Grok 🐸

This repository contains the implementation for our paper "Grokking After Grokking: Case study of Task Transfer in Modular Arithmetics" investigating how neural networks transfer learned computational patterns between modular arithmetic tasks.

[[Paper Link]](https://drive.google.com/file/d/1_DSXhYo8Mf-VbKW3d0_EiWaYE07BWT_X/view?usp=sharing)

## Overview

We study task transfer across 9 modular arithmetic tasks:
- Task1: (x + y) mod p
- Task2: (x - y) mod p
- Task3: ((x + y)²) mod p
- Task4: (x² + y²) mod p
- Task5: (x · y⁻¹) mod p
- Task6: (2xy) mod p
- Task7: (x³ + y³) mod p
- Task8: ((x + y)³) mod p
- Task9: (xy) mod p

## Training

Use `train_grok.py` to train models on individual tasks or perform transfer learning experiments.

### Arguments
- `--fn_name`: Function to train (Task1-Task9)
- `--project_name`: Name for wandb tracking
- `--ckpt`: Path to resume from checkpoint (optional)

## Analysis & Visualization

The `visualize.ipynb` notebook provides tools for:
- Analyzing Fourier components
- Visualizing activation patterns, etc.

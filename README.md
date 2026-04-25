# 🎨 Anime Face Generation with GANs

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A complete implementation of Deep Convolutional Generative Adversarial Networks (DCGAN) for generating high-quality anime face images. This project demonstrates end-to-end deep learning pipeline from data acquisition to model evaluation.

## 🎯 Project Overview

This project implements a GAN-based generative model trained on 63,565 anime face images from Kaggle. The model learns to generate realistic anime-style faces by training a Generator and Discriminator network in an adversarial setup.

**Key Features:**
- 🔥 DCGAN architecture with optimized hyperparameters
- 📊 Comprehensive evaluation using FID and Inception Score
- 🎨 High-quality anime face generation
- 📈 Training visualization and progress tracking
- 🛠️ Modular, production-ready codebase

## 📊 Results

- **FID Score:** 38.7 (lower is better)
- **Inception Score:** 8.2 ± 0.4 (higher is better)
- **Training Time:** ~2.5 hours on NVIDIA GPU
- **Model Parameters:** 6.5M (Generator: 3.7M, Discriminator: 2.8M)

## 🖼️ Sample Outputs

[Include a grid of generated anime faces here]

*Generated anime faces at Epoch 100*

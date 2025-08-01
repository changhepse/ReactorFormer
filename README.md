# ReactorFormer for multiphysics modeling of chemical reactors

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the implementation of **ReactorFormer**, an open framework developed for multiphysics modeling of chemical reactors.
## 🔧 System Requirements

- Operating System: Linux / macOS / Windows 10+
- Python ≥ 3.8
- Recommended: Anaconda or virtualenv
- Dependencies (see `requirements.txt`), including:
  - numpy
  - scipy
  - matplotlib
  - torch
  - etc.

## 💻 Installation Guide

### Option 1: Install with pip
```bash
git clone https://github.com/changhepse/ReactorFormer.git
cd ReactorFormer
pip install -r requirements.txt
```

### Option 2: Create a Conda environment
```bash
conda create -n reactorformer-env python=3.8
conda activate reactorformer-env
pip install -r requirements.txt
```

## 📘 Demo
The model generates the nRMSE errors between true and predicted results.
The training time of the models are around 0.5-2 hours per epoch.

## 📘 Instructions for Use

### Generate datasets:
Generate datasets using the files in ./datasets

### Load datasets:
Loading datasets using data_loaders.py

### Run Main Program:
```bash
python main.py --config configs/base.yaml
```

Modify `configs/base.yaml` to change model architecture, number of training epochs, learning rate, and other settings.


## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 🌐 Open Source Repository

- GitHub: [https://github.com/changhepse/ReactorFormer](https://github.com/changhepse/ReactorFormer)

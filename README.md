# MTCL-UAV: Multi-scale Transformers with Contrastive Learning for UAV Anomaly Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10.1+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 Overview

MTCL-UAV is a novel framework for UAV anomaly detection using flight data. It combines multi-scale transformers with contrastive learning to effectively detect anomalies in UAV flight patterns. The framework leverages the power of transformer architectures with multi-scale patch processing and contrastive learning techniques to achieve superior anomaly detection performance.

## 🏗️ Overview

![Framework](figs/overview_v4_00.png)

The framework consists of:

- **Multi-scale Transformer Layers**: Processes input sequences at different scales using patch-based attention mechanisms
- **Mixture of Experts (MoE)**: Dynamically routes information through expert networks
- **Contrastive Learning**: Enhances feature learning through contrastive objectives

## 📋 Requirements

- Python 3.8+
- PyTorch 1.10.1+
- CUDA 11.1+ (for GPU acceleration)

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone https://github.com/SteelHu/MTCL-UAV.git
cd MTCL-UAV
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

## 📊 Datasets

### ALFA Dataset
The AirLab Failure and Anomaly (ALFA) dataset is used for UAV anomaly detection experiments.

**Download**: [ALFA Dataset](https://theairlab.org/alfa-dataset/)

**Dataset Structure**:
```
dataset/
├── ALFA/
│   ├── train.csv
│   └── test.csv
```

The dataset can be preprocessed as needed and in the end all you need are two csv files.

### FW-UAV Dataset
Flight Data dataset for additional experiments.

**Download**: [[FW-UAV](https://github.com/mrtbrnz/fault_detection)]

**Dataset Structure**:
```
dataset/
├── FD/
│   ├── train_X.csv
│   ├── train_y.csv
│   ├── test_X.csv
│   └── test_y.csv
```

## 🎯 Usage

### Basic Training

```bash
python run.py \
    --is_training 1 \
    --model MTCL \
    --data ALFA_ad \
    --root_path ./dataset/ALFA/ \
    --seq_len 96 \
    --pred_len 96 \
    --d_model 16 \
    --d_ff 16 \
    --num_nodes 18 \
    --layer_nums 3 \
    --batch_size 128 \
    --train_epochs 100 \
    --learning_rate 0.001 \
    --anomaly_ratio 3
```

### Testing Only

```bash
python run.py \
    --is_training 0 \
    --model MTCL \
    --data ALFA_ad \
    --root_path ./dataset/ALFA/ \
    --seq_len 96 \
    --pred_len 96
```

### Key Parameters

| Parameter         | Description                  | Default |
| ----------------- | ---------------------------- | ------- |
| `--model`         | Model name (MTCL)            | MTCL    |
| `--data`          | Dataset type (ALFA_ad, FD)   | FD      |
| `--seq_len`       | Input sequence length        | 96      |
| `--pred_len`      | Prediction sequence length   | 96      |
| `--d_model`       | Model dimension              | 16      |
| `--d_ff`          | Feed-forward dimension       | 16      |
| `--num_nodes`     | Number of features           | 8       |
| `--layer_nums`    | Number of transformer layers | 3       |
| `--anomaly_ratio` | Prior anomaly ratio (%)      | 3       |
| `--batch_size`    | Training batch size          | 128     |
| `--train_epochs`  | Number of training epochs    | 1       |
| `--learning_rate` | Learning rate                | 1E-4    |

## 📁 Project Structure

```
MTCL-UAV/
├── data_provider/          # Data loading and preprocessing
│   ├── data_factory.py     # Data provider factory
│   └── data_loader.py      # Dataset classes
├── exp/                    # Experiment configurations
│   ├── exp_anomaly_detection.py  # Anomaly detection experiments
│   └── exp_basic.py        # Basic experiment class
├── layers/                 # Neural network layers
│   ├── Embedding.py        # Embedding layers
│   ├── Layer.py           # Base layer definitions
│   ├── MoE.py             # Mixture of Experts
│   └── RevIN.py           # Reversible Instance Normalization
├── models/                 # Model definitions
│   └── MTCL.py            # MTCL model implementation
├── utils/                  # Utility functions
│   ├── decomposition.py    # Signal decomposition
│   ├── masking.py         # Masking utilities
│   ├── metrics.py         # Evaluation metrics
│   ├── timefeatures.py    # Time feature encoding
│   └── tools.py           # General utilities
├── scripts/                # Experiment scripts
│   ├── manuscript/         # Manuscript experiments
│   └── revision/          # Revision experiments
├── dataset/               # Dataset directory
├── checkpoints/           # Model checkpoints
├── logs/                  # Training logs
├── run.py                 # Main training script
└── requirements.txt       # Python dependencies
```

### Custom Experiments

You can create custom experiments by modifying the parameters in `run.py` or creating new shell scripts based on the existing ones.

## 📚 Citation

If you use this code in your research, please cite our paper:

```bibtex
@ARTICLE{11006654,
  author={Hu, Gang and Zhou, Zhongliang and Li, Zhengxin and Dong, Zheng and Fang, Jiayong and Zhao, Yu and Zhou, Chuhan},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={Multiscale Transformers With Contrastive Learning for UAV Anomaly Detection}, 
  year={2025},
  volume={74},
  number={},
  pages={1-15},
  doi={10.1109/TIM.2025.3571126}}
```

## 🙏 Acknowledgments

1. Keipour, M. Mousaei, and S. Scherer, “Automatic Real-time Anomaly Detection for Autonomous Aerial Vehicles,” in 2019 IEEE International Conference on Robotics and Automation (ICRA), May 2019, pp.5679-5685. doi: 10.1109/ICRA.2019.8794286.
2. M. Bronz, E. Baskaya, D. Delahaye and S. Puechmore, "Real-time Fault Detection on Small Fixed-Wing UAVs using Machine Learning," 2020 AIAA/IEEE 39th Digital Avionics Systems Conference (DASC), San Antonio, TX, USA, 2020, pp. 1-10, doi: 10.1109/DASC50938.2020.9256800.
3. H. Wu, T. Hu, Y. Liu, H. Zhou, J. Wang, and M. Long, "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis", in International Conference on Learning Representations, 2023.

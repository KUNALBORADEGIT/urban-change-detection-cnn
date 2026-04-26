# Urban Change Detection Using CNN Models

> **Assignment 2 — Group Research Project**  
> University of Leicester | Group 1  
> *Application of CNN Models to Remote Sensing Using Satellite Imagery*

---

## Overview

This project implements a complete deep learning pipeline for **binary urban change detection** using multi-temporal satellite imagery. Two CNN architectures are implemented and compared on the [LEVIR-CD](https://chenhao.in/LEVIR/) benchmark dataset:

| Model | Approach | Test IoU | Test F1 |
|---|---|---|---|
| **U-Net** | Early fusion (6-channel input) | 0.6138 | 0.7261 |
| **Siamese U-Net** | Shared encoder + feature difference | 0.5000 | 0.6576 |

Given a pair of co-registered satellite images captured at two different times (pre-change and post-change), the models produce a binary mask identifying pixels where urban change (building construction or demolition) has occurred.

---

## Project Structure

```
urban-change-detection/
│
├── data/
│   └── LEVIR CD/
│       ├── train/
│       │   ├── A/          # Pre-change images (time T1)
│       │   ├── B/          # Post-change images (time T2)
│       │   └── label/      # Binary ground truth masks
│       ├── val/
│       │   ├── A/
│       │   ├── B/
│       │   └── label/
│       └── test/
│           ├── A/
│           ├── B/
│           └── label/
│
├── src/
│   ├── config/
│   │   └── config.yaml         # All hyperparameters and paths
│   ├── models/
│   │   ├── unet.py             # U-Net (early fusion baseline)
│   │   └── siamese_unet.py     # Siamese U-Net (shared encoder)
│   └── utils/
│       ├── dataset.py          # LEVIR-CD data loader
│       ├── train.py            # Training loop, DiceBCE loss, evaluation
│       ├── test.py             # Test evaluation, CSV export
│       └── metrics.py          # IoU, F1, precision, recall, accuracy
│
├── runs/
│   ├── unet/                   # U-Net checkpoints and results
│   │   ├── best.pt
│   │   └── test_metrics.csv
│   └── siamese_unet/           # Siamese U-Net checkpoints and results
│       ├── best.pt
│       └── test_metrics.csv
│
├── main.py                     # Entry point — train or test
├── requirements.txt
└── README.md            
```

---

## Requirements

- Python 3.8+
- PyTorch 1.12+
- See `requirements.txt` for the full list

### Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU recommended.** Training was developed on CPU for portability. To enable GPU, set `use_cuda: true` in `src/config/config.yaml` before running.

---

## Dataset Setup

This project uses the **LEVIR-CD** dataset — a large-scale benchmark for building change detection in very high-resolution satellite imagery.

1. Download the dataset from: https://www.kaggle.com/datasets/mdrifaturrahman33/levir-cd
2. Extract and place it so the folder structure matches:

```
data/LEVIR CD/train/A/
data/LEVIR CD/train/B/
data/LEVIR CD/train/label/
data/LEVIR CD/val/...
data/LEVIR CD/test/...
```

The dataset loader (`dataset.py`) automatically handles common folder naming variants (`A/`, `t1/`, `image1/`, `ImagesA/`, etc.).

---

## Usage

All commands are run from the **project root directory**.

### Train U-Net

```bash
python main.py --mode train --model unet
```

### Train Siamese U-Net

```bash
python main.py --mode train --model siamese_unet
```

### Test U-Net

```bash
python main.py --mode test --model unet
```

### Test Siamese U-Net

```bash
python main.py --mode test --model siamese_unet
```

Checkpoints and test results are saved to `runs/<model_name>/`.

---

## Configuration

All hyperparameters are controlled via `src/config/config.yaml`:

```yaml
data:
  root_dir: "data/LEVIR CD"
  img_size: [256, 256]

model:
  name: "unet"          # "unet" or "siamese_unet"
  base_channels: 32

train:
  use_cuda: false       # Set true if GPU is available
  batch_size: 8
  num_workers: 0
  epochs: 50
  lr: 0.0003
  weight_decay: 0.01
  threshold: 0.45
  loss: "dice_bce"
  bce_weight: 0.5
  out_dir: "runs/default"
```

### CLI overrides

Any config value can be overridden directly from the command line without editing the YAML file:

```bash
python main.py --mode train --model unet --epochs 100 --lr 0.0001 --batch_size 16
```

| Flag | Description | Default |
|---|---|---|
| `--model` | `unet` or `siamese_unet` | `unet` |
| `--epochs` | Number of training epochs | 50 |
| `--batch_size` | Batch size | 8 |
| `--lr` | Initial learning rate | 0.0003 |
| `--threshold` | Binarisation threshold | 0.45 |
| `--out_dir` | Output directory for checkpoints | `runs/<model>` |

---

## Model Architectures

### U-Net (Early Fusion)

The standard U-Net encoder-decoder with skip connections, adapted for change detection by concatenating the pre-change and post-change RGB images along the channel axis to form a **6-channel input**. The network learns cross-temporal correlations from the first convolution layer.

- 4 encoder stages + bottleneck + 4 decoder stages
- Base channels: 32 (encoder: 32→64→128→256→512)
- ~7.76M parameters

### Siamese U-Net (Shared Encoder + Feature Difference)

Each temporal image is passed independently through a **shared-weight encoder**. Absolute feature differences `|f_A − f_B|` are computed at every encoder scale and used as skip connections for the decoder. This encodes change information explicitly in a learned feature space rather than raw pixel space.

- Same encoder/decoder structure as U-Net
- Weights shared between the two branches (encoder applied twice per forward pass)
- ~11.64M effective parameters

---

## Training Details

Both models were trained under identical conditions for a fair comparison:

- **Optimiser:** AdamW (weight decay = 0.01)
- **Loss:** Combined Dice-BCE (α = 0.5) — robust to class imbalance
- **Scheduler:** ReduceLROnPlateau (mode=max, factor=0.5, patience=3)
- **Checkpoint:** Best model saved by maximum validation IoU
- **Augmentation:** Random horizontal and vertical flips (p = 0.5 each)

---

## Results

Test set evaluation on LEVIR-CD (best checkpoint per model):

| Metric | U-Net | Siamese U-Net |
|---|---|---|
| Loss | 0.2528 | 0.3115 |
| **IoU** | **0.6138** | 0.5000 |
| **F1-Score** | **0.7261** | 0.6576 |
| Accuracy | 0.9809 | 0.9757 |
| Precision | 0.7991 | 0.7217 |
| Recall | 0.7374 | 0.6133 |

Test metrics are automatically saved to `runs/<model>/test_metrics.csv` after running `--mode test`.

---

## Reproducing Results

To fully reproduce both sets of results from scratch:

```bash
# 1. Train U-Net
python main.py --mode train --model unet

# 2. Test U-Net
python main.py --mode test --model unet

# 3. Train Siamese U-Net
python main.py --mode train --model siamese_unet

# 4. Test Siamese U-Net
python main.py --mode test --model siamese_unet
```

Results will be written to:
- `runs/unet/test_metrics.csv`
- `runs/siamese_unet/test_metrics.csv`

---

## References

- Ronneberger et al. (2015) — U-Net: Convolutional Networks for Biomedical Image Segmentation
- Daudt et al. (2018) — Fully Convolutional Siamese Networks for Change Detection
- Chen & Shi (2020) — LEVIR-CD Dataset and BIT Architecture
- Loshchilov & Hutter (2019) — AdamW: Decoupled Weight Decay Regularization

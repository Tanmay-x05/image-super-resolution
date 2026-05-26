# image-super-resolution
4× image super-resolution using three GAN variants (Basic GAN → SRGAN → ESRGAN) on 685 LR/HR image pairs. Each model upscales 64×64 → 256×256, evaluated via PSNR. Ends with a side-by-side loss/PSNR comparison across all three.


# Image Super-Resolution using GANs (SRGAN & ESRGAN)

A PyTorch implementation of 4× image super-resolution using three progressively advanced GAN architectures, trained on paired low-resolution / high-resolution image datasets.

## Overview

Upscales images from **64×64 → 256×256** using three models of increasing complexity:

| Model | Generator | Loss |
|---|---|---|
| Basic GAN | Simple ConvNet | MSE + Adversarial |
| SRGAN | ResNet blocks | Perceptual (VGG) + Adversarial |
| ESRGAN | RRDB blocks | Enhanced Perceptual + Relativistic Adversarial |

## Project Structure

├── notebook.ipynb          # Main notebook
├── dataset/
│   └── train/
│       ├── low_res/        # 64×64 input images
│       └── high_res/       # 256×256 target images

## Requirements

```bash
pip install torch torchvision kagglehub tqdm matplotlib pillow
```

## Dataset

Downloaded automatically via `kagglehub`:

```python
import kagglehub
path = kagglehub.dataset_download("adityachandrasekhar/image-super-resolution")
```

685 paired LR/HR image pairs used for training.

## Pipeline

1. **Dataset** — paired LR/HR loading with normalization
2. **Generator pretraining** — MSE-only warmup for stable GAN initialization
3. **GAN training** — full adversarial training with perceptual + adversarial losses
4. **Evaluation** — PSNR metric computed per epoch
5. **Comparison** — side-by-side PSNR, generator loss, and discriminator loss curves across all three models

## Results

Models are compared using **PSNR (dB)** — higher is better. Training curves for all three models are plotted together at the end of the notebook.

## Key Design Choices

- Generator is **pretrained with MSE** before adversarial training to avoid early instability
- ESRGAN uses **Residual-in-Residual Dense Blocks (RRDB)** for richer feature extraction
- ESRGAN uses a **relativistic discriminator** for more stable training signal
- Mixed precision (AMP) supported automatically when CUDA is available

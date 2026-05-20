# WFDViM

This is the official code repository for **"WFDViM: Wavelet Frequency Decoupling Vision Mamba for Weak-Boundary Medical Image Segmentation"**.

## Abstract

Convolution--Mamba hybrid architectures provide a new option for efficient long-range modeling in medical image segmentation. However, directly feeding full-frequency features into state space modules introduces redundant state propagation and fails to exploit Mamba's intrinsic preference for low-frequency global information modeling, making it difficult to stably recover lesion contours under weak-boundary and noise-interference scenarios. To address this issue, we propose WFDViM (Wavelet Frequency Decoupling Vision Mamba). In the encoding stage, the Wavelet Frequency Decoupling Block (WFDB) explicitly separates low-frequency semantics and high-frequency details through discrete wavelet transform, making low-frequency components more suitable for global modeling by Mamba, while high-frequency components are adaptively denoised by Learnable Anisotropic Diffusion (LAD), suppressing speckle noise, specular reflections, and texture noise while preserving true boundary structures. In the decoding stage, a lightweight Mask-Aware Decoding module (MAD) generates an initial regional mask, and the Wavelet-Guided Attention module (WGA) progressively injects the denoised high-frequency geometric priors into skip connections, thereby stably recovering weak-boundary contours. On four datasets, namely ISIC 2017, ISIC 2018, CVC-ClinicDB, and BUSI, WFDViM achieves DSC scores of 89.80%, 90.33%, 93.73%, and 85.20%, respectively, with only 13.06M parameters and 3.05 GFLOPs, demonstrating a favorable balance among segmentation accuracy, boundary recovery, and model efficiency.

## Visual Results

### Visual results on ISIC 2017

<p align="center">
  <img src="figures/isic2017_visualize.png" alt="Visual results on ISIC 2017" width="95%">
</p>

### Visual results on ISIC 2018

<p align="center">
  <img src="figures/isic2018_visualize.png" alt="Visual results on ISIC 2018" width="95%">
</p>

### Visual results on CVC-ClinicDB

<p align="center">
  <img src="figures/cvc_visualize.png" alt="Visual results on CVC-ClinicDB" width="95%">
</p>

### Visual results on BUSI

<p align="center">
  <img src="figures/busi_visualize.png" alt="Visual results on BUSI" width="95%">
</p>

## 0. Main Environments

Create and activate a conda environment:

```bash
conda create -n wfdvim python=3.10
conda activate wfdvim
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Install the custom CUDA kernels:

```bash
cd kernels/selective_scan
pip install .

cd ../dwconv2d
python3 setup.py install --user
```

## 1. Prepare the Dataset

The current release provides the training configuration for ISIC. CVC-ClinicDB and BUSI can be organized with the same image-mask folder format if users extend the configuration for these datasets.

### ISIC datasets

The ISIC 2017 and ISIC 2018 datasets can be downloaded from the [ISIC Challenge data page](https://challenge.isic-archive.com/data/). After downloading the dataset, organize the files as follows:

```text
/path/to/ISIC/
  train/
    images/
      xxx.png
    masks/
      xxx.png
  val/
    images/
      xxx.png
    masks/
      xxx.png
```

Then set the dataset path in `config/tiny_config_isic.py`:

```python
data_path = "/path/to/ISIC/"
```

### CVC-ClinicDB

CVC-ClinicDB is a colonoscopy polyp segmentation dataset released by the Polytechnic University of Catalonia. It contains 612 clinically acquired images with pixel-level annotations. The images are collected from real endoscopic procedures and present high structural complexity and visual noise. Due to the blurred boundaries, irregular shapes, and large-scale variations of polyps, this dataset is often used to evaluate segmentation under complex background conditions.

The dataset can be organized as:

```text
/path/to/CVC-ClinicDB/
  train/
    images/
    masks/
  val/
    images/
    masks/
```

### BUSI

The Breast Ultrasound Images (BUSI) dataset contains breast ultrasound images collected from women aged 25--75. It provides image-mask pairs across normal, benign, and malignant categories. Following common practice for lesion segmentation, benign and malignant samples with lesion masks can be used and split into training and validation sets.

The dataset can be organized as:

```text
/path/to/BUSI/
  train/
    images/
    masks/
  val/
    images/
    masks/
```

## 2. Train WFDViM

Run the training script from the repository root:

```bash
python tiny_train_isic.py
```

Training logs, checkpoints, and outputs are saved under:

```text
results/
```

The best model checkpoint is saved as:

```text
results/WFDViM_isic_xxxxx/checkpoints/best_model.pth
```

## 3. Main Files

```text
config/tiny_config_isic.py    ISIC training configuration
datasets/dataset.py           Dataset loading and basic transforms
models/wfdvim/                WFDViM model implementation
models/wavelet/               Wavelet frequency decoupling modules
kernels/                      Custom CUDA kernels
tiny_train_isic.py            Training entry
engine.py                     Training and validation loops
utils.py                      Losses, logging, and utilities
```

## 4. Citation

The paper is currently under submission. Citation information will be updated after publication.

## 5. Acknowledgments

We thank the authors of [VM-UNet](https://github.com/JCruan519/VM-UNet), [TinyViM](https://github.com/xwmaxwma/TinyViM), [Spatial-Mamba](https://github.com/EdwardChasel/Spatial-Mamba), [VMamba](https://github.com/MzeroMiko/VMamba), and [Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet) for their open-source codes.

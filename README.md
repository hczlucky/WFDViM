# WFDViM

This repository provides the official implementation of **WFDViM: Wavelet Frequency Decoupling Vision Mamba with Wavelet-Guided Attention for Medical Image Segmentation**.

WFDViM is a frequency-aware hybrid Vision Mamba network for weak-boundary medical image segmentation. It uses wavelet frequency decoupling to separate low-frequency semantics and high-frequency details, applies learnable anisotropic diffusion to suppress noise in high-frequency components, and introduces wavelet-guided attention to progressively inject geometric priors into the decoder.

This release includes the core model implementation, the custom Spatial-Mamba CUDA kernels, and an ISIC training configuration.

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

CVC-ClinicDB is a colonoscopy polyp segmentation dataset released by the Polytechnic University of Catalonia. It contains 612 clinically acquired images with pixel-level annotations and is commonly used to evaluate polyp segmentation under blurred boundaries, irregular shapes, specular highlights, and complex backgrounds.

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

The Breast Ultrasound Images dataset contains breast ultrasound images from normal, benign, and malignant categories. Following common practice for lesion segmentation, benign and malignant samples with lesion masks can be used and split into training and validation sets.

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

## 2. Model Configuration

The released configuration follows a TinyViM Base-style encoder setting and uses an asymmetric decoder for medical image segmentation.

| Item | Setting |
| --- | --- |
| Input size | 256 x 256 |
| Number of classes | 1 |
| Encoder embed dims | [48, 96, 192, 384] |
| Encoder depths | [4, 3, 10, 5] |
| Decoder dims | [384, 192, 96, 48] |
| Decoder depths | [5, 10, 3, 4] |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Batch size | 32 |
| Epochs | 300 |

The full configuration is available in:

```text
config/tiny_config_isic.py
```

## 3. Train WFDViM

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

## 4. Main Files

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

## 5. Citation

If you find this work useful, please consider citing our paper. The citation information will be updated after publication.

```bibtex
@article{wfdvim,
  title={WFDViM: Wavelet Frequency Decoupling Vision Mamba with Wavelet-Guided Attention for Medical Image Segmentation},
  author={},
  journal={},
  year={2026}
}
```

## 6. Acknowledgments

We thank the authors of [VM-UNet](https://github.com/JCruan519/VM-UNet), [TinyViM](https://github.com/xwmaxwma/TinyViM), [Spatial-Mamba](https://github.com/EdwardChasel/Spatial-Mamba), [VMamba](https://github.com/MzeroMiko/VMamba), and [Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet) for their open-source codes.

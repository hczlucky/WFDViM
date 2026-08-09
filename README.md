# WFDViM

This is the official PyTorch implementation of **"WFDViM: Wavelet Frequency-Decoupled Vision Mamba for Robust Weak-Boundary Medical Image Segmentation"**.

## Abstract

WFDViM separates low-frequency semantics from high-frequency structures for weak-boundary medical image segmentation. Its encoder combines wavelet frequency decoupling, structure-aware state-space modeling, and learnable anisotropic diffusion, while the decoder uses multi-scale aggregation and wavelet-guided attention for progressive boundary refinement. Across three independent runs on ISIC 2017, ISIC 2018, CVC-ClinicDB, and BUSI, WFDViM achieves mean Dice scores of 89.82%, 90.36%, 93.66%, and 85.10%, respectively, with 13.06M parameters and 3.05 GFLOPs.

## 0. Main Environments

Create and activate the environment, then install the dependencies:

```bash
conda create -n wfdvim python=3.10
conda activate wfdvim
pip install -r requirements.txt
```

Install the custom CUDA kernels from the repository root:

```bash
cd kernels/selective_scan
pip install .
cd ../dwconv2d
python setup.py install --user
cd ../..
```

## 1. Prepare the Datasets

### ISIC datasets

- [ISIC 2017 (Google Drive)](https://drive.google.com/file/d/1ZTOVI5Vp3KTQFDt5moJThJ_xYp2pKBAK/view?usp=sharing): 1,500 training and 650 test image-mask pairs.
- [ISIC 2018 (Google Drive)](https://drive.google.com/file/d/1AOpPgSEAfgUS2w4rCGaJBbNYbRh3Z_FQ/view?usp=sharing): 1,886 training and 808 test image-mask pairs.

Place the datasets in `./data/isic17/` and `./data/isic18/`. The `val` directory stores the fixed test split used by the current loader.

```text
data/isic17/
├── train/
│   ├── images/
│   │   └── *.png
│   └── masks/
│       └── *.png
└── val/
    ├── images/
    │   └── *.png
    └── masks/
        └── *.png
```

Use the same structure for `data/isic18/`.

### CVC-ClinicDB

- [Training dataset (Google Drive)](https://drive.google.com/file/d/1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb/view?usp=sharing): use the CVC-ClinicDB subset containing 550 training images.
- [Testing dataset (Google Drive)](https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view?usp=sharing): use the CVC-ClinicDB subset containing 62 test images.

The linked PraNet packages also contain other polyp datasets; only extract the CVC-ClinicDB image-mask pairs for this split.

```text
data/CVC_ClinicDB/
├── train/
│   ├── images/
│   │   └── *.png
│   └── masks/
│       └── *.png
└── val/
    ├── images/
    │   └── *.png
    └── masks/
        └── *.png
```

### BUSI

Download BUSI from [Kaggle](https://www.kaggle.com/aryashah2k/breast-ultrasound-images-dataset). The dataset contains 780 images; the reported experiments use the 647 benign and malignant cases, with a fixed split of 517 training and 130 test images. Normal cases are not used because they contain no lesion regions.

```text
inputs/BUSI/
├── images/
│   └── *.png
└── masks/
    └── 0/
        └── *.png
```

For binary segmentation, use mask folder `0`.

## 2. Configure the Dataset Path

Set `data_path` in `config/tiny_config_isic.py`:

```python
data_path = "/path/to/ISIC/"
```

The current release provides the ISIC training configuration. The CVC-ClinicDB and BUSI links above document the fixed datasets used in the paper.

## 3. Train WFDViM

Run the training script from the repository root:

```bash
python tiny_train_isic.py
```

Training outputs are saved under `results/`.

## 4. Weak-Boundary Robustness Protocol

`robustness/busi_weak_boundary.py` provides the model-independent BUSI protocol used in the paper:

- It applies a `bior2.2` DWT to each resized RGB channel, retains the low-frequency subband, injects deterministic Gaussian fields into the three high-frequency subbands, and scales the reconstructed image to 25 dB PSNR after clipping.
- It defines the weak-boundary subset using a normalized boundary-gradient score. The first quartile from the 517 training images is used as a fixed threshold; this selects 31 of the 130 test images in the reported split.

The protocol uses only images and ground-truth masks. It contains no model weights, checkpoint paths, or prediction-dependent subset selection.

## 5. Main Files

```text
config/tiny_config_isic.py       ISIC training configuration
datasets/dataset.py              Dataset loading and transforms
models/wfdvim/                   WFDViM implementation
models/wavelet/                  Wavelet and frequency-decoupling modules
robustness/busi_weak_boundary.py BUSI perturbation and subset protocol
kernels/                         Custom CUDA kernels
tiny_train_isic.py               Training entry
engine.py                        Training and validation loops
utils.py                         Losses and utilities
```

## 6. Citation

The paper is currently under submission. Citation information will be updated after publication.

## 7. Acknowledgments

This repository is built upon and inspired by [VM-UNet](https://github.com/JCruan519/VM-UNet), [TinyViM](https://github.com/xwmaxwma/TinyViM), [Spatial-Mamba](https://github.com/EdwardChasel/Spatial-Mamba), [VMamba](https://github.com/MzeroMiko/VMamba), and [Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet). We thank the authors for their public implementations.

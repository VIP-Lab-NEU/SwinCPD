# SwinCPD: A Swin Transformer-Based Network for Color-Polarization Demosaicing

This repository contains the source code of our paper **"SwinCPD: A Swin Transformer-Based Network for Color-Polarization Demosaicing"**.

---

## Abstract

Color-Polarization Filter Array (CPFA) sensors provide a single-snapshot solution for simultaneously capturing color and polarization information. However, the resulting raw mosaic makes the demosaicing process a significant challenge, as the CPFA sensor samples the color and polarization information of scenes very sparsely. Existing methods struggle with this task and often result in insufficient accuracy and high computational complexity. In this paper, a novel approach for color-polarization demosaicing is proposed. First, the demosaicing problem is reframed as a full-resolution, 12-channel image restoration task by generating a dense input via nearest-neighbor interpolation, which preserves spatial integrity and produces predictable, block-like artifacts ideal for network-based correction. Second, we propose a Swin Transformer-based network for color-polarization demosaicing (SwinCPD). The input data are processed in four parallel streams corresponding to each polarization angle. A dynamic Feature-wise Linear Modulation (FiLM) mechanism is employed to intelligently model the complex inter-polarization correlations. Extensive experiments on public datasets demonstrate that the proposed method significantly outperforms state-of-the-art techniques, improving the Peak Signal-to-Noise Ratio (PSNR) by up to 2.0 dB and reducing the Angle of Linear Polarization (AoLP) error by nearly 2 degrees.

---

## Introduction

Please refer to [Introduction.md](Introduction.md) for a brief overview.

---

## Requirements

| Component   | Minimum Version/Amount         |
|-------------|--------------------------------|
| PyTorch     | 2.0.0 or later                 |
| CUDA        | 11.8 or later                  |
| VRAM        | ≥ 40 GB                        |

---

## Datasets

We have reorganized the original datasets to align with SwinCPD's training and testing pipeline. Rearranged versions of four existing high-quality datasets are available via the following links:

🔗 **[BaiduNetdisk](https://pan.baidu.com/s/19OoyK5yG0wpubVu_ThZpDQ?pwd=33jn)**
🔗 **[OneDrive](https://1drv.ms/u/c/cc548fe0a93a23b3/EVci7642W_hAhsR_MBtxQIcBWv_o4lSVCLBqln8vW2Zq8A?e=YFea1c)**

> **Note**: Our code supports both **8-bit** and **16-bit** images. The bit depth of the dataset **must match** the version of the code you are using.

### Required Dataset Structure
```
Dataset/
├── Train/
│   ├── 1/
│   ├── 2/
│   └── ...
├── Val/
│   └── ...
└── Test/
    └── ...
```

---

## Training and Testing

After organizing the dataset in the required structure, execute `Preprocessing.py` to generate SwinCPD input images. Next, run `Train.py` to train the model. Finally, use `Test.py` to evaluate the trained model.

---

## Original Dataset Sources
- [**Qiu's Dataset**](https://github.com/qsimeng/Polarization-Demosaic-Code) (8-bit PNG)  

- [**Wen's Dataset**](https://github.com/wsj890411/CPDNet) (8-bit PNG)  

- [**Monno's Dataset**](https://github.com/ymonno/EARI-Polarization-Demosaicking) (16-bit PNG)  

- [**Guo's Dataset**](https://github.com/PRIS-CV/PIDSR) (16-bit PNG)  

> In our paper, we primarily evaluate on **Qiu's** and **Monno's** datasets, but we encourage experimentation across all four.

---

## Acknowledgements

We sincerely thank the authors of the works mentioned above for providing high-quality color-polarization datasets that enabled research in this field.

Additionally, SwinCPD is built upon the **Swin Transformer** architecture. We gratefully acknowledge:
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
- [SwinIR](https://github.com/JingyunLiang/SwinIR)

---

## Citation

If you find SwinCPD helpful to your research, please cite:

T. Wang, X. Zhang, J. Li, Y. Zheng, J. Ru, and H. Xu, “SwinCPD: A Swin Transformer-Based Network for Color-Polarization Demosaicing,” IEEE Sensors Journal, [doi: 10.1109/JSEN.2025.3630744](https://doi.org/10.1109/JSEN.2025.3630744).

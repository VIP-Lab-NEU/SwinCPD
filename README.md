# 🌟 SwinCPD: A Swin Transformer-Based Network for Color-Polarization Demosaicing

This repository contains the source code of our paper **"SwinCPD: A Swin Transformer-Based Network for Color-Polarization Demosaicing"**.

---

## 📖 Abstract

Color-Polarization Filter Array (CPFA) sensors provide a single-snapshot solution for simultaneously capturing color and polarization information. However, the resulting raw mosaic makes the demosaicing process a significant challenge, as the CPFA sensor samples the color and polarization information of scenes very sparsely. Existing methods struggle with this task and often result in insufficient accuracy and high computational complexity. In this paper, a novel approach for color-polarization demosaicing is proposed. First, the demosaicing problem is reframed as a full-resolution, 12-channel image restoration task by generating a dense input via nearest-neighbor interpolation, which preserves spatial integrity and produces predictable, block-like artifacts ideal for network-based correction. Second, we propose a Swin Transformer-based network for color-polarization demosaicing (SwinCPD). The input data are processed in four parallel streams corresponding to each polarization angle. A dynamic Feature-wise Linear Modulation (FiLM) mechanism is employed to intelligently model the complex inter-polarization correlations. Extensive experiments on public datasets demonstrate that the proposed method significantly outperforms state-of-the-art techniques, improving the Peak Signal-to-Noise Ratio (PSNR) by up to 2.0 dB and reducing the Angle of Linear Polarization (AoLP) error by nearly 2 degrees.

---

## 🧩 Introduction

Please refer to [Introduction.md](Introduction.md) for a brief overview.

---

## 🛠️ Requirements

| Component   | Minimum Version/Amount         |
|-------------|--------------------------------|
| PyTorch     | 2.0.0 or later                 |
| CUDA        | 11.8 or later                  |
| VRAM        | ≥ 40 GB                        |

---

## 📂 Datasets

We have reorganized the original datasets to align with SwinCPD's training and testing pipeline. Rearranged versions of four existing high-quality datasets are available via the following links:

🔗 **[BaiduNetdisk](https://pan.baidu.com/s/19OoyK5yG0wpubVu_ThZpDQ?pwd=33jn)**
🔗 **[OneDrive](https://1drv.ms/u/c/cc548fe0a93a23b3/EVci7642W_hAhsR_MBtxQIcBWv_o4lSVCLBqln8vW2Zq8A?e=YFea1c)**

> **Note**: Our code supports both **8-bit** and **16-bit** images. The bit depth of the dataset **must match** the version of the code you are using.

> **The datasets provided here already contain synthesized raw images. If you are using your own synthesized raw images, please modify the sensor layout in Preprocessing.py to match the sensor layout you used when synthesizing the raw images.**

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
Preprocessing.py
SwinCPD_Model.py
Test.py
Train.py
```

---

## 🔬 Training and Testing

After organizing the dataset in the required structure, execute `Preprocessing.py` to generate SwinCPD input images. Next, run `Train.py` to train the model. Finally, use `Test.py` to evaluate the trained model.

---

## 💾 Original Dataset Sources
- [**Qiu's Dataset**](https://github.com/qsimeng/Polarization-Demosaic-Code) (8-bit PNG)  

- [**Wen's Dataset**](https://github.com/wsj890411/CPDNet) (8-bit PNG)  

- [**Monno's Dataset**](https://github.com/ymonno/EARI-Polarization-Demosaicking) (16-bit PNG)  

- [**Guo's Dataset**](https://github.com/PRIS-CV/PIDSR) (16-bit PNG)  

> In our paper, we primarily evaluate on **Qiu's** and **Monno's** datasets, but we encourage experimentation across all four.

---

## 🧹 Update: A Specialized Denoising Tool for Computational Noise in AoLP Images

We have developed a post-processing algorithm specifically designed to mitigate the severe computational noise often observed in Angle of Linear Polarization (AoLP) images. Unlike standard sensor noise, this artifact is a byproduct of the mathematical instability inherent in polarization calculations, particularly in regions with a low Degree of Linear Polarization (DoLP).

**The Challenge: Instability in Low-DoLP Regions**

The AoLP is derived using the formula 0.5*arctan(S2/S1). In unpolarized or weakly polarized regions (low DoLP), the values of the Stokes parameters S1 and S2 are both close to zero. Consequently, the ratio S2/S1 becomes mathematically unstable, fluctuating wildly between extremely large positive and negative values. This instability manifests visually as chaotic white and black dots that significantly degrades image interpretability, even if the original sensor noise was minimal.

**Our Solution: Denoising Normalized Stokes Precursors**

Attempting to denoise the AoLP image directly is ineffective due to the phase wrapping property of angles (where -π/2 and π/2 represent the same physical orientation but opposite pixel values). Standard denoisers often blur it, creating artifacts. Instead, our approach targets the **Normalized Stokes Parameters**. The process is as follows:

1.  **Normalization:** We first calculate the normalized parameters P1 = S1/S0 and P2 = S2/S0.
2.  **Denoising:** We apply the **BM3D** (Block-Matching and 3D Filtering) algorithm separately to P1 and P2.
3.  **Reconstruction:** The final clean AoLP image is recalculated using the denoised P1 and P2 components.

**Usage Instructions:**
The source code is in the `AoLP Denoising` folder. First, select the script corresponding to your dataset's bit depth (**8-bit** or **16-bit**).

For 8-bit datasets, run `Calculate_Stokes_8bit.py` to generate the S0 S1 S2 P1 P2 images required for the denoising process, then run `Denoise_8bit.py` to generate the denoised AoLP image.

For 16-bit datasets, run `Calculate_Stokes_16bit.py` to generate the S0 S1 S2 P1 P2 images required for the denoising process, then run `Denoise_16bit.py` to generate the denoised AoLP image.

**Optional**
1.  **Denoising DoLP:** DoLP is calculated as sqrt(S1^2+S2^2)/S0, which equals to sqrt(P1^2+P2^2). We also save the DoLP calculated by the denoised P1 and P2. The denoised DoLP has better perceptual quality.
2.  **Enhancing DoLP Contrast:** We can notice that the DoLP calculated from demosaiced images appears "darker" or more washed out than the Ground Truth. This occurs because deep learning models tend to output smoothed predictions in uncertain areas to minimize error. This smoothing effect dampens the amplitude of the polarization signals, acting as a low-pass filter that reduces the distinction between polarization angles. Since DoLP is calculated as the magnitude of these signals, over-smoothing mathematically lowers the final DoLP values. To restore this lost signal, we apply an **unsharp masking** operation to the denoised P1 and P2 maps. This boosts the high-frequency details and signal amplitude that the network may have attenuated, resulting in a brighter DoLP image with clearer textures and more accurate polarization contrast. This can increase the PSNR of DoLP by about 0.1-0.3 dB.

---

## 🤝 Acknowledgements

We sincerely thank the authors of the works mentioned above for providing high-quality color-polarization datasets that enabled research in this field.

Additionally, SwinCPD is built upon the **Swin Transformer** architecture. We gratefully acknowledge:
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
- [SwinIR](https://github.com/JingyunLiang/SwinIR)

---

## 📚 Citation

If you find SwinCPD helpful to your research, please cite:

T. Wang, X. Zhang, J. Li, Y. Zheng, J. Ru, and H. Xu, “SwinCPD: A Swin Transformer-Based Network for Color-Polarization Demosaicing,” IEEE Sensors Journal, [doi: 10.1109/JSEN.2025.3630744](https://doi.org/10.1109/JSEN.2025.3630744).

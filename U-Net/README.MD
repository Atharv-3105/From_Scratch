# 🧬 U-Net from Scratch

This repository contains a **from-scratch implementation of U-Net**, a convolutional neural network architecture primarily used for **image segmentation** tasks. The model was first introduced in the paper *"U-Net: Convolutional Networks for Biomedical Image Segmentation"* by Ronneberger et al. (2015).  

---

## 📌 Overview
- Implemented U-Net architecture **from scratch** (without relying on high-level libraries like `segmentation_models`).
- Supports **encoder-decoder** architecture with **skip connections**.
- Trained and tested on standard **segmentation datasets** (e.g., biomedical or custom datasets).
- Modular code structure for easy modifications and experimentation.

---

## ⚙️ Architecture
The U-Net consists of:
1. **Contracting Path (Encoder)** – captures context using convolution + pooling.
2. **Bottleneck** – deepest representation of features.
3. **Expanding Path (Decoder)** – enables precise localization via up-convolutions.
4. **Skip Connections** – combine high-resolution features from encoder with decoder outputs.

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1200/1*L6JX2jHV3jQvZ8D5JYjC5A.png" width="500"/>
</p>

---

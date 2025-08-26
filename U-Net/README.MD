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
  <img src="https://www.google.com/url?sa=i&url=https%3A%2F%2Flmb.informatik.uni-freiburg.de%2Fpeople%2Fronneber%2Fu-net%2F&psig=AOvVaw3-IOLkHMY6Z9NhBPRdh7WO&ust=1756301887127000&source=images&cd=vfe&opi=89978449&ved=0CBUQjRxqFwoTCICunMXMqI8DFQAAAAAdAAAAABAE" width="500"/>
</p>

---

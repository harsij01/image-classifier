# PyTorch Image Classifier with Gradio

A real-time image classification application powered by **PyTorch** and **ResNet-50**, featuring an interactive web user interface built with **Gradio**.

This project processes uploaded images, runs inference using a pre-trained Deep Residual Network, and displays top predicted ImageNet object categories with confidence scores.

---

## ✨ Features

- **Pre-trained ResNet-50 Engine**: Leverages deep feature representations trained on the 1,000-class ImageNet dataset.
- **Top-5 Confidence Breakdown**: Displays categorical probabilities using Gradio's visual confidence meters.
- **Robust Image Processing**: Handles standard JPEG, PNG with transparency (RGBA conversion), and arbitrary aspect ratios via PyTorch transforms.
- **Interactive UI**: User-friendly drag-and-drop web interface built using Gradio.

---

## 🛠️ Project Structure

```text
├── main.py             # Main application script with PyTorch logic & Gradio interface
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
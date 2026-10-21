# Face Image Generation using GAN

## 📌 Project Overview

This project implements a **Generative Adversarial Network (GAN)** using PyTorch to generate new face images by learning patterns from the CelebA face dataset.

A GAN consists of two neural networks, a Generator and a Discriminator, that compete against each other during training to produce realistic images.

## 🎯 Objectives

- Understand the working of Generative Adversarial Networks.
- Generate new face images from random noise.
- Train a Generator and Discriminator using PyTorch.
- Visualize generated images after each training epoch.

## 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Pillow (PIL)

## 🧠 Model Architecture

### 1. Generator

The Generator takes a random noise vector of 100 dimensions as input and transforms it into a 64 × 64 RGB image.

**Architecture:**
- Linear (100 → 256) + ReLU
- Linear (256 → 512) + ReLU
- Linear (512 → 1024) + ReLU
- Linear (1024 → 12,288) + Tanh

### 2. Discriminator

The Discriminator takes a 64 × 64 RGB image as input and predicts whether it is real or generated.

**Architecture:**
- Flatten the input image
- Linear (12,288 → 1,024) + ReLU
- Linear (1,024 → 512) + LeakyReLU
- Linear (512 → 256) + LeakyReLU
- Linear (256 → 1) + Sigmoid

## 📂 Dataset

This project uses the **CelebA (CelebFaces Attributes) dataset**, which contains celebrity face images.

The images are preprocessed using the following transformations:

- Center cropping
- Resizing to 64 × 64 pixels
- Conversion to PyTorch tensors
- Normalization to the range −1 to +1

Download the dataset from the [official CelebA website](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

Update the dataset path in the code according to your environment.

## ⚙️ Installation

Install the required libraries:

```bash
pip install torch torchvision numpy matplotlib pillow
```

## 🚀 How to Run

1. Download and extract the CelebA dataset.
2. Update `root_dir_path` in the Python code to point to the folder containing the images.
3. Run the Python script or execute the code in Google Colab.
4. The model will train for 8 epochs.
5. Generated images will be displayed after each epoch.

The code automatically uses a CUDA-compatible GPU when available; otherwise, it uses the CPU.

## 🔄 Training Process

1. Load real face images in batches using PyTorch DataLoader.
2. Generate fake images from random noise.
3. Train the Discriminator to distinguish real images from fake images.
4. Train the Generator to produce images that can fool the Discriminator.
5. Calculate losses using Binary Cross Entropy Loss.
6. Update the networks using the Adam optimizer.
7. Visualize generated images after each epoch.

## 📊 Hyperparameters

| Parameter | Value |
|---|---|
| Latent dimension | 100 |
| Image size | 64 × 64 |
| Image channels | 3 (RGB) |
| Batch size | 128 |
| Number of epochs | 8 |
| Learning rate | 0.0002 |
| Optimizer | Adam |
| Loss function | Binary Cross Entropy Loss |

## 📈 Results

The model generates and displays batches of synthetic face images after each epoch. The generated images can be inspected to observe how the Generator learns from the CelebA dataset over time.

The quality of the generated images depends on training duration, model architecture, hyperparameters, and dataset quality.

## 🔮 Future Improvements

- Implement a Deep Convolutional GAN (DCGAN) for improved image generation.
- Train for more epochs and experiment with hyperparameters.
- Save generated images automatically for comparison.
- Save and reload trained model checkpoints.
- Explore techniques to improve GAN training stability.

## 📚 Learning Outcomes

- Understanding GAN architecture and adversarial training.
- Working with custom PyTorch datasets and DataLoaders.
- Learning about neural network layers, activation functions, and optimizers.
- Understanding loss functions and gradient-based learning.
- Generating and visualizing synthetic images.

## 👨‍💻 Author

**Priyanshu Chaubey**


---

*This project was developed for learning and experimenting with Generative Adversarial Networks and image generation using PyTorch.*

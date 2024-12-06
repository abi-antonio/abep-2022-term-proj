# Trash Classifier using EfficientNet

This repository contains the source code and results for a term project aimed at building an image classifier for trash categorization using deep learning. The project leverages the EfficientNet-V2 and ResNet-50 architectures, with an emphasis on overcoming challenges associated with small and noisy datasets through data augmentation.

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Results](#results)
- [How to Run](#how-to-run)
- [Acknowledgments](#acknowledgments)

---

## Problem Statement
Properly managing trash is crucial to prevent environmental pollution. This project aims to automate trash classification in images using machine learning, facilitating its application in drones, robots, or anti-littering surveillance systems. 

Given images of trash in diverse environments, the goal is to classify the objects into specific categories.

---

## Dataset
The project uses the **TACO Dataset**, which contains 4,700 labeled data points. Key preprocessing and modifications include:
- Selected 904 usable images with classifications: **Bottle**, **Can**, **Cup**.
- Train-test split: 75%-25% hold-out cross-validation.
- Preprocessing: 
  - Cropped bounding boxes from the dataset.
  - Resized images to \(300 \times 300\) with reflective padding for aspect ratio preservation.
- Data Augmentation:
  - Horizontal and vertical flips.
  - Rotations.

---

## Model Architecture
### EfficientNet-V2
- Pre-trained on ImageNet.
- Fine-tuned with the final fully connected layer replaced for this task.

### ResNet-50
- Also trained using data augmentation for comparison.

---

## Training Details
- **Original Dataset**:
  - Epochs: 30
  - Batch size: 16
  - Learning rate: 0.001
- **Dataset with Augmentation**:
  - Epochs: 10
  - Batch size: 8
  - Learning rate: 0.001

## Results
### Training Set
| Metric                  | EfficientNet-V2 (Original) | EfficientNet-V2 (Augmented) | ResNet-50 (Augmented) |
|-------------------------|----------------------------|-----------------------------|------------------------|
| Training Accuracy       | 92%                       | 98%                         | 98%                    |
| Training F1-score       | 0.86                      | 0.99                        | 0.99                   |

### Test Set
| Metric                  | EfficientNet-V2 (Original) | EfficientNet-V2 (Augmented) | ResNet-50 (Augmented) |
|-------------------------|----------------------------|-----------------------------|------------------------|
| Test Loss               | 0.4310                    | 0.3632                      | 0.5678                 |
| Test Accuracy           | 83%                       | 87%                         | 81%                    |
| Weighted Precision      | 0.84                      | 0.88                        | 0.81                   |
| Weighted Recall         | 0.83                      | 0.87                        | 0.81                   |
| Weighted F1-score       | 0.83                      | 0.87                        | 0.81                   |



### Observations
- Data augmentation significantly improved performance.
- Misclassifications were often due to image quality or inherent dataset noise.
- Overfitting was observed with the augmented data, highlighting the need for more diverse training data.

## How to Run
### Prerequisites
- Python 3.8+
- TensorFlow
- Jupyter Notebook

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```
2. Open the provided Colab notebooks for specific experiments:
   - [Dataset Cleaning](https://colab.research.google.com/drive/1-Bca_r6tP4vibWfilvTo-aJmnjfxxd-I?usp=sharing)
   - [EfficientNet (Original)](https://colab.research.google.com/drive/1QUoxtqN3vGHkvhBjggo6wQcHcoz_xI-_?usp=sharing)
   - [EfficientNet (Augmented)](https://colab.research.google.com/drive/1WK5mRxyZhJyIqsgntFURGalFvNMzVqZQ?usp=sharing)

3. Train the models or evaluate them using the preprocessed data.

---

## Acknowledgments
This project was developed as part of an academic term project:
- **Author**: Abigail Althea A. Antonio  
- **Contact**: [abiax4@gmail.com](mailto:abiax4@gmail.com)

The dataset used is open-sourced under the **TACO Dataset** project. 

---

Contributions and feedback are welcome! Please feel free to submit issues or pull requests.
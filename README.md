# CIFAR-10: CNN Classification, Adversarial Attack, Mitigation & Dynamic Quantization

This project provides an end-to-end exploration of image classification on the CIFAR-10 dataset. It covers:
1.  Training a custom Convolutional Neural Network (CNN).
2.  Demonstrating its vulnerability to Fast Gradient Sign Method (FGSM) adversarial attacks.
3.  Evaluating common image processing techniques as mitigation strategies.
4.  Investigating the impact of dynamic quantization (Linear layers only) on the trained model, assessing changes in model size, accuracy, and its robustness against transferred adversarial attacks and subsequent mitigation efforts.

## Table of Contents
- [Project Summary](#project-summary)
- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [How to Run](#how-to-run)
- [Expected Output & Results](#expected-output--results)
- [File Structure](#file-structure)

## Project Summary

The notebook walks through the following stages:

1.  **Dataset & Preprocessing**:
    *   **Dataset**: CIFAR-10 (32x32 RGB images, 10 classes).
    *   **Split**: Standard 50k train (10% validation) / 10k test.
    *   **Augmentation (Train)**: Random Cropping (padding 4), Random Horizontal Flips.
    *   **Normalization**: CIFAR-10 specific mean/std for train and test sets.
    *   **Defense Preprocessing**: Denormalization -> Image Processing -> Re-normalization.

2.  **Model Architecture & Training (FP32 ImprovedCNN)**:
    *   **Architecture**: Custom CNN with 3 convolutional blocks (Conv2D → BatchNorm → ReLU → Conv2D → BatchNorm → ReLU → MaxPool) followed by a classifier head (Flatten → Dropout → Linear → ReLU → Dropout → Linear).
    *   **Training**: 25 epochs, AdamW optimizer, ReduceLROnPlateau scheduler, Automatic Mixed Precision (AMP) on CUDA, Early Stopping.
    *   **FP32 Performance**: Achieved ~86.74% test accuracy.

3.  **Adversarial Attack (FGSM on FP32 Model)**:
    *   **Method**: Fast Gradient Sign Method (FGSM) from `torchattacks`.
    *   **Parameters**: Epsilon (ε) = 0.03 (max perturbation in `[0,1]` image space).
    *   **Outcome**: Successfully misclassified images (e.g., "ship" to "automobile").

4.  **Mitigation Strategies (on FP32 Adversarial Images)**:
    *   Evaluated four image pre-processing techniques:
        *   **Gaussian Blur (k=3)**: Successful.
        *   **JPEG Compression (q=30)**: Successful.
        *   **Median Filter (k=3)**: Failed.
        *   **Total Variation (TV) Denoise (weight=0.1)**: Failed (misclassified as "airplane").

5.  **Model Quantization (Dynamic)**:
    *   **Method**: Dynamic quantization applied to the trained FP32 ImprovedCNN.
    *   **Scope**: Only `torch.nn.Linear` layers quantized to `torch.qint8`.
    *   **Evaluation**: Performed on CPU.

6.  **Effects of Quantization & Robustness of Quantized Model**:
    *   **Model Size Reduction**: From 13.02 MB (FP32) to 6.72 MB (Dynamically Quantized).
    *   **Accuracy Preservation**: Minimal drop from 90.62% (FP32) to 90.55% (Quantized).
    *   **Robustness (Transfer Attack)**: Adversarial examples generated on the FP32 model were tested against the quantized model. The transfer attack was successful (e.g., "ship" misclassified as "automobile" by the quantized model).
    *   **Mitigation (Quantized Model)**: Similar mitigation strategies were applied to the transferred adversarial images:
        *   **Gaussian Blur (k=3)**: Successful.
        *   **JPEG Compression (q=30)**: Successful.
        *   **Median Filter (k=3)**: Failed.
        *   **Total Variation (TV) Denoise (weight=0.1)**: Successful.

## Key Features
*   Custom CNN model training for CIFAR-10.
*   Implementation of FGSM adversarial attack.
*   Evaluation of four common image processing defense mechanisms:
    *   Gaussian Blur
    *   JPEG Compression
    *   Median Filter
    *   Total Variation (TV) Denoise
*   Dynamic quantization of Linear layers.
*   Comparative analysis of FP32 vs. quantized model (size, accuracy).
*   Robustness testing of the quantized model against transfer attacks.
*   Visualization of original, adversarial, and defended images with model predictions.

## Prerequisites
*   Python 3.8+
*   PyTorch
*   TorchVision
*   TorchAttacks
*   NumPy
*   Matplotlib
*   OpenCV (cv2)
*   Pillow (PIL)
*   scikit-image
*   Pandas
*   (Optional but Recommended) NVIDIA GPU with CUDA for faster training.

## Setup

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file with the following content:
    ```txt
    torch
    torchvision
    torchattacks
    matplotlib
    numpy
    opencv-python
    Pillow
    scikit-image
    pandas
    ipykernel # For Jupyter
    ```
    Then install using pip:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Ensure you install a PyTorch version compatible with your CUDA setup if you plan to use a GPU.*

## How to Run
The project is contained within the `Quantized_Adversarial_Attack_and_Mitigation_Model.ipynb` Jupyter Notebook.

1.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
    Or Jupyter Lab:
    ```bash
    jupyter lab
    ```
2.  Open the `Quantized_Adversarial_Attack_and_Mitigation_Model.ipynb` file.
3.  **Run cells sequentially:**
    *   **Cell 0 (Setup)**: Imports libraries, defines the model, constants, and utility functions.
    *   **Cell 1 (Model Training)**: Trains the FP32 `ImprovedCNN` model. This will take some time, especially without a GPU. It saves the best model weights to `best_model.pth`.
    *   **Cell 2 (Adversarial Attack and Mitigation - FP32 Model)**: Loads the trained FP32 model, performs an FGSM attack on a sample image, applies defense mechanisms, and visualizes the results.
    *   **Cell 3 (Quantization & Robustness - Dynamic Quantized Model)**: Loads the FP32 model, applies dynamic quantization to its Linear layers, evaluates the quantized model, tests its robustness against adversarial examples generated by the FP32 model (transfer attack), applies defenses, and visualizes these results. It also prints a summary table comparing model sizes and accuracies.

    *Important*:
    *   Cell 1 must be run before Cells 2 and 3 as they depend on the `best_model.pth` file.
    *   The notebook will download the CIFAR-10 dataset automatically if not found in the `./data` directory.

## Expected Output & Results

*   **Training Output**: Training progress per epoch, validation accuracy, and saving of the best model.
*   **FP32 Model Performance**: Test accuracy of ~86.74%.
*   **FGSM Attack on FP32 Model**: Successful misclassification of a sample "ship" image to "automobile".
*   **Mitigation on FP32 Model**:
    *   Gaussian Blur: Restored correct classification ("ship").
    *   JPEG Compression: Restored correct classification ("ship").
    *   Median Filter: Failed ("automobile").
    *   TV Denoise: Failed (misclassified as "airplane").
*   **Dynamic Quantization**:
    *   Model size reduced from 13.02 MB to 6.72 MB.
    *   Test accuracy maintained at ~86.72% (negligible drop from FP32).
*   **FGSM Transfer Attack on Quantized Model**: The adversarial image generated by the FP32 model also fooled the quantized model (misclassified as "automobile").
*   **Mitigation on Quantized Model (after transfer attack)**:
    *   Gaussian Blur: Restored correct classification ("ship").
    *   JPEG Compression: Restored correct classification ("ship").
    *   Median Filter: Failed ("automobile").
    *   TV Denoise: Restored correct classification ("ship").
*   **Visualizations**: Plots showing the original image, adversarial image, and images after applying different defense mechanisms, along with the model's prediction for each.
*   **Summary Table**: A pandas DataFrame comparing the FP32 and dynamically quantized models in terms of size, accuracy, and evaluation device.

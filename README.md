# AI vs Real Image Detection using Global FFT

## 1. Overview

**Synapse-7** is a deep learning classifier designed to distinguish between **AI-generated synthetic imagery** and **real images**. Unlike traditional CNNs that rely solely on spatial pixel patterns, this project implements a **Frequency Domain Analysis** pipeline to detect the spectral artifacts often left behind by generative models (GANs and Diffusion models).

By transforming input images into frequency maps using **Fast Fourier Transforms (FFT)**, the model exposes high-frequency irregularities invisible to the human eye. These spectral features are then processed by a custom-tuned **ResNet-34** backbone.

---

## 2. Core Idea
AI-generated images contain subtle, consistent artifacts in the frequency domain that are invisible to the naked eye but obvious to spectral analysis. While every generator (e.g., Midjourney, BigGAN) has a unique style, they all share common mathematical irregularities caused by how they synthesize pixels.

Instead of looking at what is in the image, this model looks at how the image is constructed. By transforming the full image into its Global FFT Magnitude, the model learns to ignore the subject matter and focus on these hidden synthetic patterns. This allows Synapse-7 to act as a universal detector, distinguishing Real from AI regardless of which specific tool created the image.

---

## 3. Repository Structure

```
AI-Generated-Image-Detector/
├── README.md                  # Project overview & training details
├── notebooks/
│   ├── resnet34-globalfft-v1-training.ipynb
│   └── resnet34-globalfft-v1-evaluation.ipynb
├── results/
    └── classification report
    └── confusion matrix
    └── generator wise performance results                      
├── models/
│   └── README.md              # External model weight links
└── requirements.txt
```

---

## 4. Model Architecture

* **Backbone:** ResNet-34 (ImageNet pretrained)
* **Input:** RGB image
* **Feature Augmentation:** Global FFT magnitude representation
* **Output:** Binary prediction (AI vs Real)

The FFT features reduce over-reliance on spatial appearance and help the model capture generator-independent frequency artifacts.

---

## 5. Training Strategy

Due to the relatively small size of the Tiny GenImage dataset, **only the final layers of ResNet-34 are fine-tuned**.

The early layers are kept frozen because they already capture generic visual primitives such as:

* edges
* textures
* gradients
* low-frequency structural patterns

Freezing these layers:

* Improves convergence speed
* Stabilizes gradients
* Reduces overfitting risk

Full end-to-end retraining is generally beneficial **only when the dataset is sufficiently large**.


## 6. Dataset & Pipeline

**Tiny GenImage (ImageNet-based)**

### AI Image Generators

* Wukong
* GLIDE
* BigGAN
* VQ-Diffusion (VQDM)
* Midjourney
* Stable Diffusion v5
* ADM (Diffusion Model)

### Real Images

* ImageNet (natural images)


The system utilizes a custom `MultiGenDataset` class designed to aggregate training data from multiple diverse sources (e.g., Midjourney, DALL-E, Stable Diffusion) into a single unified stream. This ensures the model learns generalized artifacts rather than overfitting to a specific generator's style.

### **A. Data Directory Structure**

To reproduce the training results, the dataset must be organized hierarchically. The data loader iterates through every sub-directory in the root `Data/` folder, treating each as an independent source.

```text
Data/
├── [Generator_Source_A] (e.g., Midjourney)
│   ├── train
│   │   ├── ai        # Synthetic Images (Label: 1)
│   │   └── nature    # Real Images (Label: 0)
│   └── val
│       ├── ai
│       └── nature
├── [Generator_Source_B] (e.g., Stable_Diffusion)
│   ├── train...
│   └── val...

```

### **Preprocessing Pipeline**

Before Frequency Domain analysis, raw images undergo a standardized transformation pipeline using `torchvision.transforms`:

1. **Resize & Crop:** Images are resized to `256px` and center-cropped to `224x224` to match the ResNet input requirements while maintaining aspect ratio consistency.
2. **Normalization:** Pixel values are scaled to the standard ImageNet mean and standard deviation:
* `mean=[0.485, 0.456, 0.406]`
* `std=[0.229, 0.224, 0.225]`


3. **Tensor Conversion:** Converted to PyTorch tensors (`C x H x W`).

### **C. Data Module**

The training logic is encapsulated in a `LightningDataModule`, which handles:

* **Batching:** Default batch size of `32` 
* **Workers:** Utilizes `multiprocessing` (2 workers) for efficient data loading.


## 7. Evaluation

### Confusion Matrix
![Screenshot_13-2-2026_22588_www kaggle com](https://github.com/user-attachments/assets/53cede17-1a1f-45e7-ad90-1d05513a3ac1)

### Classification Report

```
              precision    recall  f1-score   support

        REAL       0.92      0.92      0.92      3500
          AI       0.92      0.92      0.92      3500

    accuracy                           0.92      7000
   macro avg       0.92      0.92      0.92      7000
weighted avg       0.92      0.92      0.92      7000
```
### Generator wise evaluation

1. Generator: imagenet_ai_0424_wukong
```
              precision    recall  f1-score   support

           0       0.88      0.94      0.91       500
           1       0.93      0.87      0.90       500

    accuracy                           0.90      1000
   macro avg       0.91      0.90      0.90      1000
weighted avg       0.91      0.90      0.90      1000
```

2. Generator: imagenet_glide
```
              precision    recall  f1-score   support

           0       1.00      0.90      0.95       500
           1       0.91      1.00      0.95       500

    accuracy                           0.95      1000
   macro avg       0.96      0.95      0.95      1000
weighted avg       0.96      0.95      0.95      1000
```

3. Generator: imagenet_ai_0419_biggan
```
              precision    recall  f1-score   support

           0       1.00      0.94      0.97       500
           1       0.94      1.00      0.97       500

    accuracy                           0.97      1000
   macro avg       0.97      0.97      0.97      1000
weighted avg       0.97      0.97      0.97      1000
```

4. Generator: imagenet_ai_0419_vqdm
```
              precision    recall  f1-score   support

           0       0.94      0.91      0.93       500
           1       0.92      0.94      0.93       500

    accuracy                           0.93      1000
   macro avg       0.93      0.93      0.93      1000
weighted avg       0.93      0.93      0.93      1000
```

5. Generator: imagenet_midjourney
```
              precision    recall  f1-score   support

           0       0.78      0.91      0.84       500
           1       0.89      0.74      0.81       500

    accuracy                           0.82      1000
   macro avg       0.83      0.82      0.82      1000
weighted avg       0.83      0.82      0.82      1000
```

6. Generator: imagenet_ai_0424_sdv5
```
              precision    recall  f1-score   support

           0       0.94      0.92      0.93       500
           1       0.92      0.94      0.93       500

    accuracy                           0.93      1000
   macro avg       0.93      0.93      0.93      1000
weighted avg       0.93      0.93      0.93      1000
```

7. Generator: imagenet_ai_0508_adm
```
              precision    recall  f1-score   support

           0       0.95      0.91      0.93       500
           1       0.92      0.96      0.94       500

    accuracy                           0.93      1000
   macro avg       0.93      0.93      0.93      1000
weighted avg       0.93      0.93      0.93      1000
```

---

## 8. Limitations & Future Work

While Synapse-7 achieves high accuracy on the validation set, the current implementation operates under specific constraints. This section outlines the boundaries of the system and identifies key areas for scaling.

### **A. Resource Constraints & Scope**

1. **Dataset Reduction:**
* The model was trained on the **"Tiny GenImage"** subset (~35,000 images) rather than the full **GenImage Benchmark** (which contains >1 million pairs). While sufficient for proof-of-concept, this reduced scope limits the model's exposure to the long-tail diversity of the full dataset.


2. **Backbone Efficiency vs. Capacity:**
* We deliberately selected **ResNet-34** over larger variants (like ResNet-50 or ResNet-101).
* *Rationale:* Given the hardware constraints (single GPU), ResNet-34 offered the optimal trade-off between training speed and feature extraction capability, preventing bottlenecks without sacrificing convergence stability.


3. **Hyperparameter Heuristics:**
* Due to hardware limitations, extensive **Neural Architecture Search (NAS)** and automated hyperparameter tuning (e.g., Optuna sweeps) were not performed. The current configuration relies on heuristic best practices (e.g., standard AdamW defaults) rather than empirically optimized values.


4. **. Legacy Generator Bias (Temporal Domain Shift)**

* **The Issue:** The training dataset is derived from older, "legacy" generative architectures (e.g., **BigGAN**, **GLIDE**, **Midjourney v4**, and **Stable Diffusion v1.5/v2**).
* **The Consequence:** These earlier models left distinct spectral fingerprints (e.g., strong checkerboard artifacts from older upsampling layers) that Synapse-7 has learned to target.
* **The Risk:** The model may experience performance degradation when inference is run on **state-of-the-art (SOTA)** generators like **Flux.1**, **Midjourney v6**, or **DALL-E 3**. Newer architectures use advanced sampling schedulers and transformer backbones that produce "cleaner" frequency maps, potentially evading detection (False Negatives).


### **B. Future Scope**

1. **Scale-Up:**
* Train on the full **GenImage** dataset using multi-GPU distributed training (DDP) to validate performance at scale.


2. **Advanced Architectures:**
* Experiment with **Swin Transformers** or **ConvNeXt**, which may capture global frequency dependencies better than standard CNNs.
---



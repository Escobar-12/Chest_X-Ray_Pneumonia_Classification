# Chest X-Ray Classification using EfficientNetB0

This repository contains a TensorFlow/Keras implementation of a deep learning model to classify chest X-ray images into **Normal** and **Pneumonia** categories using the EfficientNetB0 architecture. The model leverages transfer learning and data augmentation to improve accuracy and generalization.

---

## Table of Contents

* [Overview](#overview)
* [Dataset](#dataset)
* [Model Architecture](#model-architecture)
* [Installation](#installation)
* [Training](#training)
* [Evaluation](#evaluation)
* [Results](#results)
* [Saving & Loading Model](#saving--loading-model)

---

## Overview

This project aims to build a high-performance chest X-ray classifier that can differentiate between healthy lungs and pneumonia-affected lungs. The model uses **EfficientNetB0** pre-trained on ImageNet and fine-tunes a few dense layers for binary classification.

**Features:**

* Transfer learning with EfficientNetB0
* Data augmentation for better generalization
* Early stopping and learning rate reduction on plateau
* Confusion matrix and classification report for evaluation

---

## Dataset

Kaggle link: https://www.kaggle.com/datasets/yusufmurtaza01/chest-xray-pneumonia-balanced-dataset
The dataset should have the following structure:

```
data/
├── train/
│   ├── Normal/
│   └── Pneumonia/
└── test/
    ├── Normal/
    └── Pneumonia/
```

* `train`: training and validation split (80-20)
* `test`: independent test set
* Images resized to 224x224 pixels

> **Note:** Adjust the paths in the code according to your dataset location.

---

## Model Architecture

The model uses EfficientNetB0 as a backbone:

* Pre-trained EfficientNetB0 (weights=`imagenet`, `include_top=False`)
* Global Max Pooling
* Dense layers:

  * 512 units + ReLU + Dropout 0.5
  * 256 units + ReLU + Dropout 0.4
  * Output: 2 units (softmax for Normal/Pneumonia)

**Training Strategy:**

* Optimizer: Adam (`learning_rate=0.001`)
* Loss: Categorical Crossentropy
* Metrics: Accuracy
* Callbacks: EarlyStopping and ReduceLROnPlateau

---

## Installation

Clone the repository and install required packages:

```bash
git clone https://github.com/Escobar-12/Chest_X-Ray_Pneumonia_Classification.git
cd chest-xray-classification
pip install -r requirements.txt
```

---

## Requirements
- Python 3.10
- TensorFlow 2.10.1
- NumPy 1.26.4
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- CUDA 11.2 and cuDNN 8.1 (for GPU support)
- Conda (recommended for environment management)

---

## Installation if using GPU for training

1. **Create a conda environment:**

conda create -n tf_gpu python=3.10  
conda activate tf_gpu


2. **Install GPU dependencies (if using GPU):**

conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0


3. **Install Python libraries:**

pip install tensorflow==2.10.1 numpy==1.26.4 pandas matplotlib seaborn scikit-learn

3. Visualize training results and evaluate the model.

---

## Training

* Training is performed with a batch size of 32
* Image preprocessing uses EfficientNetB0 preprocessing (`preprocess_input`)
* Data augmentation includes random brightness and contrast

Example plot outputs:

* Training vs Validation Accuracy
* Training vs Validation Loss

---

## Evaluation

After training, the model evaluates on the test set:

* Confusion Matrix
* Classification Report (Precision, Recall, F1-Score)

Example:

```python
y_pred_probs = model.predict(test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = ...

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes)
```

---

## Results

* Test Accuracy: ~97% (may vary depending on dataset)
* Confusion matrix and classification report included in scripts

---

## Saving & Loading Model

Save model weights:

```python
model.save_weights('./models/chest_xray_model_weights.h5')
```

Load weights:

```python
model.load_weights('./models/chest_xray_model_weights.h5')
```

---

## License

This project is licensed under the MIT License.

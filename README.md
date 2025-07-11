# 🌌 Galaxy Image Classification with VGG19

## 📌 Project Overview

This project focuses on **classifying galaxies based on their morphology** using convolutional neural networks (CNNs), particularly a fine-tuned **VGG19** model. It leverages the **Galaxy10_DECals** dataset to distinguish between different types of galaxies such as spiral, elliptical, mergers, and more.

The main objective is to apply **transfer learning** and evaluate its performance on a visual classification task using real astronomical data.

---

## 🧰 Technologies Used

- **Python 3.x**
- **TensorFlow / Keras**
- NumPy, Pandas
- Matplotlib, Seaborn, Plotly
- scikit-learn

---

## 📁 Project Structure

- `notebooks/` – exploratory and training notebooks
- `src/` – modular Python scripts (data prep, training, evaluation)
- `models/` – saved models and weights
- `README.md` – this file



---

## 🔍 Key Steps

1. Data preprocessing (filtering selected classes)
2. Image normalization and augmentation
3. Dataset split: training / validation / test
4. Transfer learning with **VGG19** (last 5 layers unfrozen)
5. Model training with:
   - `EarlyStopping`
   - `Dropout`
   - `BatchNormalization`
6. Model evaluation using:
   - Accuracy
   - Confusion matrix
   - Classification report

---

## 🧪 Galaxy Classes

The model is trained to recognize the following 7 galaxy classes:

| Label | Description            |
|-------|------------------------|
| 0     | Disturbed              |
| 1     | Merger                 |
| 2     | Round Smooth           |
| 3     | In-between Smooth      |
| 4     | Cigar Shaped Smooth    |
| 5     | Barred Spiral          |
| 6     | Unbarred Spiral        |

You can limit training to a smaller subset depending on data availability or performance goals.

---

## 📊 Model Architecture

- 🔹 Pretrained **VGG19** convolutional base
- 🔹 Last 5 convolutional layers set as trainable
- 🔹 Custom top:
  - `Flatten`
  - `Dense(256, activation='relu')`
  - `Dropout(0.5)`
  - `Dense(7, activation='softmax')`

---

## ✨ Sample Results

| Epoch | Validation Accuracy | Validation Loss |
|-------|---------------------|-----------------|
| 20    | 0.81                | 0.60            |

The confusion matrix and prediction examples are available in the notebook for visual inspection and performance analysis.



## 🚀 Getting Started

To clone and run this project:

```bash
git clone https://github.com/your-username/galaxy-classification-cnn.git
cd galaxy-classification-cnn

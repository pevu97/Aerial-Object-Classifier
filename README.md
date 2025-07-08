🌌 Galaxy Classification with CNN (VGG19)
📌 Project Overview
This project aims to classify galaxy images into morphological categories using convolutional neural networks. Leveraging the Galaxy10 DECals dataset, we explore the effectiveness of transfer learning with VGG19, fine-tuning it to detect various galaxy types.

The goal is to build a robust image classifier capable of distinguishing between galactic structures such as spiral, elliptical, and merging galaxies.

🧰 Technologies Used
Python 3.x

Pandas, NumPy, Matplotlib, Seaborn, Plotly

Scikit-learn

TensorFlow / Keras

Pretrained VGG19 model

📁 Project Structure

Aerial-Object-Classifier/
│
├── dataset/               # Images divided into train, validation, test
│   ├── train/
│   ├── validation/
│   └── test/
│
├── notebooks/             # Jupyter/Colab notebooks for EDA, training
├── models/                # Saved model weights (HDF5 or SavedModel)
├── src/                   # Optional: scripts for loading, preprocessing, metrics
├── utils/                 # Helpers (e.g. plotting confusion matrix)
└── README.md              # This file
🔍 Key Steps
Dataset filtering and preprocessing

Image augmentation (rescale, flip, rotation)

Transfer learning using VGG19 with selected trainable layers

Early stopping, dropout, and batch normalization

Performance evaluation: accuracy, loss, and confusion matrix

🧪 Galaxy Classes
This model was trained to recognize the following galaxy types:

0: Disturbed

1: Merger

2: Round Smooth

3: In-between Smooth

4: Cigar Shaped Smooth

5: Barred Spiral

6: Unbarred Spiral

(You can restrict training to fewer classes based on dataset availability.)

📊 Metrics Used
Categorical Crossentropy

Accuracy (train and validation)

Confusion matrix

Classification report (precision, recall, F1-score)

✨ Sample Results
Epoch	Val Accuracy	Val Loss
20	0.81	0.60

Confusion matrix and prediction samples are included in the notebook for visual evaluation.

🚀 Getting Started
Clone the repository:

git clone https://github.com/your-username/galaxy-classification-cnn.git
cd galaxy-classification-cnn




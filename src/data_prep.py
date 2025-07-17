
def imports():
  import os
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
  from sklearn.metrics import confusion_matrix, classification_report
  from PIL import Image
  import shutil
  import random
  import plotly.graph_objects as go
  import requests
  from io import BytesIO
  from IPython.display import display

  from tensorflow.keras.preprocessing import image
  from tensorflow.keras.preprocessing.image import ImageDataGenerator
  from tensorflow.keras.models import Sequential
  from tensorflow.keras import layers
  from tensorflow.keras import optimizers
  from tensorflow.keras.optimizers import Adam
  from tensorflow.keras.callbacks import TensorBoard
  from tensorflow.keras.applications import VGG19
  from tensorflow.keras.preprocessing import image_dataset_from_directory
  from tensorflow.keras.callbacks import EarlyStopping
  import h5py
  import numpy as np
  from tensorflow.keras.utils import to_categorical

  np.set_printoptions(precision=6, suppress=True)

     

imports()
     

def download_data():
  !wget https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5

def data_organization():
  with h5py.File('Galaxy10_DECals.h5', 'r') as f:
    images = f['images'][:10500]     # tylko 10500 obrazów
    labels = f['ans'][:10500]        # tylko 10500 etykiet



  images = images.astype('float32')

  labels = labels.astype(int)

  # oryginalne unikalne klasy:
  unique_classes = np.unique(labels)


  # stworzenie mapowania np. {0: 0, 1: 1, 2: 2, 3: 3}
  class_map = {old: new for new, old in enumerate(unique_classes)}


  # zastosowanie mapowania
  labels_mapped = np.array([class_map[label] for label in labels])

  # folder docelowy
  output_dir = 'sorted_images'
  os.makedirs(output_dir, exist_ok=True)

  for i, (img, label) in enumerate(zip(images, labels_mapped)):
      label_dir = os.path.join(output_dir, str(label))
      os.makedirs(label_dir, exist_ok=True)

      # zapisz obraz jako JPEG
      img_path = os.path.join(label_dir, f'image_{i:05d}.jpg')
      img_uint8 = img.astype(np.uint8)  # jeśli potrzebne
      Image.fromarray(img_uint8).save(img_path)


def split_dataset(source_dir, output_dir, split=(0.7, 0.2, 0.1), seed=42):
  random.seed(seed)
  class_names = os.listdir(source_dir)

  for class_name in class_names:
    src_class_dir = os.path.join(source_dir, class_name)
    images = os.listdir(src_class_dir)
    random.shuffle(images)

    n_total = len(images)
    n_train = int(split[0] * n_total)
    n_val   = int(split[1] * n_total)

    split_points = {
    'train': images[:n_train],
    'val': images[n_train:n_train + n_val],
    'test': images[n_train + n_val:]
     }

    for split_name, file_list in split_points.items():
      dest_class_dir = os.path.join(output_dir, split_name, class_name)
      os.makedirs(dest_class_dir, exist_ok=True)

    for fname in file_list:
      shutil.copy2(os.path.join(src_class_dir, fname), os.path.join(dest_class_dir, fname))



if __name__ == "__main__":
    split_dataset("sorted_images", "dataset", split=(0.7, 0.2, 0.1))
     

def show_sample_images_per_class():

  base_dir = "/content/sorted_images"
  class_names = sorted(os.listdir(base_dir))[:6]  # wybierz 4 klasy

  plt.figure(figsize=(10, 10))

  for i, class_name in enumerate(class_names):
      class_dir = os.path.join(base_dir, class_name)
      image_name = random.choice(os.listdir(class_dir))  # losowe zdjęcie
      image_path = os.path.join(class_dir, image_name)

      img = Image.open(image_path)

      class_labels = {
      '0': 'Disturbed',
      '1': 'Merger',
      '2': 'Round Smooth',
      '3': 'In-between Smooth',
      '4': 'Cigar Shaped Smooth',
      '5': 'Barred Spiral Galaxy',
      '6': 'Unbarred Spiral Galaxy'
      }

      name_tag = class_labels.get(class_name, 'Inna klasa')


      plt.subplot(3, 3,i + 1)
      plt.imshow(img)
      plt.title(f"Klasa: {name_tag}")
      plt.axis("off")

  plt.tight_layout()
  plt.show()

  print(img.size)

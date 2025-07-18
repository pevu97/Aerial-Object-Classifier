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

def show_all_label_samples():
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

def display_augmented_images(directory, idx):
    """
    Funkcja zwraca wykres przykładowych obrazów uzyskanych za pomocą techniki
    augmentacji danych.
    """
    fnames = [os.path.join(directory, fname) for fname in os.listdir(directory)]
    img_path = fnames[idx]
    img = image.load_img(img_path, target_size=(150, 150))

    x = image.img_to_array(img)
    x = x.reshape((1, ) + x.shape)

    i = 1
    plt.figure(figsize=(16, 8))
    for batch in train_datagen.flow(x, batch_size=1):
        plt.subplot(3, 4, i)
        plt.grid(False)
        imgplot = plt.imshow(image.array_to_img(batch[0]))
        i += 1
        if i % 13 == 0:
            break


def plot_hist(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    # Accuracy
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=hist['epoch'], y=hist['accuracy'], name='accuracy', mode='markers+lines'))
    fig1.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_accuracy'], name='val_accuracy', mode='markers+lines'))
    fig1.update_layout(width=1000, height=500, title='Accuracy vs. Val Accuracy',
                      xaxis_title='Epoki', yaxis_title='Accuracy', yaxis_type='log')

    fig1.show()

    # Loss
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=hist['epoch'], y=hist['loss'], name='loss', mode='markers+lines'))
    fig2.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_loss'], name='val_loss', mode='markers+lines'))
    fig2.update_layout(width=1000, height=500, title='Loss vs. Val Loss',
                      xaxis_title='Epoki', yaxis_title='Loss', yaxis_type='log')
    fig2.show()
def img_hist():
  import requests
  from io import BytesIO
  from IPython.display import display
  
  url1 = 'https://kocotmeble.com/wp-content/uploads/2025/07/newplot-1.png'
  url2 = 'https://kocotmeble.com/wp-content/uploads/2025/07/newplot.png'
  
  
  response = requests.get(url1)
  img = Image.open(BytesIO(response.content))
  display(img)
  
  
  response = requests.get(url2)
  img = Image.open(BytesIO(response.content))
  display(img)

def plot_confusion_matrix(cm):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()

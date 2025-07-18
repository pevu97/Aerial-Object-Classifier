def generators():
  train_datagen = ImageDataGenerator(
      rotation_range=40,
      rescale=1./255.,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest'
  )
  
  valid_datagen = ImageDataGenerator(
      rescale=1./255.
  )

  train_generator = train_datagen.flow_from_directory(
    '/content/dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
  )

  valid_generator = valid_datagen.flow_from_directory(
      '/content/dataset/val',
      target_size=(224, 224),
      batch_size = 32,
      class_mode = 'categorical'
    )

  train_generator = train_datagen.flow_from_directory(
    '/content/dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
  )

  valid_generator = valid_datagen.flow_from_directory(
      '/content/dataset/val',
      target_size=(224, 224),
      batch_size = 32,
      class_mode = 'categorical'
  )

def build_model():
  conv_base = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
  conv_base.trainable = True
  set_trainable = False

  for layer in conv_base.layers:
      if layer.name == 'block4_conv1':
        set_trainable = True
      if set_trainable:
          layer.trainable = True
      else:
          layer.trainable = False
  
  model = Sequential()
  model.add(conv_base)
  model.add(layers.Flatten())
  model.add(layers.Dense(units=512, activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(units=256, activation='relu'))
  model.add(layers.Dropout(0.3))
  model.add(layers.Dense(units=7, activation='softmax'))
  
  model.compile(optimizer=Adam(learning_rate=5e-6),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  
  model.summary()

def train_model():
  from tensorflow.keras.callbacks import EarlyStopping

  early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
  
  history = model.fit(train_generator,
                      epochs = 50,
                      validation_data = valid_generator,
                      callbacks = early_stopping)
  return history

def test_model():
  test_datagen = ImageDataGenerator(rescale=1./255.)

  test_generator = test_datagen.flow_from_directory(
      '/content/dataset/test',
      target_size = (224, 224),
      batch_size = 1,
      class_mode = 'categorical',
      shuffle = False
  )
  
  y_prob = model.predict(test_generator)
  y_pred = np.argmax(y_prob, axis=-1)
  
  predictions = pd.DataFrame(y_pred, columns=['Class number'])
  y_true = test_generator.classes
  y_pred = predictions['Class number'].values
  classes = list(test_generator.class_indices.keys())
  cm = confusion_matrix(y_true, y_pred)
  accuracy = np.mean(y_pred == y_true)
  print(f"Dokładność: {accuracy * 100:.2f}%")
  
  return cm
  
  

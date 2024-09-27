import os
import urllib.request
import zipfile
import keras
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.preprocessing.image import ImageDataGenerator  # for data augmentation
import matplotlib.pyplot as plt

# 학습데이터 준비
train_url = 'https://storage.googleapis.com/learning-datasets/horse-or-human.zip'
file_name = 'horse-or-human.zip'
train_path = 'horse-or-human/training'

# 검증데이터 준비
val_url = 'https://storage.googleapis.com/learning-datasets/validation-horse-or-human.zip'
val_file_name = 'validation-horse-or-human.zip'
val_path = 'horse-or-human/validation'

if not (os.path.isfile(file_name)):
    urllib.request.urlretrieve(train_url, file_name)
    zip_ref = zipfile.ZipFile(file_name, 'r')
    zip_ref.extractall(train_path)
    zip_ref.close()

    urllib.request.urlretrieve(val_url, val_file_name)
    zip_ref = zipfile.ZipFile(val_file_name, 'r')
    zip_ref.extractall(val_path)
    zip_ref.close()

training_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

training_data = training_datagen.flow_from_directory(
    directory=train_path,
    target_size=(150, 150),  # Setting the size of output images to have in same size.
    batch_size=32,
    class_mode='binary',
    shuffle=True,
    seed=42
)

# print(training_data.class_indices)

# Doing the same for validation dataset.
# But here we do not need to generate the images for Validation, so we just use rescale.
valid_datagen = ImageDataGenerator(rescale=1. / 255)

valid_data = valid_datagen.flow_from_directory(
    directory=val_path,
    target_size=(150, 150),  # Setting the size of output images to have in same size.
    batch_size=32,
    class_mode='binary' # default value
)
cnn_model = keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, input_shape=[150, 150, 3]),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=3),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # drop out regularization
        tf.keras.layers.Dropout(0.5),

        # Neural Network Building
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation='relu'),  # Input Layer
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(units=256, activation='relu'),  # Hidden Layer
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(units=1, activation='sigmoid'),  # Output Layer
    ]
)

from tensorflow.keras.optimizers.legacy import Adam
from keras.callbacks import ModelCheckpoint

cnn_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model_path = 'horse_human_model.h5'
checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = cnn_model.fit(
    training_data,
    epochs = 100,
    verbose = 1,
    validation_data = valid_data,
    callbacks = callbacks_list
)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch or iteration')
plt.legend(["train", 'valid'], loc = 'upper left')
plt.show()
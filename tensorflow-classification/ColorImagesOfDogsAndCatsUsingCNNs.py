# binary classification based machine learning model using CNNs without image augmentation - color images (3D)
# supervised

# CNNs is Convolutional Neural Networks
# convolution is applying a filter/kernel to an image for improving its accuracy

# step 1 is importing dependencies
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import math
import numpy as np
import logging as log
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D

tf.get_logger().setLevel(log.ERROR)
print("tensorflow version is {}".format(tf.__version__))

# step 2 is data loading
_URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
zip_file_path = tf.keras.utils.get_file('cats_and_dogs_filtered.zip', origin=_URL, extract=True)

zip_base_dir = os.path.dirname(zip_file_path)
base_dir = os.path.join(zip_base_dir, 'cats_and_dogs_filtered')

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

total_train_cats_examples = len(os.listdir(train_cats_dir))
total_train_dogs_examples = len(os.listdir(train_dogs_dir))

total_validation_cats_examples = len(os.listdir(validation_cats_dir))
total_validation_dogs_examples = len(os.listdir(validation_dogs_dir))

total_train_examples = total_train_cats_examples + total_train_dogs_examples
total_validation_examples = total_validation_cats_examples + total_validation_dogs_examples

print(zip_base_dir)
print(base_dir)
print(train_dir)
print(validation_dir)
print(train_cats_dir)
print(train_dogs_dir)
print(validation_cats_dir)
print(validation_dogs_dir)

print("Total Train Cats Examples - {}".format(total_train_cats_examples))
print("Total Train Dogs Examples - {}".format(total_train_dogs_examples))
print("Total Validation Cats Examples - {}".format(total_validation_cats_examples))
print("Total Validation Dogs Examples - {}".format(total_validation_dogs_examples))
print("Total Train Examples - {}".format(total_train_examples))
print("Total Validation Examples - {}".format(total_validation_examples))

# step 3 is setting model parameters

# batch size defines number of training examples to process before updating model variables
# image shape means that our training model using each image of 150 pixels width and 150 pixels height

BATCH_SIZE = 100
IMG_SHAPE = 150

# step 4 is data preparation
train_img_generator = ImageDataGenerator(rescale=1. / 255)
validation_img_generator = ImageDataGenerator(rescale=1. / 255)

train_img_dataset = train_img_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                            directory=train_dir,
                                                            shuffle=True,
                                                            target_size=(IMG_SHAPE, IMG_SHAPE),
                                                            class_mode='binary')

validation_img_dataset = validation_img_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                                      directory=validation_dir,
                                                                      shuffle=False,
                                                                      target_size=(IMG_SHAPE, IMG_SHAPE),
                                                                      class_mode='binary')

# dataset type is <class 'keras_preprocessing.image.directory_iterator.DirectoryIterator'>

sample_train_images, _ = next(train_img_dataset)
sample_train_images = sample_train_images[:5]

fig, axes = plt.subplots(1, 5, figsize=(20, 10))
axes = axes.flatten()

for img, ax in zip(sample_train_images, axes):
    ax.imshow(img)

plt.tight_layout()
plt.show()

# step 5 is build the model
model = Sequential([
    Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu, input_shape=(150, 150, 3)),
    MaxPool2D((2, 2), strides=2),
    Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu),
    MaxPool2D((2, 2), strides=2),
    Conv2D(128, (3, 3), padding='same', activation=tf.nn.relu),
    MaxPool2D((2, 2), strides=2),
    Conv2D(128, (3, 3), padding='same', activation=tf.nn.relu),
    MaxPool2D((2, 2), strides=2),
    Flatten(),
    Dense(512, activation=tf.nn.relu),
    Dense(2, activation=tf.nn.softmax)
])

# step 6 is compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# step 7 is display the model summary
model.summary()

# step 8 is train the model

EPOCHS = 100
# after 5 epochs, model has started to memorize instead of generalize on training dataset

history = model.fit_generator(train_img_dataset,
                              epochs=EPOCHS,
                              steps_per_epoch=math.ceil(total_train_examples / BATCH_SIZE),
                              validation_data=validation_img_dataset,
                              validation_steps=math.ceil(total_validation_examples / BATCH_SIZE))

# step 9 is visualizing the training results
plt.figure(figsize=(20, 10))
epoch_range = range(EPOCHS)

plt.subplot(1, 2, 1)
plt.plot(epoch_range, history.history['accuracy'], label='Training Accuracy')
plt.plot(epoch_range, history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epoch_range, history.history['loss'], label='Training Loss')
plt.plot(epoch_range, history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.savefig('images/dogs_and_cats_image_classification_using_cnn.png')
plt.show()

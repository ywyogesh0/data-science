# classification machine learning model using data augmentation and dropout - color images (3D)
# supervised

# importing dependencies
from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
import shutil
import os
import logging as log

tf.get_logger().setLevel(log.ERROR)
print("tensorflow version is {}".format(tf.__version__))

# data preparation
class_of_flowers = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']
_URL = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'

tgz_file_path = tf.keras.utils.get_file('flower_photos.tgz', origin=_URL, extract=True)
tgz_base_dir = os.path.dirname(tgz_file_path)

base_dir = os.path.join(tgz_base_dir, 'flower_photos')

print(tgz_base_dir)
print(base_dir)

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

for c in class_of_flowers:
    class_dir = os.path.join(base_dir, c)

    img_list = glob.glob(class_dir + "/*.jpg")
    img_list_count = round(len(img_list) * 0.8)

    train_img_list, val_img_list = img_list[:img_list_count], img_list[img_list_count:]

    train_class_dir = os.path.join(base_dir, 'train', c)
    validation_class_dir = os.path.join(base_dir, 'validation', c)

    if not os.path.isdir(train_class_dir):
        os.makedirs(train_class_dir)

        for train_img in train_img_list:
            shutil.move(train_img, train_class_dir)

    if not os.path.isdir(validation_class_dir):
        os.makedirs(validation_class_dir)

        for val_img in val_img_list:
            shutil.move(val_img, validation_class_dir)

    print(train_class_dir)
    print(validation_class_dir)

train_img_gen = ImageDataGenerator(rescale=1. / 255,
                                   horizontal_flip=True,
                                   rotation_range=45,
                                   zoom_range=0.3,
                                   height_shift_range=0.15,
                                   width_shift_range=0.15,
                                   shear_range=0.2,
                                   fill_mode='nearest')

validation_img_gen = ImageDataGenerator(rescale=1. / 255)

BATCH_SIZE = 100
IMG_SHAPE = 150

train_img_dataset = train_img_gen.flow_from_directory(batch_size=BATCH_SIZE,
                                                      shuffle=True,
                                                      directory=train_dir,
                                                      target_size=(IMG_SHAPE, IMG_SHAPE),
                                                      class_mode='sparse')

validation_img_dataset = validation_img_gen.flow_from_directory(batch_size=BATCH_SIZE,
                                                                shuffle=False,
                                                                directory=validation_dir,
                                                                target_size=(IMG_SHAPE, IMG_SHAPE),
                                                                class_mode='sparse')


# dataset type is <class 'keras_preprocessing.image.directory_iterator.DirectoryIterator'>


# visualizing the data
def plot_img(img_array):
    fig, axes = plt.subplots(1, 5, figsize=(20, 10))
    axes = axes.flatten()
    for img, ax in zip(img_array, axes):
        ax.imshow(img)

    plt.tight_layout()
    plt.show()


plot_img([train_img_dataset[0][0][0] for i in range(5)])

EPOCH = 100
epoch_range = range(EPOCH)

# building the model
model = Sequential([
    Conv2D(16, (3, 3), padding='same', activation=tf.nn.relu, input_shape=(IMG_SHAPE, IMG_SHAPE, 3)),
    MaxPool2D((2, 2), strides=2),
    Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu),
    MaxPool2D((2, 2), strides=2),
    Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu),
    MaxPool2D((2, 2), strides=2),
    Flatten(),
    Dropout(0.2),
    Dense(512, activation=tf.nn.relu),
    Dropout(0.2),
    Dense(5, activation=tf.nn.softmax)
])

# compiling the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# model summary
model.summary()

# train the model
history = model.fit_generator(train_img_dataset,
                              epochs=EPOCH,
                              steps_per_epoch=int(np.ceil(train_img_dataset.n / float(BATCH_SIZE))),
                              validation_data=validation_img_dataset,
                              validation_steps=int(np.ceil(validation_img_dataset.n / float(BATCH_SIZE))))

plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
plt.plot(epoch_range, history.history['accuracy'], label='training_accuracy')
plt.plot(epoch_range, history.history['val_accuracy'], label='validation_accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epoch_range, history.history['loss'], label='training_loss')
plt.plot(epoch_range, history.history['val_loss'], label='validation_loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.savefig('images/augmented_flowers_image_classification_using_cnn.png')
plt.show()

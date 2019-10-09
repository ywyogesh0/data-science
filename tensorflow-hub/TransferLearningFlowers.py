# transfer learning using tensorflow hub

# tensorflow hub - an online library or repository which contains high performance pre-trained machine learning models.

# transfer learning - a process in which we use an existing high performance pre-trained machine learning models and
# apply its knowledge to a new dataset or extend it to perform additional tasks.

# feature_extractor - a partial model from tensorflow hub without a final classification layer used for transfer
# learning purposes. It is called as a feature_extractor because it has already extracted the features of an object and
# only need to calculate final probability distribution for the output classes. Also, input will be directed to last
# layer of partial model which already has a number of features.

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import logging as log
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tfds.disable_progress_bar()

print(tf.__version__)

dataset, metadata = tfds.load('tf_flowers',
                              with_info=True,
                              as_supervised=True,
                              split=tfds.Split.ALL.subsplit(weighted=(70, 30)))

(train_dataset, validation_dataset) = dataset

num_train_data_examples = metadata.splits['train'].num_examples
num_of_classes = metadata.features['label'].num_classes

names_of_classes = np.array(metadata.features['label'].names)

print("train examples count is {}".format(num_train_data_examples))
print("labels count is {}".format(num_of_classes))
print("labels names are {}".format(names_of_classes))

BATCH_SIZE = 32


def prediction(img_shape, url, save_img_url):
    feature_extractor = hub.KerasLayer(url, input_shape=(img_shape, img_shape, 3))
    model = tf.keras.Sequential([
        feature_extractor,
        tf.keras.layers.Dense(5, activation=tf.nn.softmax)
    ])

    feature_extractor.trainable = False
    print(feature_extractor.call(train_data_img).shape)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    epochs = 6

    history = model.fit(train_data_batches,
                        epochs=epochs,
                        steps_per_epoch=int(np.ceil(num_train_data_examples / float(BATCH_SIZE))),
                        validation_data=validation_data_batches)

    print(history.history.keys())

    epoch_range = range(epochs)
    plt.figure(figsize=(20, 10))

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

    plt.savefig(save_img_url)
    plt.show()

    result = model.predict(train_data_img)
    print(result.shape)

    predicted_label_ids = np.argmax(result, axis=-1)
    predicted_label_names = names_of_classes[predicted_label_ids]

    plt.figure(figsize=(15, 10))

    for i in range(32):
        plt.subplot(6, 7, i + 1)
        plt.imshow(train_data_img[i])
        color = 'blue' if predicted_label_ids[i] == train_data_label[i] else 'red'
        plt.title(predicted_label_names[i].title(), color=color)
        plt.axis('off')
        plt.suptitle('Image Prediction (correct: "blue", incorrect: "red")')

    plt.show()


def resize_mobile_net_img(img, label):
    img = tf.image.resize(img, (224, 224)) / 255.0
    return img, label


def resize_inception_img(img, label):
    img = tf.image.resize(img, (299, 299)) / 255.0
    return img, label


def display_train_dataset():
    data_img, data_label = next(iter(train_data_batches.take(1)))
    return data_img.numpy(), data_label.numpy()


#######################
# mobilenet model
#######################

train_data_batches = train_dataset.repeat().shuffle(num_train_data_examples).map(resize_mobile_net_img).batch(
    BATCH_SIZE).prefetch(1)
validation_data_batches = validation_dataset.map(resize_mobile_net_img).batch(BATCH_SIZE).prefetch(1)

train_data_img, train_data_label = display_train_dataset()
print("Image Shape {}".format(train_data_img.shape))
print("Image Label count is {}".format(len(train_data_label)))

prediction(224, "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",
           "images/mobile_net_transfer_learning_flowers.png")

#######################
# mobilenet model
#######################

#######################
# inception model
#######################

train_data_batches = train_dataset.repeat().shuffle(num_train_data_examples).map(resize_inception_img).batch(
    BATCH_SIZE).prefetch(1)
validation_data_batches = validation_dataset.map(resize_inception_img).batch(BATCH_SIZE).prefetch(1)

train_data_img, train_data_label = display_train_dataset()
print("Image Shape {}".format(train_data_img.shape))
print("Image Label count is {}".format(len(train_data_label)))

prediction(299, "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4",
           "images/inception_transfer_learning_flowers.png")

#######################
# inception model
#######################

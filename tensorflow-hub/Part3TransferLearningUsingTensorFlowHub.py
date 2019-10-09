# transfer learning using tensorflow hub

# tensorflow hub - an online library or repository which contains high performance pre-trained machine learning models.

# transfer learning - a process in which we use an existing high performance pre-trained machine learning models and
# apply its knowledge to a new dataset or extend it to perform additional tasks.

# feature_extractor - a partial model from tensorflow hub without a final classification layer used for transfer
# learning purposes. It is called as a feature_extractor because it has already extracted the features of an object and
# only need to calculate final probability distribution for the output classes. Also, input will be directed to last
# layer of partial model which already has a number of features.

# importing dependencies
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import logging as log
import os

from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tfds.disable_progress_bar()

print(tf.__version__)

# ********************************
# part-3 : using tensorflow hub 'MobileNet' feature_vector module for cats vs dogs prediction (image classification)
# *********************************

dataset_splits, metadata = tfds.load('cats_vs_dogs',
                                     as_supervised=True,
                                     with_info=True,
                                     split=tfds.Split.ALL.subsplit(weighted=(80, 20)))

(train_dataset, validation_dataset) = dataset_splits

# dataset type is <class 'tensorflow.python.data.ops.dataset_ops._OptionsDataset'>

num_train_data_examples = metadata.splits['train'].num_examples
num_of_classes = metadata.features['label'].num_classes

names_of_classes = np.array(metadata.features['label'].names)

print("train examples count is {}".format(num_train_data_examples))
print("labels count is {}".format(num_of_classes))
print("labels names are {}".format(names_of_classes))

for i, train_data in enumerate(train_dataset.take(3)):
    # train_data type is <class 'tuple'>, length is 2
    # each tuple value type is <class 'tensorflow.python.framework.ops.EagerTensor'>

    print("Training Image {} Shape is {}".format(i + 1, train_data[0].shape))

for i, validation_data in enumerate(validation_dataset.take(3)):
    # validation_data type is <class 'tuple'>, length is 2
    # each tuple value type is <class 'tensorflow.python.framework.ops.EagerTensor'>

    print("Validation Image {} Shape is {}".format(i + 1, validation_data[0].shape))

BATCH_SIZE = 32
IMG_SHAPE = 224


def img_resize(img, label):
    img = tf.image.resize(img, (IMG_SHAPE, IMG_SHAPE)) / 255.0
    # img type is <class 'tensorflow.python.framework.ops.Tensor'>

    return img, label


train_data_batches = train_dataset.repeat().shuffle(num_train_data_examples).map(img_resize).batch(BATCH_SIZE).prefetch(
    1)
validation_data_batches = validation_dataset.map(img_resize).batch(BATCH_SIZE).prefetch(1)

# dataset type is <class 'tensorflow.python.data.ops.dataset_ops.PrefetchDataset'>

train_data_img, train_data_label = next(iter(train_data_batches.take(1)))
train_data_img = train_data_img.numpy()
train_data_label = train_data_label.numpy()

feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(feature_extractor_url, input_shape=(IMG_SHAPE, IMG_SHAPE, 3))

# feature_extractor type is <class 'tensorflow_hub.keras_layer.KerasLayer'>

feature_batch = feature_extractor.call(train_data_img)
print(feature_batch.shape)

# feature_batch type is <class 'tensorflow.python.framework.ops.EagerTensor'>
# feature_batch shape is (32, 1280)

# 32 input images, 1280 neurons in last layer of partial model

# freeze the variables in feature_extractor layer so as to train only the classification layer
feature_extractor.trainable = False

# attaching a classification head
model = tf.keras.Sequential([
    feature_extractor,
    tf.keras.layers.Dense(2, activation='softmax')
])

model.summary()

EPOCHS = 2

# train the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(train_data_batches,
                    epochs=EPOCHS,
                    steps_per_epoch=int(np.ceil(num_train_data_examples / float(BATCH_SIZE))),
                    validation_data=validation_data_batches)

print(history.history.keys())

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

plt.savefig('images/transfer_learning_dogs_and_cats_using_mobile_net_module.png')
plt.show()

result = model.predict(train_data_img)
# result shape is (32, 2)

result_data_label = np.argmax(result, axis=-1)
# result_data_label shape is (32,)

# verify the prediction
predicted_labels = names_of_classes[result_data_label]

plt.figure(figsize=(15, 10))
for i in range(32):
    plt.subplot(7, 5, i + 1)
    plt.subplots_adjust(hspace=0.3)
    plt.imshow(train_data_img[i])

    if result_data_label[i] == train_data_label[i]:
        color = 'blue'
    else:
        color = 'red'

    plt.title(predicted_labels[i].title(), color=color)
    plt.axis('off')

plt.suptitle('Image Predictions (correct "blue" : incorrect "red")')
plt.show()

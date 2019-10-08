# transfer learning using tensorflow hub

# tensorflow hub - an online library or repository which contains high performance pre-trained machine learning models.

# transfer learning - a process in which we use an existing high performance pre-trained machine learning models and
# apply its knowledge to a new dataset or extend it to perform additional tasks.

# importing dependencies
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import logging as log

from PIL import Image

tf.get_logger().setLevel(log.ERROR)
tfds.disable_progress_bar()

print(tf.__version__)

# ********************************
# part-2 : using tensorflow hub model for cats vs dogs prediction (image classification)
# *********************************

dataset_splits, metadata = tfds.load('cats_vs_dogs',
                                     as_supervised=True,
                                     with_info=True,
                                     split=tfds.Split.ALL.subsplit(weighted=(80, 20)))

(train_dataset, validation_dataset) = dataset_splits

# dataset type is <class 'tensorflow.python.data.ops.dataset_ops._OptionsDataset'>

train_data_examples = metadata.splits['train'].num_examples
num_of_classes = metadata.features['label'].num_classes

print("train examples count is {}".format(train_data_examples))
print("labels count is {}".format(num_of_classes))

for i, train_data in enumerate(train_dataset.take(3)):
    # train_data type is <class 'tuple'>, length is 2
    # tuple values type is <class 'tensorflow.python.framework.ops.EagerTensor'>

    print("Image {} Shape is {}".format(i + 1, train_data[0].shape))

BATCH_SIZE = 32
IMG_SHAPE = 224


def img_resize(img, label):
    img = tf.image.resize(img, (IMG_SHAPE, IMG_SHAPE)) / 255.0
    # img type is <class 'tensorflow.python.framework.ops.Tensor'>

    return img, label


train_data_batches = train_dataset.shuffle(train_data_examples // 4).map(img_resize).batch(BATCH_SIZE).prefetch(1)
validation_data_batches = validation_dataset.map(img_resize).batch(BATCH_SIZE).prefetch(1)

# dataset type is <class 'tensorflow.python.data.ops.dataset_ops.PrefetchDataset'>

model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4",
                   input_shape=(IMG_SHAPE, IMG_SHAPE, 3))
])

train_data_img, train_data_label = next(iter(train_data_batches.take(1)))
train_data_img = train_data_img.numpy()

result = model.predict(train_data_img)
# result shape is (32, 1001)

result_index_arr = np.argmax(result, axis=-1)
# result_index_arr shape is (32,)

# verify the prediction
MOBILE_NET_LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
mobile_net_labels_file_path = tf.keras.utils.get_file('ImageNetLabels.txt', origin=MOBILE_NET_LABELS_URL)

label_names_arr = np.array(open(mobile_net_labels_file_path).read().splitlines())[result_index_arr]
print(label_names_arr)

plt.figure(figsize=(10, 9))
for i in range(32):
    plt.subplot(7, 5, i + 1)
    plt.subplots_adjust(hspace=0.3)
    plt.imshow(train_data_img[i])
    plt.title(label_names_arr[i])
    plt.axis('off')

plt.suptitle('Image Predictions')
plt.show()

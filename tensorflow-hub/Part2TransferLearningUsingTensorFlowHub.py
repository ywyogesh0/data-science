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

print("train examples count is {}".format(metadata.splits['train'].num_examples))
print("labels count is {}".format(metadata.features['label'].num_classes))

for i, train_data in enumerate(train_dataset.take(3)):
    # train_data type is 'EagerTensor'
    print("Image {} Shape is {}".format(i, train_data[0].shape))

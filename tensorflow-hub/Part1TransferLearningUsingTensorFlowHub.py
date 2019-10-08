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
# part-1 : using tensorflow hub 'MobileNet' model for prediction (image classification)
# *********************************

# download the classifier
CLASSIFIER_URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
IMG_SHAPE = 224

model = tf.keras.Sequential([
    hub.KerasLayer(CLASSIFIER_URL, input_shape=(IMG_SHAPE, IMG_SHAPE, 3))
])

# type(model) - <class 'tensorflow.python.keras.engine.sequential.Sequential'>

# predict using single image
IMG_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg"

# type(grace_hopper_img_path) - <class 'str'>
grace_hopper_img_path = tf.keras.utils.get_file('grace_hopper.jpg', origin=IMG_URL)

# type(grace_hopper_img) - <class 'PIL.Image.Image'>
grace_hopper_img = Image.open(grace_hopper_img_path).resize((IMG_SHAPE, IMG_SHAPE))

# type(img_arr) - <class 'numpy.ndarray'> - (224, 224, 3)
img_arr = np.array(grace_hopper_img) / 255.0

# model prediction after adding batch dimension - (1, 224, 224, 3)

# type(result) - <class 'numpy.ndarray'> - (1, 1001)
result = model.predict(img_arr[np.newaxis, ...])
print(result.shape)

# result is a 1001 element vector of logits, rating the probability of each class for the image.

# logits - a function that maps probabilities in range - [0, 1]
# probability of 0.5 means logit of 0, < 0.5 means negative logit, > 0.5 means positive logit

result_index = np.argmax(result[0])
print(result_index)

# verify the prediction
MOBILE_NET_LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
mobile_net_labels_file_path = tf.keras.utils.get_file('ImageNetLabels.txt', origin=MOBILE_NET_LABELS_URL)

result_label = np.array(open(mobile_net_labels_file_path).read().splitlines())[result_index]

plt.imshow(grace_hopper_img)
plt.axis('off')
plt.title("Prediction: " + result_label.title())
plt.show()

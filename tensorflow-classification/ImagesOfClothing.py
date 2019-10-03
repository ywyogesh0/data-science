# classification based machine learning model
# supervised

# step 1 is importing dependencies

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import math
import logging as log

tf.get_logger().setLevel(log.ERROR)
tfds.disable_progress_bar()
print("tensorflow version is {}".format(tf.__version__))

# step 2 is importing the fashion_MNIST dataset
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
training_dataset = dataset['train']
testing_dataset = dataset['test']

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# step 3 is explore the training dataset
training_num_examples = metadata.splits['train'].num_examples
testing_num_examples = metadata.splits['test'].num_examples

print("training examples count is {}, testing examples count is {}".format(training_num_examples, testing_num_examples))


# step 4 is pre-processing the data
def normalize(image, label):
    return (tf.cast(image, tf.float32) / 255), label


training_dataset = training_dataset.map(normalize)
testing_dataset = testing_dataset.map(normalize)

# step 5 is exploring the processed dataset
plt.figure(figsize=(10, 10))
i = 0
for test_image, test_label in testing_dataset.take(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(test_image.numpy().reshape((28, 28)), cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(class_names[test_label])
    i += 1

plt.show()

# step 6 is building the model by creating and assembling layer(s) in the model definition
l0 = tf.keras.layers.Flatten(input_shape=(28, 28, 1))
l1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
l2 = tf.keras.layers.Dense(10, activation=tf.nn.softmax)

model = tf.keras.Sequential([
    l0, l1, l2
])

# step 7 is compiling the model using 'loss function' and 'optimizer function'
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(0.001),
              metrics=['accuracy'])

# step 8 is training the model
BATCH_SIZE = 32
training_dataset = training_dataset.repeat().shuffle(training_num_examples).batch(BATCH_SIZE)
testing_dataset = testing_dataset.batch(BATCH_SIZE)

history = model.fit(training_dataset, epochs=5, steps_per_epoch=math.ceil(training_num_examples / BATCH_SIZE))
print("Finished Model Training...")

# step 9 is evaluating the accuracy
test_loss, test_accuracy = model.evaluate(testing_dataset, steps=math.ceil(testing_num_examples / BATCH_SIZE))
print("test loss is {}, test accuracy is {}".format(test_loss, test_accuracy))
print("test accuracy % is {0:.2f}".format(test_accuracy * 100))

# step 10 is predicting values using trained model
# step 11 is exploring the layer weights
print("These are the layer variables (weights and biases): {}".format(l0.get_weights()))

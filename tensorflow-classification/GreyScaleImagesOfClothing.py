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

# dataset type is <class 'tensorflow.python.data.ops.dataset_ops._OptionsDataset'>

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# step 3 is explore the dataset
training_num_examples = metadata.splits['train'].num_examples
testing_num_examples = metadata.splits['test'].num_examples

print("training examples count is {}, testing examples count is {}".format(training_num_examples, testing_num_examples))


# step 4 is pre-processing the data in the dataset
def normalize(image, label):
    return (tf.cast(image, tf.float32) / 255), label


training_dataset = training_dataset.map(normalize)
testing_dataset = testing_dataset.map(normalize)

# after map, dataset type is <class 'tensorflow.python.data.ops.dataset_ops.MapDataset'>

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

# after batching, dataset type is <class 'tensorflow.python.data.ops.dataset_ops.BatchDataset'>

history = model.fit(training_dataset, epochs=5, steps_per_epoch=math.ceil(training_num_examples / BATCH_SIZE))
print("Finished Model Training...")

# step 9 is evaluating the accuracy
test_loss, test_accuracy = model.evaluate(testing_dataset, steps=math.ceil(testing_num_examples / BATCH_SIZE))
print("test loss is {}, test accuracy is {}".format(test_loss, test_accuracy))
print("test accuracy % is {:.2f}".format(test_accuracy * 100))

# step 10 is predict image class and explore
for test_images, test_labels in testing_dataset.take(1):
    # before, test_images and test_labels type is <class 'tensorflow.python.framework.ops.EagerTensor'>

    test_images = test_images.numpy()
    test_labels = test_labels.numpy()

    # after, test_images and test_labels type is <class 'numpy.ndarray'>

    predictions = model.predict(test_images)

print("test_images shape is {}".format(test_images.shape))
print("test_labels shape is {}".format(test_labels.shape))
print("predictions shape is {}".format(predictions.shape))


def plot_img(x, true_images, true_labels, model_predictions):
    true_image = true_images[x]
    true_label = true_labels[x]
    prediction = model_predictions[x]

    predicted_value = np.max(prediction)
    predicted_label = np.argmax(prediction)

    predicted_class = class_names[predicted_label]
    true_class = class_names[true_label]

    plt.xticks([])
    plt.yticks([])

    if true_label == predicted_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(predicted_class, 100 * predicted_value, true_class),
               color=color)

    plt.imshow(true_image.reshape((28, 28)), cmap=plt.cm.binary)


def plot_label_bar(x, true_labels, model_predictions):
    true_label = true_labels[x]
    prediction = model_predictions[x]

    predicted_label = np.argmax(prediction)
    plt.ylim([0, 1])

    plt.xticks(range(10))
    plt.yticks([])

    bar_plt = plt.bar(range(10), prediction, color='#777777')
    bar_plt[predicted_label].set_color('red')
    bar_plt[true_label].set_color('blue')


no_of_rows = 5
no_of_columns = 3
no_of_images = no_of_rows * no_of_columns

plt.figure(figsize=(2 * 2 * no_of_columns, 2 * no_of_rows))

for i in range(no_of_images):
    plt.subplot(no_of_rows, 2 * no_of_columns, 2 * i + 1)
    plot_img(i, test_images, test_labels, predictions)

    plt.subplot(no_of_rows, 2 * no_of_columns, 2 * i + 2)
    plot_label_bar(i, test_labels, predictions)

plt.show()

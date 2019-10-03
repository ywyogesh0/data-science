# regression based machine learning model
# supervised
# algorithm is 'f = 1.8 * c + 32'

# step 1 is importing dependencies

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import logging as log

tf.get_logger().setLevel(log.ERROR)
print("tensorflow version is {}".format(tf.__version__))

# step 2 is setting up the training dataset

c_dataset = np.array([-35, -23, -12, 0, 10, 25, 54], dtype=float)
f_dataset = np.array([-31, -9.4, 10.4, 32, 50, 77, 129.2], dtype=float)

for i, c in enumerate(c_dataset):
    print("Celsius is {}, Fahrenheit is {}".format(c, f_dataset[i]))

# step 3 is building the model by creating and assembling layer(s) in the model definition
l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([
    l0
])

# step 4 is compiling the model using 'loss function' and 'optimizer function'
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))

# step 5 is training the model
history = model.fit(c_dataset, f_dataset, epochs=5000, verbose=False)
print("Finished Model Training...")

# step 6 is exploring the training statistics using matplotlib.pyplot
plt.xlabel('epoch number')
plt.ylabel('loss magnitude')
plt.plot(history.history['loss'])
plt.show()

# step 7 is predicting values using trained model
c = 53.0
print("Celsius is {}, Fahrenheit is {}".format(c, model.predict([c])))

# step 8 is exploring the layer weights
print("These are the layer variables (weights and biases): {}".format(l0.get_weights()))

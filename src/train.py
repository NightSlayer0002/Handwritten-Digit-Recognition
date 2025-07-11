import os
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist 
(x_train, y_train), (x_test, y_test) = mnist.load_data()    #training set

#dev set

#normalizing and reshaping
x_train = x_train.reshape(-1, 28, 28, 1)/255.0      #reshape for proper model training
x_test = x_test.reshape(-1, 28, 28, 1)/255.0

#initialize and add layers (CNN for more accuracy)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

#compiling
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#fit(training the model)
model.fit(x_train, y_train, epochs = 5, validation_data = (x_test, y_test))      #default batch size: 32

#saving the model
model.save("models/handwritten.keras")
print("Model saved to handwritten.keras")

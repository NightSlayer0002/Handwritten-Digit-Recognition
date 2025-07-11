import os
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28,28))
    img = np.invert(img)
    img = img/255.0
    return img.reshape(1, 28, 28, 1)

def predict_digit(path):
    model = tf.keras.models.load_model("models/handwritten.keras")
    img = preprocess_image(path)
    prediction = model.predict(img)
    return np.argmax(prediction)

def predict_and_plot(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    digit = predict_digit(path)
    plt.imshow(img, cmap = 'gray')
    plt.title(f"Predicted digit: {digit}")
    plt.axis('off')
    plt.show()
    
if __name__ == "__main__":
    for path in sorted(glob.glob("digits/digit*.png")):
        predict_and_plot(path)                         
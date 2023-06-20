import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def categorical(arr):
    res = []
    
    for i in arr:
        l = np.zeros(10)
        l[i] = 1
        res.append(l)
    
    return np.array(res)
    
if __name__ == "__main__":
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    train_labels_c = categorical(train_labels)
    test_labels_c = categorical(test_labels)
    
    train_images = train_images / 255.
    train_images = train_images.reshape(60000, 28, 28, 1)
    test_images = test_images / 255.
    test_images = test_images.reshape(10000, 28, 28, 1)
    
    model = keras.models.Sequential([
        keras.layers.Conv2D(input_shape = (28,28,1),
                            kernel_size = (2,2),
                            filters = 4,
                            padding = "same",
                            activation = "relu"),
        keras.layers.MaxPooling2D(pool_size = (2,2),
                                  strides = 2),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation = "relu"),
        keras.layers.Dense(10, activation = "softmax"),
        ])
    
    adam = keras.optimizers.Adam(learning_rate = 0.001)
    model.compile(optimizer = adam, loss = "categorical_crossentropy", metrics = ["accuracy"])
    history = model.fit(train_images, train_labels_c, epochs = 1, batch_size = 64)
    
    res = model.evaluate(test_images, test_labels_c)
    print(res[0], res[1])
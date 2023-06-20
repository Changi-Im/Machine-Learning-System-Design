import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def categorical(arr):
    res = []
    
    for x in arr:
        l = np.zeros(10)
        l[x] = 1
        res.append(l)
    
    return np.array(res)

if __name__ == "__main__":
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    
    train_images = train_images / 255.
    test_images = test_images / 255.

    train_labels_c = categorical(train_labels)
    test_labels_c = categorical(test_labels)

    model = keras.models.Sequential([
        keras.layers.Conv2D(input_shape = (28, 28, 1),
                            kernel_size = (3, 3),
                            filters = 32,
                            padding = "same",
                            activation = "relu"),
        keras.layers.MaxPooling2D(pool_size = (2,2)),
        keras.layers.Conv2D(kernel_size = (3, 3),
                            filters = 64,
                            padding = "same",
                            activation = "relu"),
        keras.layers.MaxPooling2D(pool_size = (2,2)),
        keras.layers.Conv2D(kernel_size = (3, 3),
                            filters = 32,
                            padding = "same",
                            activation = "relu"),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation = "relu"),
        keras.layers.Dense(32, activation = "relu"),
        keras.layers.Dense(10, activation = "softmax"),
        ])
    
    model.summary()
    
    model.compile(optimizer = 'adam', loss = "categorical_crossentropy", metrics = ['accuracy'])
    history = model.fit(train_images, train_labels_c, epochs = 1)
    
    res = model.evaluate(test_images, test_labels_c)
    print("테스트 정확도:", res[-1])
    
    label = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3:"Dress", 4:"Dress", 5:"Sandal", 6:"Shirt", 7:"Sneaker", 8:"Bag", 9:"Ankle boot"}
    ax = plt.figure(figsize=(10, 10))
    test_x = test_images[:25, :, :]
    pred = model.predict(test_x)
    
    for i in range(25):
        ax1 = ax.add_subplot(5,5,i+1)
        img = test_x[i] * 255
        ax1.imshow(img)
        ax1.set_title(label[np.argmax(pred[i])])
        ax1.axis(False)
    
    
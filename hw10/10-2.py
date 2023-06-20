import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == "__main__":
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labes) = fashion_mnist.load_data()
    
    r = np.random.choice(range(0,60000), int(60000*0.75), replace = False)
    
    X_train = []
    y_train = []
    
    X_test = np.delete(train_images, r, axis = 0)
    y_test = np.delete(train_labels, r)
    
    X_test = X_test / 255.
    
    for _ in r:
        X_train.append(train_images[_])
        y_train.append(train_labels[_])
        
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    X_train = X_train / 255.
    
    model = keras.models.Sequential([
            keras.layers.Flatten(input_shape = (28,28)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(10, activation='softmax'),
        ])
    
    model.compile(optimizer = 'adam', loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
    history = model.fit(X_train, y_train, batch_size = 64, epochs = 10, validation_data=(X_test, y_test))
    
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    
    xx = np.arange(0, 10, 1)
    plt.plot(xx, loss, "b", label = "train")
    plt.plot(xx, val_loss, "r--", label = "validation")
    plt.legend(loc = "upper right")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.pause(1e-10)
    
    plt.plot(xx, acc, "b", label = "train")
    plt.plot(xx, val_acc, "r--", label = "validation")
    plt.legend(loc = "upper left")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.pause(1e-10)
    
    
    ans = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress",
           4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"}
    
    X = test_images[:25]
    y = test_labes[:25]
    
    X = X / 255.
    y_hat = model.predict(X)
    
    ax = plt.figure(figsize = (10,10))
    
    for i in range(25):
        ax1  = ax.add_subplot(5, 5, i+1)
        ax1.imshow(X[i]*255)
        ax1.set_title("{}".format(ans[np.argmax(y_hat[i])]))
        ax1.axis("off")
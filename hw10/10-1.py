import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

if __name__ == '__main__':
    df = load_iris()
    
    X = df['data']
    y = df['target']
    
    
    r1 = np.random.choice(range(0,50), 35, replace=False)
    r2 = np.random.choice(range(50,100), 35, replace=False)
    r3 = np.random.choice(range(100,150), 35, replace=False)
    
    r = np.concatenate((r1,r2,r3))
    
    X_test = X.copy()
    y_test = y.copy()
    
    X_test = np.delete(X_test, r, axis=0)
    y_test = np.delete(y_test, r)
    
    X_train = []
    y_train = []
    
    for i in range(105):   
        X_train.append(X[r[i]])
        y_train.append(y[r[i]])
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    model = keras.models.Sequential([
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(10, activation="relu"),
            keras.layers.Dense(3, activation="softmax"),
        ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, batch_size = 5, epochs = 30)
    
    loss = history.history['loss']
    acc = history.history['acc']
    
    res = model.evaluate(X_test, y_test)
    
    xx = np.arange(0, 30, 1)
    plt.plot(xx, acc, 'r', label='accuracy')
    plt.plot(xx, loss, 'b', label='loss value')
    plt.legend(loc="lower left")
    
        
        
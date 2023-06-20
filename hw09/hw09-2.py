# hw 09

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras

if __name__ == "__main__":
    df = pd.read_csv("./nonlinear.csv")
    
    x = df[['x']].to_numpy()
    y = df['y'].to_numpy()
    
    model = keras.models.Sequential([
            keras.layers.Dense(32, activation='tanh'),
            keras.layers.Dense(16, activation='tanh'),
            keras.layers.Dense(8, activation='tanh'),
            keras.layers.Dense(4, activation='tanh'),
            keras.layers.Dense(1, activation='tanh'),
        ])
    
    optimizer = keras.optimizers.SGD(learning_rate = 0.1)
    model.compile(optimizer=optimizer, loss = 'mse')
    model.fit(x, y, epochs = 100)
    
    domain = np.linspace(0,1,100).reshape(-1,1)
    y_hat = model.predict(domain)
    
    plt.plot(x,y,'o')
    plt.plot(domain,y_hat,'o')
    
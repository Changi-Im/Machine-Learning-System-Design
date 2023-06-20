import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def mse(arr1, arr2):
    sd = (arr1 - arr2)**2
    return sd.sum()/len(sd)

def sequence_gen(size, seq_len):
    seq_X = np.empty(shape = (size, seq_len, 1))
    Y = np.empty(shape=(size,))
    x = np.linspace(0, 10, 51)
    
    for i in range(size):
        t = np.random.randint(0, 200)
        c = np.sin(t + x[:50])
        seq_X[i] = c[:, np.newaxis]
        Y[i] = np.sin(t + x[50])
    
    return seq_X, Y

if __name__ == "__main__":
    n, seq_len = 100, 50
    seq_X, Y = sequence_gen(n, seq_len)
    
    train_seq_X, train_Y = seq_X[:80], Y[:80]
    test_seq_X, test_Y = seq_X[80:], Y[80:]
    
    xx = np.linspace(0, 10, 51)
    for i in range(0,4):
        plt.plot(xx[:50], train_seq_X[i], "o")
        plt.plot(xx[50], train_Y[i], "o")
        plt.pause(1e-8)
        
    n_units = 10
    x = np.arange(0, 50, 1)
    
    simpleRNN_model = tf.keras.Sequential([
            tf.keras.layers.SimpleRNN(units = n_units, return_sequences=False,
                                      input_shape = [seq_len, 1]),
            tf.keras.layers.Dense(1)
        ])
    simpleRNN_model.compile(optimizer='adam', loss = 'mse', metrics = ["accuracy"])
    history = simpleRNN_model.fit(train_seq_X, train_Y, epochs = 50)
    test_pred = simpleRNN_model.predict(test_seq_X)
    train_pred = simpleRNN_model.predict(train_seq_X)
    
    
    plt.plot(x, history.history["loss"])
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.pause(1e-8)
    
    plt.plot(train_Y, train_pred, "o")
    plt.plot(test_Y, test_pred, "o")
    plt.pause(1e-8)
    
    plt.plot(train_Y, "black")
    plt.plot(train_pred, "r")
    plt.text(0, 0.75, "{:6f}".format(mse(train_Y, train_pred.squeeze())))
    plt.pause(1e-8)
    
    plt.plot(test_Y, "black")
    plt.plot(test_pred, "r")
    plt.text(0, 0.75, "{:6f}".format(mse(test_Y, test_pred.squeeze())))
    plt.pause(1e-8)
    
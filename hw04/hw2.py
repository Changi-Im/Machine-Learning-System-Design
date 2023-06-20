import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


def MSE_loss(y_, y):
    N = len(y_)
    s = 0
    for i in range(0, N):
        s = s + (y_[i]-y[i])**2
    
    return s / N

def function(x, w, b):
    return (w*x + b)

def gradient(x, y, w, b, lr, epoch):
    i = 0 
    while i != epoch:
        error = function(x, w, b) - y
        w = w - lr*2*(error*x).sum()/len(x)
        b = b - lr*2*error.sum()/len(x)
        
        i = i+1
    
    return w, b
        

if __name__ == "__main__":
    data = './01_dataset.csv'
    df = pd.read_csv(data)
    w = 0
    b = 1
    x = df["X"]
    y = df["Y"]
    
    plt.plot(x, function(x, w, b), 'r')
    plt.plot(x, y, 'o')
    print("mse:{0}".format(MSE_loss(function(x,w,b), y)))
    
    lr = 1e-5
    epoch = 200

    
    w, b = gradient(x, y, w, b, lr, epoch)
    
    plt.plot(x, function(x, w, b), 'r')
    plt.plot(x, y, 'o')
    print("mse:{0}".format(MSE_loss(function(x,w,b), y)))
    
    x = x.to_numpy()
    y = y.to_numpy()
    
    x = x.reshape(1, len(x))
    y = y.reshape(1, len(y))
    
    regr =linear_model.LinearRegr
    
    
    
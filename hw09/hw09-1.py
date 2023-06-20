# hw 09

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def sig(x):
    return 1/(1+np.exp(-x))

def train(X, T, U, V, W, lr, epoch):
    e = []
    for _ in range(epoch):
        loss = 0
        for i in range(len(X)):
            H = U.T.dot(X[i])
            
            h_sig = sig(H)
            h_sig_d = sig(H)*(1-sig(H))
            
            Y = V.T.dot(h_sig)
            
            y_sig = sig(Y)
            y_sig_d = sig(Y)*(1-sig(Y))
            
            Z = W.T.dot(y_sig)
            
            z_sig = sig(Z)
            z_sig_d = sig(Z)*(1-sig(Z))
                
            E = z_sig-T[i]
            
            z_delta = z_sig_d*E
            
            E_y = W.dot(z_delta)
            W -= lr*np.outer(y_sig, z_delta)
            
            y_delta = y_sig_d*E_y
            
            E_h = V.dot(y_delta)
            V -= lr*np.outer(h_sig, y_delta)
            
            h_delta = h_sig_d*E_h
            
            U -= lr*np.outer(X[i], h_delta)
            
            loss += E**2
        
        e.append(loss)
    
    return U, V, W, e

def test(X, U, V, W):
    pred = []
    for x in X:
        H = U.T.dot(x)
        h_sig = sig(H)
        Y = V.T.dot(h_sig)
        y_sig = sig(Y)
        Z = W.T.dot(y_sig)
        z_sig = sig(Z)
        pred.append(z_sig)
    return pred
            
    
if __name__ == "__main__":
    df = pd.read_csv("./nonlinear.csv")
    
    X = df['x'].to_numpy()
    Y = df['y'].to_numpy()
    
    U = np.random.randn(1,6)
    V = np.random.randn(6,4)
    W = np.random.randn(4,1)

    lr = 1.0
    epoch = 100
                 
    U, V, W, e = train(X, Y, U, V, W, lr, epoch)
    
    xx = np.linspace(0, 1, 100)
    
    pred = test(xx, U, V, W)
    
    plt.plot(X, Y, "o")
    plt.plot(xx, np.array(pred).reshape(-1), "ro")
    
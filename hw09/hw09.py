# hw 09

import numpy as np
import matplotlib.pyplot as plt

def sig(x):
    return 1/(1+np.exp(-x))

def train(X, U, W, lr, epoch):
    e = []
    for _ in range(epoch):
        loss = 0
        for x in X:
            H = U.T.dot(x)
            
            h_sig = sig(H)
            h_sig_d = sig(H)*(1-sig(H))
            
            Y = W.T.dot(h_sig)
            
            y_sig = sig(Y)
            y_sig_d = sig(Y)*(1-sig(Y))
            
            if x[0]^x[1]:
                t = [1,0]
            else:
                t = [0,1]
                
            E = y_sig-t
            
            y_delta = y_sig_d*E
            
            E_h = W.dot(y_delta)            
            W -= lr*np.outer(h_sig, y_delta)
            
            h_delta = h_sig_d*E_h
            
            U -= lr*np.outer(x, h_delta)
            
            loss += E**2
        
        e.append(loss)
    
    return U, W, e

def test(X, U, W):
    H = U.T.dot(X)
    h_sig = sig(H)
    Y = W.T.dot(h_sig)
    y_sig = sig(Y)
    
    return y_sig
            
    
if __name__ == "__main__":
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    U = np.array([[0.2,0.2,0.3], [0.3,0.1,0.2]])
    W = np.array([[0.3,0.2], [0.1,0.4], [0.2,0.3]])
    
    lr = 1.0
    epoch = 1000
                 
    U, W, e = train(X, U, W, lr, epoch)
    
    xx = np.arange(0, epoch, 1)
    
    for x in X:
        y = test(x, U, W)
        print("x1={}, x2={}, y1={}, y2={}".format(x[0],x[1],y[0],y[1]))
    
    plt.plot(xx, e)
    plt.xlabel("epoch")
    plt.ylabel("error")
    
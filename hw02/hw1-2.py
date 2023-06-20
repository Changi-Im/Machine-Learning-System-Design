import numpy as np
import matplotlib.pyplot as plt

def gradient(x, lr, epoch):
    i = 0
    points = []
    while i != epoch:
        df = 2*x
        x = x - lr*df
        points.append(x)
        
        i += 1
        
    return np.array(points)

def gradient_(x, lr, epoch):
    i = 0
    points = []
    while i != epoch:
        df = np.sin(x**2) + 2*(x**2)*np.cos(x**2)
        x = x - lr*df
        points.append(x)
        
        i += 1
        
    return np.array(points)

if __name__ == "__main__":
    x = np.arange(-10., 11., 1)
    f = x**2
    lr = 0.1
    epoch = 10
    
    points = gradient(10, lr, epoch)
    plt.plot(x, f)
    plt.plot(points, points**2,'bo')
    
    
    x = np.arange(-3., 3.01, 0.01)
    f = x*np.sin(x**2)
    lr = 0.01    
    points = gradient_(1.6, lr, epoch)
    
    plt.plot(x, f)
    plt.plot(points, points*np.sin(points**2), 'bo')
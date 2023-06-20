import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def function(x, theta):
    return x*theta[1] + theta[0] 

def gradient(x, y, theta, lr, epoch):
    for i in range(0, epoch):
        error = function(x, theta) - y
        
        theta[0] = theta[0] - 2*lr*(error*x).sum()
        theta[1] = theta[1] - 2*lr*error.sum()
        
    return theta

def norm(x):
    return (x - x.min()) / (x.max() - x.min())

def sta(x):
    return (x-x.mean()) / x.std()
    
if __name__ == "__main__":
    X = np.random.randn(500)
    N = np.random.randn(500)
    y = -2*X + 1 + 1.2*N
    
    plt.plot(X, y, 'o')
     
    x = X[:,np.newaxis]
    x = np.c_[np.ones((500, 1)), x]
    
    theta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)

    print(theta[0], theta[1])
    
    plt.plot(X, function(X, theta))
    plt.pause(0.1)
    plt.close()
    
    
    data2 = "./02_dataset.csv"
    data3 = "./03_dataset.csv"
    
    df2 = pd.read_csv(data2)
    
    y = df2["X"].to_numpy()
    x = np.arange(0, 1000)
    
    plt.plot(x, y)
    plt.pause(0.01)
    plt.close()
    plt.hist(y, 100)
    plt.pause(0.01)
    y_n = norm(y)
    
    plt.plot(x, y_n)
    plt.pause(0.01)
    
    y_s = sta(y)
    print(y_s.mean())
    print(y_s.std())
    plt.hist(y_s, bins= 100)
    plt.pause(0.01)
    
    X = y_n
    N = np.random.randn(1000)
    
    y = -2*X + 1 + 1.2*N

    X = X.reshape(-1,1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    y_hat_train = lin_model.predict(X_train)
    y_hat_tst = lin_model.predict(X_test)
    plt.plot(y_train, y_hat_train, 'ro')
    plt.plot(y_test, y_hat_tst, 'o')
    plt.plot((-4,4), (-1,1))
    plt.pause(0.01)

    X = y_s
    y = -2*X + 1 + 1.2*N
    
    X = X.reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    y_hat_train = lin_model.predict(X_train)
    y_hat_tst = lin_model.predict(X_test)
    plt.plot(y_train, y_hat_train, 'ro')
    plt.plot(y_test, y_hat_tst, 'o')
    plt.plot((-4,4), (-1,1))
    plt.pause(0.01)
    
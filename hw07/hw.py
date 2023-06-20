import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    data = "./nonlinear.csv"
    
    df = pd.read_csv(data)
    y = df["y"].to_numpy()
    x = df[["x"]].to_numpy()
    
    poly_feature3 = PolynomialFeatures(degree = 3)
    X = poly_feature3.fit_transform(x)
    lin_model = LinearRegression()
    lin_model.fit(X, y)
    
    
    xx = np.linspace(0, 1, 100).reshape(-1,1)
    xx3 = poly_feature3.fit_transform(xx)
    y_hat = lin_model.predict(xx3)
    plt.plot(x, y, 'o')
    plt.plot(xx, y_hat, 'r')
    
    
    
    
    
    
    
    
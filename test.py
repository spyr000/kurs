from sklearn.linear_model import LinearRegression
import numpy as np

if __name__ == '__main__':
    regr = LinearRegression()
    x = np.linspace(0,10,1000)
    y1 = x**2
    y2 = (x+1)**2
    print(y2-y1)
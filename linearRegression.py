# The following app calculates the constant variables for
# a regression line given a data set and plots the test,
# training and regression line.

import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

def estimate_coef(x, y): 
    # number of observations/points 
    n = np.size(x) 
  
    # mean of x and y vector 
    m_x, m_y = np.mean(x), np.mean(y) 
  
    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum(y*x) - n*m_y*m_x 
    SS_xx = np.sum(x*x) - n*m_x*m_x 
  
    # calculating regression coefficients 
    b_1 = SS_xy / SS_xx 
    b_0 = m_y - b_1*m_x 
  
    return(b_0, b_1)

def plot_regression_line(x, y, b): 
  
    # predicted response vector 
    y_pred = b[0] + b[1]*x 
  
    # plotting the regression line 
    plt.plot(x, y_pred, color = "blue", label = "Regression Line") 
  
    # putting labels 
    plt.xlabel('x') 
    plt.ylabel('y')

def main(): 
    # observations 
    d = load_diabetes()
    d_X = d.data[:, np.newaxis, 2]
    dx_train = d_X[:-20].squeeze()
    dy_train = d.target[:-20]
    dx_test = d_X[-20:].squeeze()
    dy_test = d.target[-20:]

    print(dx_train)
  
    # estimating coefficients 
    b = estimate_coef(dx_train, dy_train) 
    print("Estimated coefficients:\nb_0 = {}  \nb_1 = {}".format(b[0], b[1])) 
  
    # plotting regression line 
    plot_regression_line(dx_test, dy_test, b)
    
    # plotting the actual and test points as scatter plot 
    plt.scatter(dx_train, dy_train, color = "red", 
               marker = "o", s = 30, label = "Train Data")
    plt.scatter(dx_test, dy_test, color = "green", 
               marker = "o", s = 30, label = "Test Data")
    
    # function to show plot
    plt.legend()
    plt.show()
  
if __name__ == "__main__": 
    main() 
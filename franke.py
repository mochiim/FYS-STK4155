from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from random import random, seed
from sklearn.linear_model import LinearRegression


def FrankeFunction(x,y, noise = False):
    if noise:
        epsilon = np.random.normal(0, .1, size = x.shape)
    else:
        epsilon = 0
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + epsilon

def R2(y_data, y_model):
    '''
    Calculate the R1 (coefficient of determinatoin) score for a regression model.

    Input:
    y_data (ndarray): The true value of the dependent variable.
    y_mode (ndarray): The predicted value of the dependent variable generated by the regression model.

    Output:
    R2 (float): The R2 score ranging from 0 to 1.
    '''
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)

def MSE(y_data,y_model):
    '''
    Calculate the Mean Squared Error score for a regression model.

    Input:
    y_data (ndarray): The true value of the dependent variable.
    y_mode (ndarray): The predicted value of the dependent variable generated by the regression model.

    Output:
    MSE (float): The MSE score ranging from 0 to 1.
    '''
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def plot_3d(x, y, z):
    fig = plt.figure(figsize = (8, 6))  
    ax = fig.gca(projection = '3d')
    z = FrankeFunction(x, y, noise = True)
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(-0.10, 1.40) 
    ax.zaxis.set_major_locator(LinearLocator(10)) 
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)

#def OLS():
    
maxdegree = 5 # polynomial degree
polynomial_orders = [i + 1 for i in range(maxdegree)]

r2_test = np.zeros(len(polynomial_orders))
r2_train = np.zeros(len(polynomial_orders))

#for p in maxdegree:
    # make data
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    X, Y = np.meshgrid(x,y)                                 
    z = FrankeFunction(X, Y, noise = True)


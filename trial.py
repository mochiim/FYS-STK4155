import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, LinearRegression, Lasso, Ridge
from sklearn.model_selection import cross_validate

np.random.seed(4155)

### a)

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

# Generate the data
nrow = 100
ncol = 200
ax_row = np.random.uniform(0, 1, size=nrow)
ax_col = np.random.uniform(0, 1, size=ncol)

ind_sort_row = np.argsort(ax_row)
ind_sort_col = np.argsort(ax_col)

ax_row_sorted = ax_row[ind_sort_row]
ax_col_sorted = ax_col[ind_sort_col]

colmat, rowmat = np.meshgrid(ax_col_sorted, ax_row_sorted)

noise_str = .0
noise = np.random.randn(nrow, ncol)

z = FrankeFunction(rowmat, colmat) + noise_str * noise

row_arr = rowmat.ravel()
col_arr = colmat.ravel()
z_arr = z.ravel()

# Generate the design matrix
p = 10
poly = PolynomialFeatures(degree = p)
X = poly.fit_transform(np.c_[row_arr, col_arr])

## Perform OLS
linreg = LinearRegression()
linreg.fit(X, z_arr)

zpred = linreg.predict(X)
zplot = zpred.reshape(nrow, ncol)

# Plot the reuslting fit beside the original surface
fig = plt.figure()

ax = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(colmat, rowmat, z, cmap=cm.viridis, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Franke')

ax = fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(colmat, rowmat, zplot, cmap=cm.viridis, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Fitted Franke')

plt.show()
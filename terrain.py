import numpy as np
from imageio import imread
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

# Load the terrain
terrain = imread('SRTM_data_Norway_1.tif')

N = 1000
m = 5 # polynomial orderx
terrain = terrain[:N,:N]
# Creates mesh of image pixels
x = np.linspace(0, 1, np.shape(terrain)[0])
y = np.linspace(0, 1, np.shape(terrain)[1])
x_mesh, y_mesh = np.meshgrid(x,y)

z = terrain

fig = plt.figure(figsize = (8, 6)) 
ax = fig.gca(projection='3d') 
surf = ax.plot_surface(x_mesh, y_mesh, z, cmap = cm.coolwarm, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig('terrain_3D.png')
plt.show()

import numpy as np
import imageio 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import tifffile as tiff
import scipy as sc

# Load the terrain
terrain1 = imageio.imread('SRTM_data_Norway_1.tif')


# Show the terrain
fig, ax = plt.subplots(figsize = (5, 6))
ax.set_title('Terrain over Norway')
image = plt.imshow(terrain1, cmap = 'seismic')
ax.set_xlabel('X')
ax.set_ylabel('Y')
fig.colorbar(image)
fig.tight_layout()
#plt.savefig('terrain1.png')
plt.show()

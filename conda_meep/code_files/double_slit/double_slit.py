import double_slit_meep_3d as ds
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

wavelength = 0.589
x_dist = 20
y_dist = 30
z_dist = 20
dists = [x_dist, y_dist,z_dist]

pml_depth = 0 #min(dists)/10

plate_pos_x = 0
plate_pos_y = 0
plate_pos_z = 0
plate_pos = [plate_pos_x,plate_pos_y,plate_pos_z]

radius = 1
plate_thickness = 0.5
eps = 1e6 + 1e6j


eps_data, ez_data, ey_data, ex_data = ds.run(dists, plate_pos,plate_thickness, radius, wavelength,eps,pml_depth)

cmap_alpha = LinearSegmentedColormap.from_list('custom_alpha', [[1, 1, 1, 0], [1, 1, 1, 1]])
cmap_blue = LinearSegmentedColormap.from_list('custom_blue', [[0, 0, 0], [0, 0.66, 1], [1, 1, 1]])
    
plt.figure()
plt.imshow(eps_data[int(np.shape(eps_data)[0]/2),:,:].transpose())
plt.show()
    
plt.figure()
plt.imshow(np.abs(ez_data[:,:,int(np.shape(eps_data)[2]/2)].transpose())**2)
plt.imshow(eps_data[:,:,int(np.shape(eps_data)[2]/2)].transpose(),cmap=cmap_alpha)
plt.show()

plt.figure()
plt.imshow(np.abs(ez_data[-1,:,:].transpose())**2,cmap=cmap_blue)
plt.show()

#plt.figure()
#plt.imshow(eps_data.transpose())
#plt.show()
#    
#plt.figure()
#plt.imshow(np.abs(ez_data.transpose())**2)
#plt.imshow(eps_data.transpose(),cmap=cmap_alpha)
#plt.show()



from main_funcs import fdtd_meep_3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

wavelength = 0.589
x_dist = 3*wavelength
y_dist = 5*wavelength
z_dist = 5*wavelength
dists = [x_dist, y_dist,z_dist]

pml_depth = min(dists)/10

plate_pos_x = -x_dist/2+pml_depth
plate_pos_y = 0
plate_pos_z = 0
plate_pos = [plate_pos_x,plate_pos_y,plate_pos_z]

radius = 0.5
plate_thickness = 0.3
beam_width = 1
eps = 1.5 + 1e6j


eps_data, ez_data, ey_data, ex_data = fdtd_meep_3d.run(dists, plate_pos,plate_thickness, radius, wavelength,beam_width,eps,pml_depth)

cmap_alpha = LinearSegmentedColormap.from_list('custom_alpha', [[1, 1, 1, 0], [1, 1, 1, 1]])
cmap_blue = LinearSegmentedColormap.from_list('custom_blue', [[0, 0, 0], [0, 0.66, 1], [1, 1, 1]])
    
plt.figure()
plt.imshow(np.abs(ez_data[:,:,int(np.shape(eps_data)[2]/2)].transpose())**2,cmap=cmap_blue)
plt.imshow(eps_data[:,:,int(np.shape(eps_data)[1]/2)].transpose(),cmap=cmap_alpha)
plt.show()

plt.figure()
plt.imshow(np.angle(ez_data[:,:,int(np.shape(eps_data)[2]/2)].transpose()),cmap='jet')
plt.imshow(eps_data[:,:,int(np.shape(eps_data)[2]/2)].transpose(),cmap=cmap_alpha)
plt.show()

plt.figure()
plt.imshow(np.abs(ez_data[-1,:,:].transpose())**2,cmap=cmap_blue)
plt.show()

plt.figure()
plt.imshow(np.angle(ez_data[-1,:,:].transpose()),cmap='jet')
plt.show()


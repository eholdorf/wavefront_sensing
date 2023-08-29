import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# https://humaticlabs.com/blog/meep-double-slit/
def run(dists,plate_pos,plate_thickness,radius, wavelength, beam_width, eps,
        plm_depth):
    x_dist, y_dist = dists
    # set up a 3D grid (x,y,z) with direction of propagation in x
    cell = mp.Vector3(x_dist,y_dist)
    # set the position of the plate, and other variables
    plate_pos_x, plate_pos_y = plate_pos
    plate_thickness = plate_thickness
    aperature = radius

    # set parameters for the plate and beam
    freq = 1/wavelength # in meep units
    beam_width = beam_width
    real_eps = np.real(eps)
    complex_eps = np.imag(eps)

    plate = [mp.Block(
        material = mp.Medium(epsilon = real_eps, D_conductivity=2*np.pi*freq*complex_eps/real_eps),
        size = mp.Vector3(plate_thickness, mp.inf,mp.inf),
        center = mp.Vector3(plate_pos_x,plate_pos_y)
    ),
    mp.Cylinder(
        material = mp.Medium(epsilon = 1),
        radius = aperature,
        height = plate_thickness,
        center = mp.Vector3(plate_pos_x,plate_pos_y),
        axis = mp.Vector3(1,0,0)

    )]

    # append to make the total geometry
    geometry = []
    geometry.extend(plate)

    # resolution is the number of pixels per micron
    resolution = max(dists)//wavelength*20

    # set up perfectly matched layers on boundary, of 1 micron thick
    pml_layers = [mp.PML(plm_depth)]

    
    # set up the sources
    sources = [mp.Source(src = mp.ContinuousSource(frequency=freq, 
                                            is_integrated = True),
                        component=mp.Ez,
                        center=mp.Vector3(-x_dist/2+plm_depth,0,0),
                        size = mp.Vector3(y = cell[1], z = cell[2]),
                        amplitude = 1),
                        mp.Source(src = mp.ContinuousSource(frequency=freq, 
                                            is_integrated = True),
                        component=mp.Ey,
                        center=mp.Vector3(-x_dist/2+plm_depth,0,0),
                        size = mp.Vector3(y = cell[1], z = cell[2]),
                        amplitude = 1)]

    # set up the simulation
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution,
                        force_complex_fields = True)

    sim.run(until=cell[0]+100) 
    eps_data = sim.get_array(center = mp.Vector3(),size=cell, component=mp.Dielectric)
    ez_data = sim.get_array(center= mp.Vector3(),size=cell,component = mp.Ez)
    
    
    cmap_alpha = LinearSegmentedColormap.from_list(
    'custom_alpha', [[1, 1, 1, 0], [1, 1, 1, 1]])
    cmap_blue = LinearSegmentedColormap.from_list(
    'custom_blue', [[0, 0, 0], [0, 0.66, 1], [1, 1, 1]])
    plt.figure()
    plt.imshow(np.abs(ez_data[:,:].transpose())**2)
    plt.imshow(eps_data[:,:].transpose(),cmap=cmap_alpha)
    plt.show()
    
    plt.figure()
    plt.imshow(np.angle(ez_data[:,:].transpose()),cmap='jet')
    plt.imshow(eps_data[:,:].transpose(),cmap=cmap_alpha)
    plt.show()
    


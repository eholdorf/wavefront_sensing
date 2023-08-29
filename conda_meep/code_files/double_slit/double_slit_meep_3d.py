import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def plane_wave(x,wavelength=0.589):
    return np.exp(1j*2*np.pi/wavelength*x)

def gauss_beam(x,width=0.1):
    return np.exp(-x**2/(2*(width/2)**2))
    
def amp_func(pos):
    return plane_wave(pos[0]) + gauss_beam([pos[1]])

# https://humaticlabs.com/blog/meep-double-slit/
def run(dists,plate_pos,plate_thickness,slit_size, wavelength, eps,
        pml_depth,freq_width = 0, resolution=None):
    '''
    Run a FDTD simulation for a PDI with the given input parameters.
    '''    
    # extract the simulation lengths in x, y and z
    x_dist, y_dist, z_dist = dists
    # set up a 3D grid (x,y,z) with direction of propagation in x
    cell = mp.Vector3(x_dist,y_dist,z_dist)
    # set the position of the plate, and other variables
    plate_pos_x, plate_pos_y, plate_pos_z = plate_pos
    plate_thickness = plate_thickness
   

    # set parameters for the plate and beam
    freq = 1/wavelength # in meep units
    freq_width = freq_width
    
    # controls the refractive index
    real_eps = np.real(eps)
    # controls the absorption of the material
    complex_eps = np.imag(eps)

    # add the whole plate, then make slits in it
    plate = [mp.Block(
        material = mp.Medium(epsilon = real_eps, D_conductivity=2*np.pi*freq*complex_eps/real_eps),
        size = mp.Vector3(plate_thickness, mp.inf,mp.inf),
        center = mp.Vector3(plate_pos_x,plate_pos_y,plate_pos_z)
    ),
    mp.Block(
        material = mp.Medium(epsilon = 1),
        size = mp.Vector3(plate_thickness,slit_size,mp.inf),
        center = mp.Vector3(plate_pos_x,-2,plate_pos_z)
    ),
    mp.Block(
        material = mp.Medium(epsilon = 1),
        size = mp.Vector3(plate_thickness,slit_size,mp.inf),
        center = mp.Vector3(plate_pos_x,2,plate_pos_z)
    ),]

    # append to make the total geometry
    geometry = []
    geometry.extend(plate)

    # resolution is the number of pixels per micron, typically want 8-10 pixels 
    # per wavelength
    if resolution==None:
        resolution =  10

    # set up perfectly matched layers on boundary
    pml_layers = [mp.PML(pml_depth)]
   

    # set up the sources
    sources = [mp.Source(src = mp.ContinuousSource(frequency=freq, 
                                            is_integrated = True,
                                            width = freq_width),
                        component=mp.Ez,
                        center=mp.Vector3(-x_dist/2+0.1,0,0),
                        size = mp.Vector3(0,y_dist,0.01*z_dist),
                        amplitude = 10),
              mp.Source(src = mp.ContinuousSource(frequency=freq, 
                                            is_integrated = True,
                                            width = freq_width),
                        component=mp.Ey,
                        center=mp.Vector3(-x_dist/2+0.1,0,0),
                        size = mp.Vector3(0,y_dist,0.01*z_dist),
                        amplitude = 10)]


    # set up the simulation
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution,
                        force_complex_fields = True)
                        
    
    # solve this way for a continuous source
    #sim.init_sim()
    #sim.solve_cw(10**-2)
    
    sim.run(until=cell[0]+cell[0]*0.5)

    # extract and save the near field data
    eps_data = sim.get_array(center = mp.Vector3(),size=cell, component=mp.Dielectric)
    ez_data = sim.get_array(center=mp.Vector3(),size=cell,component = mp.Ez)
    ey_data = sim.get_array(center=mp.Vector3(),size=cell,component= mp.Ey)
    ex_data = sim.get_array(center=mp.Vector3(),size=cell,component= mp.Ex)
    
    np.save('/home/ehold13/PhD/conda_meep/output_files/double_slit/e0_single_slit.npy',eps_data)
    np.save('/home/ehold13/PhD/conda_meep/output_files/double_slit/ez_single_slit.npy',ez_data)
    np.save('/home/ehold13/PhD/conda_meep/output_files/double_slit/ey_single_slit.npy',ey_data)
    np.save('/home/ehold13/PhD/conda_meep/output_files/double_slit/ez_single_slit.npy',ez_data)
    

    return eps_data, ez_data, ey_data, ex_data


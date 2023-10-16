import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def plane_wave(x,wavelength=0.589):
    return np.exp(1j*2*np.pi/wavelength*x)

def gauss_beam(x,width):
    return np.exp(-x**2/(2*(width/2)**2))

# https://humaticlabs.com/blog/meep-double-slit/
def run(dists,plate_pos,plate_thickness,radius, wavelength, beam_width, eps,
        pml_depth,pinhole_pos,window_pos,freq_width = 0, resolution=None):
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
    aperature = radius

    # set parameters for the plate and beam
    freq = 1/wavelength # in meep units
    freq_width = freq_width
    beam_width = beam_width
    # controls the refractive index
    real_eps = np.real(eps)
    # controls the absorption of the material
    complex_eps = np.imag(eps)

    # add the PDI plate, by adding a block, then putting cylinder through it
    plate = [mp.Block(
        material = mp.Medium(epsilon = real_eps, D_conductivity=2*np.pi*freq*complex_eps/real_eps),
        size = mp.Vector3(plate_thickness, mp.inf,mp.inf),
        center = mp.Vector3(plate_pos_x,plate_pos_y,plate_pos_z)
    ),
#    mp.Cylinder(
#        material = mp.Medium(epsilon = real_eps, D_conductivity=2*np.pi*freq*complex_eps/real_eps),
#        radius = 2*beam_width,
#        height = plate_thickness,
#        center = mp.Vector3(plate_pos_x,window_pos[0],window_pos[1]),
#        axis = mp.Vector3(1,0,0)

#    ),
    mp.Cylinder(
        material = mp.Medium(epsilon = 1),
        radius = aperature,
        height = plate_thickness,
        center = mp.Vector3(plate_pos_x,plate_pos_y,plate_pos_z),
        axis = mp.Vector3(1,0,0)

    ),
    ]

    # append to make the total geometry
    geometry = []
    geometry.extend(plate)

    # resolution is the number of pixels per micron, typically want 8-10 pixels 
    # per wavelength
    if resolution==None:
        resolution = max(dists)//wavelength*10

    # set up perfectly matched layers on boundary
    pml_layers = [mp.PML(pml_depth)]

    # set up the sources
    sources = [mp.Source(src = mp.ContinuousSource(frequency=freq, 
                                            is_integrated = True,
                                            width = freq_width),
                        component=mp.Ez,
                        center=mp.Vector3(-x_dist/2+pml_depth,window_pos[0],window_pos[1]),
                        size = mp.Vector3(y = beam_width, z = beam_width),
                        amplitude = 1),
              mp.Source(src = mp.ContinuousSource(frequency=freq, 
                                            is_integrated = True,
                                            width = freq_width),
                        component=mp.Ey,
                        center=mp.Vector3(-x_dist/2+pml_depth,window_pos[0],window_pos[1]),
                        size = mp.Vector3(y = beam_width, z = beam_width),
                        amplitude = 1),
                        
             mp.Source(src = mp.ContinuousSource(frequency=freq, 
                                            is_integrated = True,
                                            width = freq_width),
                        component=mp.Ez,
                        center=mp.Vector3(-x_dist/2+pml_depth,pinhole_pos[0],pinhole_pos[1]),
                        size = mp.Vector3(y = beam_width, z = beam_width),
                        amplitude = 1),
              mp.Source(src = mp.ContinuousSource(frequency=freq, 
                                            is_integrated = True,
                                            width = freq_width),
                        component=mp.Ey,
                        center=mp.Vector3(-x_dist/2+pml_depth,pinhole_pos[0],pinhole_pos[1]),
                        size = mp.Vector3(y = beam_width, z = beam_width),
                        amplitude = 1)]


    # set up the simulation
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution,
                        force_complex_fields = False)
                        
    
    # solve this way for a continuous source
    #sim.init_sim()
    #sim.solve_cw()
    
    sim.run(until=cell[0]+100)

    # extract and save the near field data
    eps_data = sim.get_array(center = mp.Vector3(),size=cell, component=mp.Dielectric)
    ez_data = sim.get_array(center=mp.Vector3(),size=cell,component = mp.Ez)
    ey_data = sim.get_array(center=mp.Vector3(),size=cell,component= mp.Ey)
    ex_data = sim.get_array(center=mp.Vector3(),size=cell,component= mp.Ex)
    
    np.save('/home/ehold13/PhD/conda_meep/output_files/e0_r_'+str(radius)+ '_bw_'+str(beam_width)+'.npy',eps_data)
    np.save('/home/ehold13/PhD/conda_meep/output_files/ez_r_'+str(radius)+ '_bw_'+str(beam_width)+'.npy',ez_data)
    np.save('/home/ehold13/PhD/conda_meep/output_files/ey_r_'+str(radius)+ '_bw_'+str(beam_width)+'.npy',ey_data)
    np.save('/home/ehold13/PhD/conda_meep/output_files/ez_r_'+str(radius)+ '_bw_'+str(beam_width)+'.npy',ez_data)
    

    return eps_data, ez_data, ey_data, ex_data

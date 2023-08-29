import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import math

def plane_wave(x,wavelength=0.589):
    return np.exp(1j*2*np.pi/wavelength*x)

def gauss_beam(x,width):
    return np.exp(-x**2/(2*(width/2)**2))
    
# generate Zernike wave
def zernike(x,y,z,m,n,c,A=1):
    if m < 0:
        sign = True
    else: 
        sign = False
    m = np.abs(m)

    Z = np.zeros((len(x),len(y),len(z)))
    R = np.zeros((len(x),len(y),len(z)))
    
    y,z = np.meshgrid(y,z,indexing='ij')

    rho = np.sqrt(y**2+z**2)/np.max(y)
    
    theta = np.arctan2(y,z)
    k = 0
    while k <= int((n-m)/2):
        numerator = (-1)**k * math.factorial(np.abs(n-k)) * rho**(n-2*k)
        denominator = math.factorial(abs(k)) * math.factorial(abs(int((n+m)/2)-k))\
                                * math.factorial(np.abs(int((n-m)/2)-k))
        R[0,:,:] += numerator/denominator
        k += 1
    
    if sign == False :
        kron = 0
        if m==n:
            kron = 1
        N = np.sqrt((2*n+2)/(1+kron))
        Z[0,:,:] = c*N*np.cos(m * theta) * R[0,:,:]
    else:
        kron = 0
        if m==n:
            kron = 1
        N = np.sqrt((2*n+2)/(1+kron))
        Z[0,:,:] = c*N*np.sin(m * theta) * R[0,:,:]
    
    Z[0,rho>1] = 0  
    A = np.full_like(Z,A)
    A[0,rho>1] = 0      

    return A*np.exp(1j*np.real(Z))

# https://humaticlabs.com/blog/meep-double-slit/
def run(dists,plate_pos,plate_thickness,radius, wavelength, beam_width, eps,
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
    mp.Cylinder(
        material = mp.Medium(epsilon = 1),
        radius = aperature,
        height = plate_thickness,
        center = mp.Vector3(plate_pos_x,plate_pos_y,plate_pos_z),
        axis = mp.Vector3(1,0,0)

    )]

    # append to make the total geometry
    geometry = []
    geometry.extend(plate)

    # resolution is the number of pixels per micron, typically want 8-10 pixels 
    # per wavelength
    if resolution==None:
        resolution = max(dists)//wavelength*10

    # set up perfectly matched layers on boundary
    pml_layers = [mp.PML(pml_depth, direction=mp.Y),
                    mp.PML(pml_depth, direction=mp.Z)]
    
    # generate an array which will hold the initial phase information
    init_phase = np.zeros((int(x_dist*resolution),int(y_dist*resolution),int(y_dist*resolution)), dtype = complex)
    
    # generate the desired phase on the beam
    if beam_width < z_dist:
        beam_phase = zernike(np.linspace(-x_dist/2,x_dist/2,int(x_dist*resolution)),np.linspace(-beam_width/2,beam_width/2,int(beam_width*resolution)),np.linspace(-beam_width/2,beam_width/2,int(beam_width*resolution)),-1,1,1,A=1)
    else:
        beam_phase = zernike(np.linspace(-x_dist/2,x_dist/2,int(x_dist*resolution)),np.linspace(-y_dist/2,y_dist/2,int(y_dist*resolution)),np.linspace(-z_dist/2,z_dist/2,int(z_dist*resolution)),-1,1,1,A=1)
        
    # check the size difference in the arrays to allow for beam_phase to be inset
    size_difference = np.shape(beam_phase)[1] - np.shape(init_phase[0,int(np.shape(init_phase)[1]/2)-int(np.shape(beam_phase)[1]/2):int(np.shape(init_phase)[1]/2)+int(np.shape(beam_phase)[1]/2),
    int(np.shape(init_phase)[2]/2)-int(np.shape(beam_phase)[2]/2):int(np.shape(init_phase)[2]/2)+int(np.shape(beam_phase)[2]/2)])[1]
    
     # inset the beam width into the inital phase array 
    init_phase[0,int(np.shape(init_phase)[1]/2)-int(np.shape(beam_phase)[1]/2):int(np.shape(init_phase)[1]/2)+int(np.shape(beam_phase)[1]/2+size_difference),
    int(np.shape(init_phase)[2]/2)-int(np.shape(beam_phase)[2]/2):int(np.shape(init_phase)[2]/2)+int(np.shape(beam_phase)[2]/2)+size_difference]= beam_phase[0,:,:]
    # inset the beam width into the inital phase array


    # set up the sources
    sources = [mp.Source(src=mp.ContinuousSource(frequency=freq,
        is_integrated = True),
        component=mp.Ez,
        center=mp.Vector3(-x_dist/2+pml_depth,0,0),
        size = mp.Vector3(y = beam_width, z = beam_width),
        amp_data=init_phase),
        
        mp.Source(src=mp.ContinuousSource(frequency=freq,
        is_integrated = True),
        component=mp.Ey,
        center=mp.Vector3(-x_dist/2+pml_depth,0,0),
        size = mp.Vector3(y = beam_width, z = beam_width),
        amp_data=init_phase)
    ]
    

    # set up the simulation
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution,
                        force_complex_fields = True)
                        
    
    # solve this way for a continuous source
    sim.init_sim()
    sim.solve_cw(10**-5)
    
    #sim.run(until=cell[0]+10)

    # extract and save the near field data
    eps_data = sim.get_array(center = mp.Vector3(),size=cell, component=mp.Dielectric)
    ez_data = sim.get_array(center=mp.Vector3(),size=cell,component = mp.Ez)
    ey_data = sim.get_array(center=mp.Vector3(),size=cell,component= mp.Ey)
    ex_data = sim.get_array(center=mp.Vector3(),size=cell,component= mp.Ex)
    
    np.save('/home/ehold13/PhD/conda_meep/output_files/e0_zernike_r_'+str(radius)+ '_bw_'+str(beam_width)+'.npy',eps_data)
    np.save('/home/ehold13/PhD/conda_meep/output_files/ez_zernike_r_'+str(radius)+ '_bw_'+str(beam_width)+'.npy',ez_data)
    np.save('/home/ehold13/PhD/conda_meep/output_files/ey_zernike_r_'+str(radius)+ '_bw_'+str(beam_width)+'.npy',ey_data)
    np.save('/home/ehold13/PhD/conda_meep/output_files/ez_zernike_r_'+str(radius)+ '_bw_'+str(beam_width)+'.npy',ez_data)
    

    return eps_data, ez_data, ey_data, ex_data

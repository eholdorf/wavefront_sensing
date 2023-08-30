import numpy as np 
import matplotlib.pyplot as plt
import astropy.constants as c
from tqdm import tqdm
import math
import astropy.units as un
import scipy.fft as sci
import scipy.signal
import scipy.ndimage
import scipy.fft
import hcipy

def FFT_BPM(x,y,z,w,wavelength):
    '''
    Based on : https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1522306&tag=1
    https://www.researchgate.net/publication/236246940_Beam_Propagation_Method_Based_on_Fast_Fourier_Transform_and_Finite_Difference_Schemes_and_Its_Application_to_Optical_Diffraction_Grating
    '''
    # set the number of points and the stepping size in each direction
    delta_z = z[1]-z[0]
    num_z = len(z)
    num_x = len(x)
    range_x = x[-1]-x[0]
    num_y = len(y)
    range_y = y[-1]-y[0]

    # calculate the wave number
    k0 = 2 * np.pi/wavelength

    # set up the matrix to put the field values into
    u = np.zeros((num_x,num_y,num_z), dtype=complex)
    # set the initial value equal to the incoming field
    u[:,:,0] = w
    # generate the wave numbers in the x and y dimensions, on a field that 
    # spaced in 1's and includes 0
    kx1 = np.linspace(0,int(num_x/2)+1,int(num_x/2))
    kx2 = np.linspace(-int(num_x/2),-1,int(num_x/2))
    kx = 2*np.pi/range_x *np.concatenate((kx1,kx2))

    ky1 = np.linspace(0,int(num_y/2)+1,int(num_y/2))
    ky2 = np.linspace(-int(num_y/2),-1,int(num_y/2))
    ky = 2*np.pi/range_y * np.concatenate((ky1,ky2))

    # create a grid of the x and y pixels
    KX, KY = np.meshgrid(kx,ky)

    for i in range(1,len(z)):
        print("BPM: {}/{}".format(i+1,len(z)), end="\r")
        # take the fourier transform of the step before
        fft = scipy.fft.fft2(u[:,:,i-1])
        fft *= np.exp(1j* delta_z*(KX**2 + KY**2)/(2*k0)) 
        ifft = scipy.fft.ifft2(fft)
        ifft *= np.exp(-1j*1.0003*k0*delta_z) 
        u[:,:,i] = ifft
       
        # absorbing boundary condition
        u[:5,:,i] = 0
        u[-5:,:,i] = 0
        u[:,:5,i] = 0
        u[:,-5:,i] = 0
        
    return u

def far_field(data,resolution,ns = [2**9,2**9,2**12],res_fact=1,wavelength=0.589):
    w = np.zeros((len(data)//res_fact,len(data)//res_fact), dtype=complex)

    for i in range(len(w)):
        for j in range(len(w)):
            w[i,j] = np.mean(data[res_fact*i:res_fact*i+res_fact,res_fact*j:res_fact*j+res_fact])

    nx,ny,nz = ns
    
    u_init = np.zeros((nx,ny), dtype=complex)
    u_init[(len(u_init)-len(w))//2:(len(u_init)-len(w))//2+len(w),(len(u_init)-len(w))//2:(len(u_init)-len(w))//2+len(w)] = w
    
    x,y,z=[np.linspace(-res_fact*resolution*nx/2,res_fact*resolution*nx/2,nx),np.linspace(-res_fact*resolution*ny/2,res_fact*resolution*ny/2,ny),np.linspace(0,res*nz,nz)]

    u = FFT_BPM(x,y,z,u_init,wavelength)
    
    return u,res_fact*resolution
    

# import the near field data

data = np.load('/home/ehold13/PhD/conda_meep/output_files/ey_r_0.25_bw_1.npy')+np.load('/home/ehold13/PhD/conda_meep/output_files/ez_r_0.25_bw_1.npy')

w_old = data[-10,:,:]
res = 0.0589
for i in range(50):
    u,res = far_field(w_old,res,res_fact=1.5,ns = [2**9,2**9,2**5])
    w_old = u[:,:,-1]
    
    del u
    
plt.figure()
plt.imshow(abs(w_old)**2)

plt.figure()
plt.imshow(np.angle(w_old))
plt.show()


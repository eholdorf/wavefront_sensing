##-- IMPORT STATEMENTS --##
from diffractio import np, plt, sp, um, mm, degrees
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.scalar_masks_XYZ import Scalar_mask_XYZ
from diffractio.scalar_fields_XYZ import Scalar_field_XYZ
from diffractio.vector_sources_XY import Vector_source_XY
from diffractio.utils_drawing import draw_several_fields    
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.fft
from skimage.restoration import unwrap_phase
import cmath
import poppy
from main_funcs import pdi_three_beams  as ptb
import scipy.signal

##-- START OF CODE --##

# define the important values for the simulation
wavelength = 0.589*um # wavelength of the light
length_xy = 1000*um # length of the simulation in x and y
z_dist = 1500*um # total distance to propagate the beam
beam_rad = 5*um # initial radius of the beam
pinhole_radius = 2* um # radius of the PDI pinhole
window_radius = 1.5*beam_rad # radius of the window, slightly larger than the beam

tilts = []
dists = np.linspace(5*window_radius,length_xy/3,10)

# for a variety of pinhole positions
for p in dists:
    pinhole_pos = (p*um,0)
    window_pos = (0,0)

    dists = [length_xy,beam_rad,pinhole_radius,window_radius] # wavelengths
    zernike_ns = [2] # list of zernike n co-efficients
    zernike_ms = [-2] # list of zernike m co-efficients
    zernike_cnms = [0.3] # coefficient of each zernike mode in radians
    A = 1 # amplitude of the wave

    # define the x,y space
    num = 10
    xs = np.linspace(-length_xy/2,length_xy/2,int(num*(length_xy/min(dists))),endpoint=True)
    ys = np.linspace(-length_xy/2,length_xy/2,int(num*(length_xy/min(dists))),endpoint=True)  

    plate = Scalar_mask_XY(xs,ys,wavelength)
    plate.reduce_matrix=None
    plate.square(r0=(0*um,0*um),size=(length_xy/2,length_xy/2))
    amp = 0.5**0.5
    plate.binarize(kind='amplitude',
                bin_level = None,
                level0=0,
                level1=0)

    window = Scalar_mask_XY(xs,ys,wavelength)
    window.reduce_matrix=None
    window.circle(r0=window_pos,radius = window_radius)

    window.binarize(kind='amplitude',
                bin_level = None,
                level0=0.0,
                level1=amp) 
                
    pinhole = Scalar_mask_XY(xs,ys,wavelength)
    pinhole.reduce_matrix=None
    pinhole.circle(r0=pinhole_pos,radius = pinhole_radius)  
            
    plate += window + pinhole

    # generate the source
    u_window = Scalar_source_XY(xs,ys,wavelength)
    u_window.plane_wave(A=1,theta=0,phi=0)

    t_window = Scalar_mask_XY(xs,ys,wavelength)
    t_window.circle(r0=window_pos,radius = beam_rad,angle=0)

    u_window *= t_window
    
    # define the source
    z_window = Scalar_source_XY(xs,ys,wavelength)
    z_window.zernike_beam(A=A,r0=window_pos,radius=beam_rad,n=zernike_ns,m=zernike_ms,c_nm=zernike_cnms)

    z_window *= u_window

    window_prop = z_window
    
    # pinhole beam
    
    u_pinhole = Scalar_source_XY(xs,ys,wavelength)
    u_pinhole.plane_wave(A=1,theta=0,phi=0)

    t_pinhole = Scalar_mask_XY(xs,ys,wavelength)
    t_pinhole.circle(r0=pinhole_pos,radius = beam_rad,angle=0)

    u_pinhole *= t_pinhole
    
    # define the source
    z_pinhole = Scalar_source_XY(xs,ys,wavelength)
    z_pinhole.zernike_beam(A=A,r0=pinhole_pos,radius=beam_rad,n=zernike_ns,m=zernike_ms,c_nm=zernike_cnms)

    z_pinhole *= u_pinhole

    pinhole_prop = z_pinhole

   # window_prop *= plate
   # pinhole_prop *= plate

    # beam at focal point
    focal_beam = window_prop + pinhole_prop

    focal_beam *= plate

    prop_pinhole = focal_beam.RS(z=z_dist,verbose= True,new_field=True)


    print("Finished propagation, starting phase retrieval")
    
    I_pdi = np.abs(prop_pinhole.u)**2
    
    plt.figure()
    plt.imshow(I_pdi)
    plt.show()
    
    fft = scipy.fft.fftshift(scipy.fft.fft(I_pdi,axis=1))

    if False:
        plt.figure()
        plt.plot(abs(np.sum(fft,0)),'k')


        plt.figure()
        plt.imshow(np.abs(fft)**2,norm=LogNorm())
       # plt.show()
    
    # find the three peaks
    m = scipy.signal.find_peaks(np.sum(abs(fft),axis=0),0.05*max(np.sum(abs(fft),axis=0)))
    # find the nearest peak - one of the phase peaks
    m = min(m[0])
    
    # find the width of the peak
    w = scipy.signal.peak_widths(np.sum(abs(fft),axis=0),[m])
    
    first_init = int(w[2][0]) - 3
    last_init = int(w[3][0]) + 4
    
    new_fft = np.zeros_like(fft)
    diff = last_init - first_init
    
    new_fft[:,int(len(new_fft)/2 - diff/2):int(len(new_fft)/2 + diff/2)] = fft[:,first_init:last_init]
    fft = new_fft
    
#    plt.figure()
#    plt.plot(abs(np.sum(fft,0)),'k')
#    plt.show()
    
    fft_shifted_ifft = scipy.fft.ifftshift(fft)
    

    ifft = scipy.fft.ifft(fft_shifted_ifft,axis=1)
    

    ifft = np.imag(np.log(ifft))
    
    ap_dist = 1.22 * wavelength * z_dist/(2*window_radius) + window_radius
    t_window = Scalar_mask_XY(xs,ys,wavelength)
    t_window.circle(r0=(0,0),radius = ap_dist,angle=0)
   
    ifft *= t_window.u
    
    # define the source
    z_window = Scalar_source_XY(xs,ys,wavelength)
    z_window.zernike_beam(A=A,r0=window_pos,radius=ap_dist,n=zernike_ns,m=zernike_ms,c_nm=zernike_cnms)
    
    z_window *= t_window
    
    zerns = poppy.zernike.decompose_opd_nonorthonormal_basis(ifft-z_window.u,nterms=10)
    tilts.append(zerns[1])
    
    print(zerns)
    
    plt.figure()
    plt.imshow(ifft)
    plt.title('Retrieved Phase')
    plt.colorbar()

    ifft -= ifft[int(len(ifft)/2),int(len(ifft)/2)]
    ifft *= t_window.u
    
    ifft = unwrap_phase(ifft,wrap_around=(True,True))
  
    plt.figure()
    plt.imshow(ifft)
    plt.title('Retrieved Phase Unwrapped')
    plt.colorbar()
    
    plt.figure()
    plt.imshow(np.angle(z_window.u))
    plt.title('Window Phase')
    plt.colorbar()
    
    plt.figure()
    plt.imshow(ifft - np.angle(z_window.u))
    plt.title('Phase Difference')
    plt.colorbar()
    
    
    plt.show()
    
plt.figure()
plt.plot(dists,tilts,'k.')
plt.show()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

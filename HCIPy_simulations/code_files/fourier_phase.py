import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.fft
from skimage.restoration import unwrap_phase
import cmath
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio import np, plt, sp, um, mm, degrees
#import poppy
from main_funcs import pdi_three_beams  as ptb
import scipy.signal

# helper functions

n = 2
m = -2
noll = 15
c = 1
orig = np.load('/home/ehold13/PhD/HCIPy_simulations/output_files/window_m'+str(m)+'_n'+str(n)+'.npy') 
pdi = np.load('/home/ehold13/PhD/HCIPy_simulations/output_files/final_m'+str(m)+'_n'+str(n)+'.npy') 

if False:
    diff = pdi-spherical

    l = np.angle(diff)-np.angle(orig)
    l = np.where(abs(l)>2,0,l)
    print(np.std(l)/(2*np.pi) *589)
    plt.figure()
    plt.imshow(l)
    plt.colorbar(label='Phase Error (nm)')
    plt.show()

if True:
    # take the fourier transform of pdi intensity
    I_pdi = np.abs(pdi)**2
    
    # define the important values for the simulation
    wavelength = ptb.wavelength # wavelength of the light
    length_xy = ptb.length_xy # length of the simulation in x and y
    z_dist = ptb.z_dist # total distance to propagate the beam
    beam_rad = ptb.beam_rad # initial radius of the beam
    pinhole_radius = ptb.pinhole_radius # radius of the PDI pinhole
    window_radius = ptb.window_radius
    
    pinhole_pos = ptb.pinhole_pos
    window_pos = ptb.window_pos
    
    dists = [length_xy,beam_rad,pinhole_radius,window_radius]
    zernike_ns = ptb.zernike_ns # list of zernike n co-efficients
    zernike_ms = ptb.zernike_ms # list of zernike m co-efficients
    zernike_cnms = ptb.zernike_cnms # coefficient of each zernike mode in radians
    A = ptb.A # amplitude of the wave
    
    num = ptb.num
    xs = ptb.xs
    ys = ptb.ys
    
    #ap = Scalar_mask_XY(xs,ys,wavelength)
    #ap.circle(r0=window_pos,radius = 2*2.44*wavelength*z_dist/window_radius,angle=0)
    
    #ap.u = np.where(ap.u==0,1e-10,ap.u)
    
    #I_pdi *= ap.u
    
    plt.figure()
    plt.imshow(I_pdi)
    
    fft = scipy.fft.fftshift(scipy.fft.fft(I_pdi,axis=1))

    if True:
        plt.figure()
        plt.plot(abs(np.sum(fft,0)),'k')


        plt.figure()
        plt.imshow(np.abs(fft)**2,norm=LogNorm())
        plt.show()
    
    # find the three peaks
    m = scipy.signal.find_peaks(np.sum(abs(fft),axis=0),0.01*max(np.sum(abs(fft),axis=0)))
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
    
    plt.figure()
    plt.plot(abs(np.sum(fft,0)),'k')
    plt.show()
    
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
    
    plt.figure()
    plt.imshow(ifft)
    plt.title('Retrieved Phase')
    plt.colorbar()

    ifft -= ifft[int(len(ifft)/2),int(len(ifft)/2)]
    ifft *= t_window.u
    
    ifft = unwrap_phase(ifft,wrap_around=(True,True))
    
    
    #zerns = poppy.zernike.decompose_opd(ifft,nterms=15,iterations = 5,verbose=False)
    #zerns = np.where(np.isnan(zerns),0,zerns)
    #print(zerns)
    
    #zerns = poppy.zernike.decompose_opd_nonorthonormal_basis(ifft,nterms=15,iterations = 5,verbose=False)
    #zerns = np.where(np.isnan(zerns),0,zerns)
    
   # print(zerns)
    
#    ns = []
#    ms = []
#    wanted = [0,1,2,3,noll-1]
#    for i in range(1,16):
#        nc,mc = poppy.zernike.noll_indices(i+1)
#        ns.append(nc)
#        ms.append((-1)**nc*mc)
#        
#    z_tilt_defoc = Scalar_source_XY(xs,ys,wavelength)
#    z_tilt_defoc.zernike_beam(A=A,r0=window_pos,radius=2.44*wavelength*z_dist/window_radius ,n=ns, m=ms, c_nm=zerns)
#    
#    z_tilt_defoc *= t_window
#    
#    zz = Scalar_source_XY(xs,ys,wavelength)
#    zz.zernike_beam(A=A,r0=window_pos,radius=2.44*wavelength*z_dist/window_radius ,n=[0,1,0], m=[0,-1,2], c_nm=[zerns[0],zerns[1],zerns[3]])
#    
#    zz *= t_window
    
#    plt.figure()
#    plt.imshow(np.angle(z_tilt_defoc.u))
#    plt.title("Zernike Retrieved Phase")
#    plt.colorbar()
#    
#    plt.figure()
#    plt.imshow(np.angle(zz.u))
#    plt.title("Tilt, Piston and Defocus Phase")
#    plt.colorbar()
#    
#    plt.figure()
#    plt.title("Zernike Phase without tilt, pison and defocus")
#    plt.imshow(np.angle(z_tilt_defoc.u) - np.angle(zz.u))
#    plt.colorbar()
   
   
    
    #ifft -= np.angle(z_window.u)
    #ifft /= ifft[int(len(ifft)/2),int(len(ifft)/2)]
    #ifft = unwrap_phase(ifft)
    
    
    #ifft -= ifft[3500,3500]
        
    #np.save('/home/ehold13/PhD/HCIPy_simulations/output_files/left_background_phase.npy',ifft)
    

    plt.figure()
    #plt.imshow(unwrap_phase(ifft,wrap_around=(True,True)))
    plt.imshow(ifft)
    plt.title('Retrieved Phase Unwrapped')
    plt.colorbar()
    
#    plt.figure()
#    #plt.imshow(unwrap_phase(np.angle(orig),wrap_around=(True,True)),vmin=-3.14159,vmax=3.14159)
#    plt.imshow(np.angle(pdi))
#    plt.title('True Phase')
#    plt.colorbar()
    
    plt.figure()
    #plt.imshow(unwrap_phase(np.angle(orig),wrap_around=(True,True)),vmin=-20,vmax = 20)
    plt.imshow(np.angle(z_window.u))
    plt.title('Window Phase')
    plt.colorbar()
    
    plt.figure()
    #plt.imshow(unwrap_phase(np.angle(orig),wrap_around=(True,True)),vmin=-20,vmax = 20)
    plt.imshow(ifft - np.angle(z_window.u))
    plt.title('Phase Difference')
    plt.colorbar()
    plt.show()







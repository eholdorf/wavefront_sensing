import poppy
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from matplotlib.colors import LogNorm
import scipy.fft
import aotools
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio import um, mm, degrees
from diffractio.scalar_masks_XY import Scalar_mask_XY

# helper functions

def rebin(in_array,ratio):
    '''
    Function which will rebin the input array to be ratio times smaller than
    the initial array in x and y by summing the data.
    '''

    out_array = np.zeros((int(np.shape(in_array)[0]/ratio),int(np.shape(in_array)[1]/ratio)))
    
    for i in range(np.shape(out_array)[0]):
        for j in range(np.shape(out_array)[1]):
            out_array[i,j] = np.nansum(in_array[i*ratio:i*ratio+ratio,j*ratio:j*ratio+ratio])
    
    return out_array


# define the important values for the simulation
wavelength = 0.589*um # wavelength of the light
length_xy = 1*mm # length of the simulation in x and y
length_xy_init = 0.1*length_xy 
num_pix_xy = 20*240 # number of pixels in each direction
pinhole_radius = 0.5*um # radius of the PDI pinhole
z_dist = 0.01 # m total distance to propagate the beam
f = 14.8
beam_rad = 1.22*f*wavelength # initial radius of the beam
zernike_ns = [2] # list of zernike n co-efficients
zernike_ms = [-2] # list of zernike m co-efficients
zernike_cnms = [1] # coefficient of each zernike mode in radians
A = 1 # amplitude of the wave

# define the x,y,z space
xs = np.linspace(-length_xy_init/2,length_xy_init/2,num_pix_xy)
ys = np.linspace(-length_xy_init/2,length_xy_init/2,num_pix_xy)

u = Scalar_source_XY(xs,ys,wavelength)
u.plane_wave(A=1,theta=0,phi=0)

t = Scalar_mask_XY(xs,ys,wavelength)
t.circle(r0=(0*um,0*um),radius = beam_rad,angle=0)

u *= t
# define the source
u0 = Scalar_source_XY(xs,ys,wavelength)
u0.zernike_beam(A=A,r0=(0*um,0*um),radius=beam_rad,n=zernike_ns,m=zernike_ms,c_nm=zernike_cnms)

u0 *= u

wavelength = 589e-9 # wavelength of the light
length_xy = 0.006 # length of the simulation in x and y
length_xy_init = 0.1*length_xy 
num_pix_xy = 20*240 # number of pixels in each direction
pinhole_radius = 0.5e-6 # radius of the PDI pinhole
z_dist = 0.1 # m total distance to propagate the beam
f = 14.8
beam_rad = 1.22*f*wavelength # initial radius of the beam

amp = 0.1
rad = (pinhole_radius)/(length_xy_init) *num_pix_xy

if rad < 1:
    raise Warning("Not Fine Enough Grid:{}".format(rad))
    
pinhole = aotools.functions.pupil.circle(rad,num_pix_xy)
pinhole = np.where(pinhole==0,amp,1)

after_pinhole = pinhole * u0.u

final = aotools.opticalpropagation.twoStepFresnel(after_pinhole,wavelength,length_xy_init/num_pix_xy,length_xy/num_pix_xy,z_dist)

plt.figure()
plt.imshow(abs(final)**2)

plt.figure()
plt.imshow(np.angle(final))
plt.show()


## important values in the simulation
#n_pix = 30*240 # number of pixels in x and y
#wavelength = 0.589*u.um # the wavelength of the light
#D_tele = 1 #m diameter of the telescope
#f = 14.8
#beam_rad = 1.22*f*wavelength/D_tele # the physical radius of the inital beam
#over_samp = 2 # the rate to oversample the beam (i.e. padding around the edge)
#pinhole_rad = 0.5*u.um # the radius of the pinhole

#detector = 0.5*u.mm
#plate_rad = detector
#z_dist = 1000*u.um

## generate the initial wavefront
#wf = poppy.FresnelWavefront(beam_radius=plate_rad,
#                            wavelength=wavelength,
#                            npix=n_pix,
#                            oversample=over_samp)
# 
## introduce a Zernike WF error (value in list is the size of the perturbation in m)
#coeffs =[0,0.5*wavelength.value]                      
#zwfe = poppy.ZernikeWFE(radius=beam_rad,coefficients=coeffs)
## make the wavefront circular
#aperature = poppy.CircularAperture(radius=beam_rad,
#                                   pad_factor=over_samp)
#wf *= aperature
#wf *= zwfe

##show the initial wavefront
#plt.figure()
#wf.display('both')

## generate the pinhole plate
#trans = np.ones((n_pix*over_samp,n_pix*over_samp)) * 0.005


#n_pix_pinhole = (pinhole_rad/(over_samp*2*detector)*n_pix).value

#for i in range(int(n_pix*over_samp/2 - n_pix_pinhole),int(n_pix*over_samp/2 + n_pix_pinhole)):
#    for j in range(int(n_pix*over_samp/2 - n_pix_pinhole),int(n_pix*over_samp/2 + n_pix_pinhole)):
#        if (i-n_pix*over_samp/2)**2 + (j-n_pix*over_samp/2)**2 <=n_pix_pinhole**2:
#            trans[i,j] = 1

#PDI = poppy.ArrayOpticalElement(name="PDI Plate",transmission=trans,pixelscale=2*detector/(n_pix*u.pixel))

## pass the wavefront through the pinhole
#wf_pinhole = wf.copy()*PDI

## propagate the wavefront distance
#wf_pinhole.propagate_fresnel(z_dist,display_intermed=False)


#np.save('/home/ehold13/PhD/HCIPy_simulations/output_files/final_'+str(coeffs) +'.npy',wf_pinhole.amplitude * np.exp(1j*wf_pinhole.phase))


#plt.figure()
#wf_pinhole.display('both',showpadding=False)

## generate the pinhole plate
#trans = np.zeros((n_pix*over_samp,n_pix*over_samp))
#n_pix_pinhole = (pinhole_rad/(over_samp*2*detector)*n_pix).value

#for i in range(int(n_pix*over_samp/2 - n_pix_pinhole),int(n_pix*over_samp/2 + n_pix_pinhole)):
#    for j in range(int(n_pix*over_samp/2 - n_pix_pinhole),int(n_pix*over_samp/2 + n_pix_pinhole)):
#        if (i-n_pix*over_samp/2)**2 + (j-n_pix*over_samp/2)**2 <=n_pix_pinhole**2:
#            trans[i,j] = 1

#PDI = poppy.ArrayOpticalElement(name="PDI Plate",transmission=trans,pixelscale=2*detector/(n_pix*u.pixel))

## pass the wavefront through the pinhole
#wf_spheric = wf.copy()*PDI

## propagate the wavefront distance
#wf_spheric.propagate_fresnel(z_dist,display_intermed=False)

#np.save('/home/ehold13/PhD/HCIPy_simulations/output_files/final_'+str(coeffs) +'.npy',wf_spheric.amplitude * np.exp(1j*wf_spheric.phase))


#plt.figure()
#wf_spheric.display('both')

#wf.propagate_fresnel(z_dist,display_intermed=False)

#plt.figure()
#wf.display('both')
#plt.show()

#I_pdi = wf_pinhole.intensity
#    
#fft = scipy.fft.fftshift(scipy.fft.fft(I_pdi,axis=1))

#if True:
#    plt.figure()
#    plt.plot(abs(np.sum(fft,0)),'k')


#    plt.figure()
#    plt.imshow(np.abs(fft)**2,norm=LogNorm())
#    plt.show()

#first_init = 1851
#last_init = 2075
#fft[:,:first_init] = 0
#fft[:,last_init:]=0
#    
#fft_shifted = np.zeros_like(fft)

#mid_int = int(len(fft_shifted)/2)
#len_phase = last_init-first_init

#fft_shifted[:,int(mid_int-len_phase/2):int(mid_int+len_phase/2)] = fft[:,first_init:last_init]

#fft_shifted_ifft = scipy.fft.ifftshift(fft)


#ifft = scipy.fft.ifft(fft_shifted_ifft,axis=1)

#ifft = np.imag(np.log(ifft))


#plt.figure()
#plt.imshow(ifft)

#plt.figure()
#plt.imshow(wf_pinhole.phase-wf_spheric.phase)

#plt.figure()
#plt.imshow(wf_spheric.phase)

#plt.show()


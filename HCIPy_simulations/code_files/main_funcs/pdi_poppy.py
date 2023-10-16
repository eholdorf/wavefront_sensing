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
    
def make_subplots(electric_field,n,length,title='',show=False):

    length *= 10**6  
    xvals = np.array([0,0.1*n,0.2*n,0.3*n,0.4*n,0.5*n,0.6*n,0.7*n,0.8*n,0.9*n,n])
    
    fig,ax = plt.subplots(1,2)
    fig.suptitle(title)
    ax[0].imshow(abs(electric_field)**2)
    ax[0].set_title('Intensity')
    ax[0].set_xticks(xvals)
    ax[0].set_xticklabels(np.round(length/n*xvals,0))
    ax[0].set_yticks(xvals)
    ax[0].set_yticklabels(np.round(length/n*xvals,0))
    ax[1].imshow(np.angle(electric_field))
    ax[1].set_title("Phase")
    ax[1].set_xticks(xvals)
    ax[1].set_xticklabels(np.round(length/n*xvals,0))
    ax[1].set_yticks(xvals)
    ax[1].set_yticklabels(np.round(length/n*xvals,0))
    
    if show:
        plt.show()


# define the important values for the simulation
wavelength = 0.589*um # wavelength of the light
num_pix_xy = 40*240 # number of pixels in each direction
f = 14.8
beam_rad = 1.22*f*wavelength*um # initial radius of the beam
pinhole_radius = 0.9*beam_rad # radius of the PDI pinhole
length_xy_init = 4*beam_rad
zernike_ns = [1] # list of zernike n co-efficients
zernike_ms = [1] # list of zernike m co-efficients
zernike_cnms = [0.1] # coefficient of each zernike mode in radians
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

# convert units to be in m from micron
wavelength *= 10**-6
length_xy_init *= 10**-6
pinhole_radius *= 10**-6
beam_rad *= 2*10**-6
z_dist = 20e-6 # m total distance to propagate the beam
coeff = 2
final_res = max([coeff * beam_rad*(1+(wavelength*z_dist/(np.pi*(beam_rad)**2))**2)**0.5, coeff*2.44*wavelength*z_dist/(2*pinhole_radius)])


focus = aotools.opticalpropagation.twoStepFresnel(u0.u,wavelength,length_xy_init/num_pix_xy,length_xy_init/num_pix_xy,z_dist)

make_subplots(focus,num_pix_xy,length_xy_init)

amp = 0.1
rad = (pinhole_radius)/(length_xy_init) *num_pix_xy

if rad < 1:
    raise Warning("Not Fine Enough Grid:{}".format(rad))
    
beam_circ = aotools.functions.pupil.circle(1/coeff *0.5*num_pix_xy,num_pix_xy)

pinhole = aotools.functions.pupil.circle(rad,num_pix_xy)

# PDI simulation
pinhole = np.where(pinhole==0,amp,1)

after_pinhole = pinhole * u0.u

final = aotools.opticalpropagation.twoStepFresnel(after_pinhole,wavelength,length_xy_init/num_pix_xy,final_res/num_pix_xy,z_dist)*beam_circ

np.save('/home/ehold13/PhD/HCIPy_simulations/output_files/final_m'+str(zernike_ms[0])+'_n'+str(zernike_ns[0])+'.npy',final)

make_subplots(final,num_pix_xy,final_res,'Interference')


# spherical simulation
pinhole = np.where(pinhole==amp,0,1)


spheric = pinhole * u0.u

spheric_final = aotools.opticalpropagation.twoStepFresnel(spheric,wavelength,length_xy_init/num_pix_xy,final_res/num_pix_xy,z_dist)*beam_circ

np.save('/home/ehold13/PhD/HCIPy_simulations/output_files/spherical_m'+str(zernike_ms[0])+'_n'+str(zernike_ns[0])+'.npy',spheric_final)

make_subplots(spheric_final,num_pix_xy,final_res,'Spherical')


orig = u0.u

orig_final = aotools.opticalpropagation.twoStepFresnel(orig,wavelength,length_xy_init/num_pix_xy,final_res/num_pix_xy,z_dist)*beam_circ

np.save('/home/ehold13/PhD/HCIPy_simulations/output_files/initial_m'+str(zernike_ms[0])+'_n'+str(zernike_ns[0])+'.npy',orig_final)

make_subplots(orig_final,num_pix_xy,final_res,'Initial Propagated')
make_subplots(spheric_final-final,num_pix_xy,length_xy_init,'Initial')
plt.show()

from diffractio import np, plt, sp, um, mm, degrees
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.scalar_masks_XYZ import Scalar_mask_XYZ
from diffractio.scalar_fields_XYZ import Scalar_field_XYZ
from diffractio.vector_sources_XY import Vector_source_XY
from diffractio.utils_drawing import draw_several_fields    

# helper functions

def rebin(in_array,ratio):
    '''
    Function which will rebin the input array to be ratio times smaller than
    the initial array in x and y by summing the data.
    '''
    
    if ratio == 1:
        return in_array

    out_array = np.zeros((int(np.shape(in_array)[0]/ratio),int(np.shape(in_array)[1]/ratio)))
    
    for i in range(np.shape(out_array)[0]):
        for j in range(np.shape(out_array)[1]):
            out_array[i,j] = np.nansum(in_array[i*ratio:i*ratio+ratio,j*ratio:j*ratio+ratio])
    
    return out_array
    

def make_subplots(electric_field, show=False):

    fig,ax = plt.subplots(1,2)
    ax[0].imshow(abs(electric_field.u)**2)
    ax[0].set_title('Intensity')
    ax[1].imshow(np.angle(electric_field.u))
    ax[1].set_title("Phase")
    
    if show:
        plt.show()
    

# define the important values for the simulation
wavelength = 0.589*um # wavelength of the light
length_xy = 4*mm # length of the simulation in x and y
length_xy_init = 0.0125*length_xy 
num_pix_xy = 40*240 # number of pixels in each direction
pinhole_radius = 0.5*um # radius of the PDI pinhole
z_dist = 1000*um # total distance to propagate the beam
D_tele = 1000*mm # telescope diameter
f = 1.48
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

make_subplots(u0)


rebinned_u0 = rebin(u0.u,int(length_xy/length_xy_init))
scaled_u0 = np.zeros_like(u0.u)
scaled_u0[int(np.shape(scaled_u0)[0]/2-np.shape(rebinned_u0)[0]/2):int(np.shape(scaled_u0)[0]/2+np.shape(rebinned_u0)[0]/2),
int(np.shape(scaled_u0)[1]/2-np.shape(rebinned_u0)[1]/2):int(np.shape(scaled_u0)[1]/2+np.shape(rebinned_u0)[1]/2)] = rebinned_u0

xf,yf = np.linspace(-length_xy/2,length_xy/2,num_pix_xy),np.linspace(-length_xy/2,length_xy/2,num_pix_xy)
u7 = Scalar_source_XY(xf,yf,wavelength)

u7.u = scaled_u0

u7.fast = False

u6 = u7.RS(z=z_dist,verbose= True,new_field=True)

#draw_several_fields((u0,u6),titles=('Original','Original Final Image'),logarithm=True)

make_subplots(u6)
#np.save('/home/ehold13/PhD/HCIPy_simulations/output_files/initial_m'+str(zernike_ms[0])+'_n'+str(zernike_ns[0])+'.npy',u6.u)
# define the pinhole plate
t0 = Scalar_mask_XY(xs,ys,wavelength)
t0.square(r0=(0*um,0*um),size=(length_xy,length_xy),angle=0)
amp = 0.05
t0.binarize(kind='amplitude',
            bin_level = None,
            level0=amp,
            level1=amp)

t1 = Scalar_mask_XY(xs,ys,wavelength)
t1.circle(r0=(0*um,0*um),radius = pinhole_radius,angle=0)

t1.binarize(kind='amplitude',
            bin_level = None,
            level0=0.0,
            level1=1-amp)           
t1 += t0

u2 = u0*t1



rebinned_u2 = rebin(u2.u,int(length_xy/length_xy_init))
scaled_u2 = np.zeros_like(u2.u)
scaled_u2[int(np.shape(scaled_u2)[0]/2-np.shape(rebinned_u2)[0]/2):int(np.shape(scaled_u2)[0]/2+np.shape(rebinned_u2)[0]/2),
int(np.shape(scaled_u2)[1]/2-np.shape(rebinned_u2)[1]/2):int(np.shape(scaled_u2)[1]/2+np.shape(rebinned_u2)[1]/2)] = rebinned_u2

xf,yf = np.linspace(-length_xy/2,length_xy/2,num_pix_xy),np.linspace(-length_xy/2,length_xy/2,num_pix_xy)
u2 = Scalar_source_XY(xf,yf,wavelength)

u2.u = scaled_u2

u2.fast=False

u3 = u2.RS(z=z_dist,verbose= True,new_field=True)

u3_detector = rebin(u3.u,int(np.shape(u3.u)[0]/240))
#np.save('/home/ehold13/PhD/HCIPy_simulations/output_files/final_m'+str(zernike_ms[0])+'_n'+str(zernike_ns[0])+'.npy',u3.u)

#draw_several_fields((t1,u3),titles=('Mask','PDI Final Image'),logarithm=True)

make_subplots(u3)

del u2,u3,t0,t1


# repeat above, but just ot look at the generated circular reference wavefront

t2 = Scalar_mask_XY(xs,ys,wavelength)
t2.square(r0=(0*um,0*um),size=(length_xy,length_xy),angle=0)

t2.binarize(kind='amplitude',
            bin_level = None,
            level0=0,
            level1=0)

t3 = Scalar_mask_XY(xs,ys,wavelength)
t3.circle(r0=(0*um,0*um),radius = pinhole_radius,angle=0)

t3 += t2

u4 = u0*t3

rebinned_u4 = rebin(u4.u,int(length_xy/length_xy_init))
scaled_u4 = np.zeros_like(u4.u)
scaled_u4[int(np.shape(scaled_u4)[0]/2-np.shape(rebinned_u4)[0]/2):int(np.shape(scaled_u4)[0]/2+np.shape(rebinned_u4)[0]/2),
int(np.shape(scaled_u4)[1]/2-np.shape(rebinned_u4)[1]/2):int(np.shape(scaled_u4)[1]/2+np.shape(rebinned_u4)[1]/2)] = rebinned_u4

u4 = Scalar_source_XY(xf,yf,wavelength)

u4.u = scaled_u4

u4.fast=False

u5 = u4.RS(z=z_dist,verbose= True,new_field=True)

u5_detector = rebin(u5.u,int(np.shape(u5.u)[0]/240))

#np.save('/home/ehold13/PhD/HCIPy_simulations/output_files/spherical_m'+str(zernike_ms[0])+'_n'+str(zernike_ns[0])+'.npy',u5.u)

make_subplots(u5)
del u4,u5,t2,t3

plt.show()


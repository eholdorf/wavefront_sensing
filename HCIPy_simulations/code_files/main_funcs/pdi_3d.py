from diffractio import np, plt, sp, um, mm, degrees
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.scalar_masks_XYZ import Scalar_mask_XYZ
from diffractio.scalar_fields_XYZ import Scalar_field_XYZ
from diffractio.vector_sources_XY import Vector_source_XY

# define the important values for the simulation
wavelength = 0.589*um
length_xy, length_z = 30*um,5*um
num_pix_xy,num_pix_z = 2**10, 2**7

# define the x,y,z space
xs = np.linspace(-length_xy/2,length_xy/2,num_pix_xy)
ys = np.linspace(-length_xy/2,length_xy/2,num_pix_xy)
zs = np.linspace(0,length_z,num_pix_z)

# define the source
u0 = Scalar_source_XY(xs,ys,wavelength)
u0.plane_wave(A=1)

slit_pos = 10*um
slit_width = 5*um
# define the plane where the source exists
t0 = Scalar_mask_XY(xs,ys,wavelength)
t0.square(r0=(0*um,0*um),size=(length_xy,length_xy),angle=0)
t0.slit(x0=-slit_pos*um,size = slit_width,angle = 0)
t0.slit(x0=slit_pos*um,size = slit_width,angle = 0)
t0.square(r0=(-slit_pos,0*um),size=(length_xy,slit_width),angle=0)
t0.square(r0=(slit_pos,0*um),size=(length_xy,slit_width),angle=0)
#t0.circle(r0=(0.*um,0*um),radius = 0.25*um,angle = 0)
# apply the field to the plane

# generate the pinhole plate
t1 = Scalar_mask_XYZ(xs,ys,zs,wavelength)
t1.square(r0=(0*um,0*um,0*um),length=(length_xy,length_xy, 0.6*um),angles=(0,0,0),refraction_index = 1e6)
t1.square(r0=(0*um,-0.5*um,0*um),length=(length_xy,0.3*um, 0.6*um),angles=(0,0,0),refraction_index = 1)
t1.square(r0=(0*um,0.5*um,0*um),length=(length_xy,0.3*um, 0.6*um),angles=(0,0,0),refraction_index = 1)
#t1.cylinder(r0=(0.*um,0*um,0.*um),radius = 0.25*um,length = 0.5*um,axis = 'z', angle = 0, refraction_index=1.0003)
t1.incident_field(u0=u0*t0)


t1.fast = False
t1.RS(verbose= True, num_processors=1)

plt.figure()
plt.imshow(abs(t1.u[:,:,-1]))
plt.show()

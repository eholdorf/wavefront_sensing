import numpy as np
import matplotlib.pyplot as plt
import aotools
from diffractio import um, mm, degrees
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.scalar_masks_XYZ import Scalar_mask_XYZ
from diffractio.scalar_fields_XYZ import Scalar_field_XYZ
from diffractio.vector_sources_XY import Vector_source_XY
from diffractio.utils_drawing import draw_several_fields  
# double slit experiment - diffractio
if False:
    # define the important values for the simulation
    wavelength = 1.5*um
    length_xy = 1.5*mm
    num_pix_xy = 2**10

    # define the x,y,z space
    xs = np.linspace(-length_xy,length_xy,num_pix_xy)
    ys = np.linspace(-0.1*mm,0.1*mm,num_pix_xy)

    # define the source
    u0 = Scalar_source_XY(xs,ys,wavelength)
    u0.plane_wave(A=1)

    slit_pos = 0.15*mm
    slit_width = 0.05*mm

    # define the plane where the source exists
#    t0 = Scalar_mask_XY(xs,ys,wavelength)
#    t0.square(r0=(0*um,0*um),size=(slit_width,length_xy),angle=0)
    
    t1 = Scalar_mask_XY(xs,ys,wavelength)
    t1.square(r0=(slit_pos,0*um),size=(slit_width,length_xy),angle=0)
    
    t2 = Scalar_mask_XY(xs,ys,wavelength)
    t2.square(r0=(-slit_pos,0*um),size=(slit_width,length_xy),angle=0)
    
#    t3 = Scalar_mask_XY(xs,ys,wavelength)
#    t3.square(r0=(2*slit_pos,0*um),size=(slit_width,length_xy),angle=0)
#    
#    t4 = Scalar_mask_XY(xs,ys,wavelength)
#    t4.square(r0=(-2*slit_pos,0*um),size=(slit_width,length_xy),angle=0)
    


    u2 = u0*(t1+t2)
    u2.fast=True

    u3 = u2.RS(z=20*mm,verbose= True,new_field=True)

    draw_several_fields((t1+t2,u3),titles=('Mask','Final Image'),logarithm=True)
    plt.show()
    
# double slit AOtools
if True:
    # define the important values for the simulation
    wavelength = 1.5*um
    length_xy = 1.5*mm
    num_pix_xy = 2**12

    # define the x,y,z space
    xs = np.linspace(-0.5*length_xy,0.5*length_xy,num_pix_xy)
    ys = np.linspace(-0.5*length_xy,0.5*length_xy,num_pix_xy)

    # define the source
    u0 = Scalar_source_XY(xs,ys,wavelength)
    u0.plane_wave(A=1)

    slit_pos = 0.15*mm
    slit_width = 0.05*mm

    # define the plane where the source exists
    t0 = Scalar_mask_XY(xs,ys,wavelength)
    t0.square(r0=(0*um,0*um),size=(slit_width,length_xy),angle=0)
    
    t1 = Scalar_mask_XY(xs,ys,wavelength)
    t1.square(r0=(slit_pos,0*um),size=(slit_width,length_xy),angle=0)
    
    t2 = Scalar_mask_XY(xs,ys,wavelength)
    t2.square(r0=(-slit_pos,0*um),size=(slit_width,length_xy),angle=0)
    
#    t3 = Scalar_mask_XY(xs,ys,wavelength)
#    t3.square(r0=(2*slit_pos,0*um),size=(slit_width,length_xy),angle=0)
#    
#    t4 = Scalar_mask_XY(xs,ys,wavelength)
#    t4.square(r0=(-2*slit_pos,0*um),size=(slit_width,length_xy),angle=0)

    # define the plane where the source exists
    pinhole = aotools.functions.pupil.circle(slit_width/length_xy*num_pix_xy,num_pix_xy)
    
    pinhole = t1.u + t2.u 
    
    wave = u0.u
    
    wave *= pinhole

    
    final = aotools.opticalpropagation.twoStepFresnel(wave,1500*10**-9,0.0015/num_pix_xy,0.0015/num_pix_xy,0.02)
    
    plt.figure()
    plt.imshow(abs(final)**2)
    xvals = np.array([0,0.1*num_pix_xy,0.2*num_pix_xy,0.3*num_pix_xy,0.4*num_pix_xy,
    0.5*num_pix_xy,0.6*num_pix_xy,0.7*num_pix_xy,0.8*num_pix_xy,0.9*num_pix_xy,num_pix_xy])
    plt.xticks(xvals,np.round(1.5/num_pix_xy*xvals,2))
    plt.yticks(xvals,np.round(1.5/num_pix_xy*xvals,2))
    plt.show()
    
    



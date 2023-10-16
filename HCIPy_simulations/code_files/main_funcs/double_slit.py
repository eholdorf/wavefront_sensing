import numpy as np
import matplotlib.pyplot as plt
import aotools
from diffractio import um, mm, degrees
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.scalar_masks_XZ import Scalar_mask_XZ
from diffractio.scalar_fields_XYZ import Scalar_field_XYZ
from diffractio.scalar_sources_X import Scalar_source_X
from diffractio.scalar_masks_X import Scalar_mask_X
from diffractio.utils_drawing import draw_several_fields  
import copy
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
if False:
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
    
    
# 2D PDI
if True:
    # define the important values for the simulation
    wavelength = 0.589*um
    length_x = 12*um
    length_z = 18*um
    thickness = 3*um
    position = -length_z/2+length_x/10+thickness/2
    
    pinhole_pos = 2*um
    window_pos = -1*um
    
    pinhole_rad = 0.3*um
    window_rad = 0.6*um
    
    
    res_fact = 100
    
    index = [0,1,2]
    
    for i in index:
        
        num = int(res_fact/wavelength*max([length_x,length_z]))
        # define the x,y,z space
        xs = np.linspace(-length_x/2,length_x/2,num)
        zs = np.linspace(-length_z/2,length_z/2,num)

        # define the source
        u0 = Scalar_source_X(xs,wavelength)
        u0.plane_wave(A=1)


        t1 = Scalar_mask_XZ(xs,zs,wavelength)
        #t1.rectangle(r0=(pinhole_pos,position),
        #          size=(2*pinhole_rad,thickness),
        #          angle=0,
        #          refraction_index=1)
        
        t1.rectangle(r0=(length_x/4+pinhole_pos/2+window_rad/2,position),
                  size=(length_x/2-pinhole_pos-pinhole_rad,thickness),
                  angle=0,
                  refraction_index=1e6+1e10j)
                  
        t1.rectangle(r0=(pinhole_pos-pinhole_rad - 0.5*(pinhole_pos-pinhole_rad +abs(window_pos)-window_rad),position),
                     size = (pinhole_pos-pinhole_rad + abs(window_pos) -window_rad,thickness),
                     angle =0,
                     refraction_index=1e6+1e10j
                     )
                  
        t1.rectangle(r0=(window_pos,position),
                  size=(2*window_rad,thickness),
                  angle=0,
                  refraction_index=1+0.01j)
        
        t1.rectangle(r0=(-(length_x/4 + abs(window_pos/2) + window_rad/2),position),
                     size = ((length_x/2 - abs(window_pos)-window_rad),thickness),
                     angle= 0,
                     refraction_index = 1e6+1e10j
        
        )


        plate = t1
        
        plate.incident_field(u0)
        
        if i ==0:
            plate.RS(verbose=True)
        elif i==1:
            plate.BPM(verbose=True)
            #final = abs(plate.u)**2
        elif i==2:
            plate.WPM(verbose=True)
            #final -= abs(plate.u)**2

        #plate.draw(kind='intensity',draw_borders=True)
        
        plt.figure()
        plt.imshow(abs(plate.u)**2)
        
        
        del plate,t1,u0
    
    meep = np.load('/home/ehold13/PhD/conda_meep/output_files/ey_r_0.3_bw_1.npy')+np.load('/home/ehold13/PhD/conda_meep/output_files/ez_r_0.3_bw_1.npy')
    meep = meep.T
    
    plt.figure()
    plt.imshow(abs(meep)**2)
    plt.show()
    
if False:
    # define the important values for the simulation
    wavelength = 0.589*um
    length_x = 10*um
    length_y = 10*um
    length_z = 100*um
    thickness = 3*um
    
    pinhole_pos = 2*um
    window_pos = -1*um
    
    pinhole_rad = 0.3*um
    window_rad = 0.6*um
    
    
    res_fact = 10
    
    index = [0,1,2]
    
    for i in index:
        
        num = int(2 ** (np.log(res_fact/wavelength*max([length_x,length_z])-2)))
        # define the x,y,z space
        xs = np.linspace(-length_x/2,length_x/2,num)
        ys = np.linspace(-length_y/2,length_y/2,num)
        zs = np.linspace(-length_z/2,length_z/2,num)

        # define the source
        u0 = Scalar_source_XY(xs,ys,wavelength)
        u0.plane_wave(A=1)


        t1 = Scalar_mask_XY(xs,ys,wavelength)
        
        t1.circle(r0=(pinhole_pos,0),
        radius = pinhole_rad, 
        angle = 0)
        
        t1.circle(r0=(window_pos,0),
        radius = window_rad, 
        angle = 0)

        plate = t1
        plate.incident_field = u0
        if i ==0:
            plate.fast = True
            plate.RS(z = length_z,verbose=True)

        plate.draw(kind='intensity')
        plt.show()
        
        
        del plate,t1,u0
    

    plt.show()



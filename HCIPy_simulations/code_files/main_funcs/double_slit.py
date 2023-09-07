
# double slit experiment
if True:
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

from diffractio import np, plt, sp, um, mm, degrees
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.scalar_masks_XYZ import Scalar_mask_XYZ
from diffractio.scalar_fields_XYZ import Scalar_field_XYZ
from diffractio.vector_sources_XY import Vector_source_XY
from diffractio.utils_drawing import draw_several_fields

if False:
    # define important values
    wavelength = 0.589*um
    length_xy,length_z = 50*um,10*um
    num_pix_xy,num_pix_z = 2**9, 2**9
    width_plate = 0.5*um
    radius_pinhole = 0.25*um

    # define the x,y,z space
    xs = np.linspace(-length_xy/2,length_xy/2,num_pix_xy)
    ys = np.linspace(-length_xy/2,length_xy/2,num_pix_xy)
    zs = np.linspace(0,length_z,num_pix_z)

    # define the source
    u0 = Scalar_source_XY(xs,ys,wavelength)
    u0.plane_wave(A=10)

    # make the pinhole

    # start with the semi-transparent plate
    t0 = Scalar_mask_XYZ(xs,ys,zs,wavelength)
    t0.square(r0=(0*um,0*um,0*um),length = (length_xy,length_xy,width_plate),refraction_index=1.5+1.0j)

    t1 = Scalar_mask_XYZ(xs,ys,zs,wavelength)
    t1.cylinder(r0=(0*um,0*um,0*um),radius = (radius_pinhole,radius_pinhole),length=width_plate,axis='z',refraction_index=1.0,angle=0)
    
    t3 = t0+t1
    t3.incident_field(u0)
    t3.fast=True

    t3.RS(verbose=True)


    draw_several_fields((t3,n[:,2**7,:],abs(t3.u[:,2**7,:])**2),titles=('Mask','Final Image'),logarithm=True)
    plt.show()
    
if True:
    # define the important values for the simulation
    wavelength = 0.589*um
    length_xy = 1000*um
    num_pix_xy = 8000
    pinhole_radius = 0.5*um
    z_dist = 500*um
    beam_rad = 10*um
    zernike_ns = [1]
    zernike_ms = [1]
    zernike_cnms = [1e-10]
    A = 1

    # define the x,y,z space
    xs = np.linspace(-length_xy/2,length_xy/2,num_pix_xy)
    ys = np.linspace(-length_xy/2,length_xy/2,num_pix_xy)

    # define the source
    u0 = Scalar_source_XY(xs,ys,wavelength)
    u0.zernike_beam(A=A,r0=(0*um,0*um),radius=beam_rad,n=zernike_ns,m=zernike_ms,c_nm=zernike_cnms)
    
    np.save('/home/ehold13/PhD/HCIPy_simulations/output_files/initial_m'+str(zernike_ms[0])+'_n'+str(zernike_ns[0])+'.npy',u0.u)
    
    # define the plane where the source exists
    t0 = Scalar_mask_XY(xs,ys,wavelength)
    #t0.gray_scale(num_levels=1,levelMin=0.1,levelMax=0.1)
    t0.square(r0=(0*um,0*um),size=(length_xy,length_xy),angle=0)
    
    t0.binarize(kind='amplitude',
                bin_level = None,
                level0=0.0003,
                level1=0.0003)

    t1 = Scalar_mask_XY(xs,ys,wavelength)
    t1.circle(r0=(0*um,0*um),radius = pinhole_radius,angle=0)
    
    t1.binarize(kind='amplitude',
                bin_level = None,
                level0=0.0,
                level1=1.21)

    u2 = u0*(t0+t1)
    u2.fast=True

    u3 = u2.RS(z=z_dist,verbose= True,new_field=True)
    
    np.save('/home/ehold13/PhD/HCIPy_simulations/output_files/final_m'+str(zernike_ms[0])+'_n'+str(zernike_ns[0])+'.npy',u3.u)
    
    plt.figure()
    draw_several_fields((t0+t1,u3),titles=('Mask','PDI Final Image'),logarithm=True)
    
    del u2,u3,t0,t1
    # define the plane where the source exists

    t2 = Scalar_mask_XY(xs,ys,wavelength)
    t2.square(r0=(0*um,0*um),size=(length_xy,length_xy),angle=0)
    
    t2.binarize(kind='amplitude',
                bin_level = None,
                level0=0,
                level1=0)
    
    t3 = Scalar_mask_XY(xs,ys,wavelength)
    t3.circle(r0=(0*um,0*um),radius = pinhole_radius,angle=0)
    t3.binarize(kind='amplitude',
                bin_level = None,
                level0=0.0,
                level1=1.21)
 
    u4 = u0*(t2+t3)
    u4.fast=True

    u5 = u4.RS(z=z_dist,verbose= True,new_field=True)
    
    np.save('/home/ehold13/PhD/HCIPy_simulations/output_files/spherical_m'+str(zernike_ms[0])+'_n'+str(zernike_ns[0])+'.npy',u5.u)
   
    
    plt.figure()
    draw_several_fields((t2+t3,u5),titles=('Mask','Spherical Final Image'),logarithm=True)
    del u4,u5,t2,t3
    
    plt.show()
    
    
    
    

# double slit experiment
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

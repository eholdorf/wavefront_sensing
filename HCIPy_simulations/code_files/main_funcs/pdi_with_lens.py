   
# using Diffractio 
if False:    
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
    length_xy = 1*mm # length of the simulation in x and y
    length_xy_init = 0.125*length_xy 
    num_pix_xy = 40*240 # number of pixels in each direction
    pinhole_radius = 1*um # radius of the PDI pinhole
    z_dist = 10000*um # total distance to propagate the beam
    beam_rad = length_xy/3 # initial radius of the beam
    zernike_ns = [2] # list of zernike n co-efficients
    zernike_ms = [-2] # list of zernike m co-efficients
    zernike_cnms = [1] # coefficient of each zernike mode in radians
    A = 1 # amplitude of the wave

    # define the x,y space
    xs = np.linspace(-length_xy/2,length_xy/2,num_pix_xy)
    ys = np.linspace(-length_xy/2,length_xy/2,num_pix_xy)


    # generate the source
    u = Scalar_source_XY(xs,ys,wavelength)
    u.plane_wave(A=1,theta=0,phi=0)

    t = Scalar_mask_XY(xs,ys,wavelength)
    t.circle(r0=(0*um,0*um),radius = beam_rad,angle=0)

    u *= t
    # define the source
    u0 = Scalar_source_XY(xs,ys,wavelength)
    u0.zernike_beam(A=A,r0=(0*um,0*um),radius=beam_rad,n=zernike_ns,m=zernike_ms,c_nm=zernike_cnms)

    u0 *= u
    
    u0.draw(kind='intensity',logarithm = True)
    u0.draw(kind='phase',logarithm = True)

    # focus on the PDI plate
    t0 = Scalar_mask_XY(xs,ys,wavelength)
    t0.lens_spherical(r0=(0,0),focal=z_dist,radius=length_xy,refraction_index=1.5)

    u0 *= t0
        
    init_prop = u0.RS(z=z_dist,verbose=True,new_field=True)
    
    init_prop.draw(kind='intensity')
    init_prop.draw(kind='phase')


    t1 = Scalar_mask_XY(xs,ys,wavelength)
    t1.square(r0=(0*um,0*um),size=(length_xy,length_xy),angle=0)
    amp = 0.05**0.5
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

    pinhole = init_prop*t1

    prop_pinhole = pinhole.RS(z=z_dist,verbose= True,new_field=True)
    
    t2 = Scalar_mask_XY(xs,ys,wavelength)
    t2.lens_spherical(r0=(0,0),focal=-z_dist,radius=length_xy/2,refraction_index=1.5)
    
    prop_pinhole *= t2
    
    u4 = prop_pinhole.RS(z=50000*um,verbose=True,new_field=True)

    #np.save('/home/ehold13/PhD/HCIPy_simulations/output_files/final_m'+str(zernike_ms[0])+'_n'+str(zernike_ns[0])+'.npy',u3.u)

    prop_pinhole.draw(kind='intensity',logarithm=True)
    prop_pinhole.draw(kind='phase')

    
    
    plt.show()


# using aotools
if False:
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
    num_pix_xy = 30*240 # number of pixels in each direction
    pinhole_radius = 0.5*um # radius of the PDI pinhole
    length_xy = 6*mm
    beam_rad = length_xy/3
    zernike_ns = [2] # list of zernike n co-efficients
    zernike_ms = [-2] # list of zernike m co-efficients
    zernike_cnms = [1] # coefficient of each zernike mode in radians
    A = 1 # amplitude of the wave
    z_dist = 10000*um

    # define the x,y space
    xs = np.linspace(-length_xy/2,length_xy/2,num_pix_xy)
    ys = np.linspace(-length_xy/2,length_xy/2,num_pix_xy)

    u = Scalar_source_XY(xs,ys,wavelength)
    u.plane_wave(A=1,theta=0,phi=0)

    t = Scalar_mask_XY(xs,ys,wavelength)
    t.circle(r0=(0*um,0*um),radius = beam_rad,angle=0)

    u *= t
    # define the source
    u0 = Scalar_source_XY(xs,ys,wavelength)
    u0.zernike_beam(A=A,r0=(0*um,0*um),radius=beam_rad,n=zernike_ns,m=zernike_ms,c_nm=zernike_cnms)

    u0 *= u
    
    make_subplots(u0.u,num_pix_xy,length_xy*10**-6,'Initial Field')
    
    # focus on the PDI plate at distance z_dist
#    t0 = Scalar_mask_XY(xs,ys,wavelength)
#    t0.lens_spherical(r0=(0,0),focal=z_dist,refraction_index = 1.5,radius=length_xy)

#    u0 *= t0

    # convert units to be in m from micron
    wavelength *= 10**-6
    length_xy *= 10**-6
    pinhole_radius *= 10**-6
    z_dist *= 10**-6 # m total distance to propagate the beam
    
    size_focus = 0.02*length_xy
    
    #focus = aotools.opticalpropagation.twoStepFresnel(u0.u,wavelength,length_xy/num_pix_xy,size_focus/num_pix_xy,z_dist)
    
    focus = aotools.opticalpropagation.lensAgainst(u0.u,wavelength,length_xy/num_pix_xy,z_dist)

    make_subplots(focus,num_pix_xy,size_focus,'Focal Image')
    

    amp = 0.01
    rad = (pinhole_radius)/(size_focus)*num_pix_xy

    if rad < 1:
        raise Warning("Not Fine Enough Grid:{}".format(rad))
        
    pinhole = aotools.functions.pupil.circle(rad,num_pix_xy)

    # PDI simulation
    pinhole = np.where(pinhole==0,amp,1)

    after_pinhole = pinhole*focus

    final = aotools.opticalpropagation.twoStepFresnel(after_pinhole,wavelength,size_focus/num_pix_xy,length_xy/num_pix_xy,z_dist)
    
    t1 = Scalar_mask_XY(xs,ys,wavelength)
    t1.lens_spherical(r0=(0,0),focal=-z_dist,refraction_index = 1.5,radius=length_xy)
    
    final *= t1.u
    
    final = aotools.opticalpropagation.oneStepFresnel(after_pinhole,wavelength,length_xy/num_pix_xy,0.1)

    np.save('/home/ehold13/PhD/HCIPy_simulations/output_files/final_m'+str(zernike_ms[0])+'_n'+str(zernike_ns[0])+'.npy',final)

    make_subplots(final,num_pix_xy,length_xy,'Interference')
    
    initial_prop = aotools.opticalpropagation.twoStepFresnel(focus,wavelength,size_focus/num_pix_xy,length_xy/num_pix_xy,z_dist)
    initial_prop *= t1.u
    initial_prop = aotools.opticalpropagation.oneStepFresnel(focus,wavelength,length_xy/num_pix_xy,0.1)
    make_subplots(initial_prop,num_pix_xy,length_xy,'Initial Propagated')
    plt.show()




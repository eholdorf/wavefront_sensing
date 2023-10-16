   # using Diffractio 
if True:    
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
            
    def make_circle(field,size,centre,radius):
    
        x,y = np.linspace(-size,size,len(field)), np.linspace(-size,size,len(field))
        
        X,Y = np.meshgrid(x,y)
        
        rho = np.sqrt((X-centre[0])**2+(Y-centre[1])**2)/size
        
        field[rho>1] = 0
        
        return field
        

    # define the important values for the simulation
    wavelength = 0.589*um # wavelength of the light
    length_xy = 800*um # length of the simulation in x and y
    z_dist = 2000*um # total distance to propagate the beam
    beam_rad = 5*um # initial radius of the beam
    pinhole_radius = 2* um #0.5*um # radius of the PDI pinhole
    window_radius = 1.1*beam_rad
    
    pinhole_pos = (150*um,0)
    window_pos = (0,0)
    
    dists = [length_xy,beam_rad,pinhole_radius,window_radius] # wavelengths
    zernike_ns = [2] # list of zernike n co-efficients
    zernike_ms = [-2] # list of zernike m co-efficients
    zernike_cnms = [0.1] # coefficient of each zernike mode in radians
    A = 1 # amplitude of the wave

    # define the x,y space
    num = 10
    xs = np.linspace(-length_xy/2,length_xy/2,int(num*(length_xy/min(dists))),endpoint=True)
    ys = np.linspace(-length_xy/2,length_xy/2,int(num*(length_xy/min(dists))),endpoint=True)  
    
    if __name__ == '__main__':
    
        plate = Scalar_mask_XY(xs,ys,wavelength)
        plate.reduce_matrix=None
        plate.square(r0=(0*um,0*um),size=(length_xy/2,length_xy/2))
        amp = 1**0.5
        plate.binarize(kind='amplitude',
                    bin_level = None,
                    level0=0,
                    level1=0)

        window = Scalar_mask_XY(xs,ys,wavelength)
        window.reduce_matrix=None
        window.circle(r0=window_pos,radius = window_radius)

        window.binarize(kind='amplitude',
                    bin_level = None,
                    level0=0.0,
                    level1=amp) 
                    
        pinhole = Scalar_mask_XY(xs,ys,wavelength)
        pinhole.reduce_matrix=None
        pinhole.circle(r0=pinhole_pos,radius = pinhole_radius)  
                
        plate += window + pinhole
        print(np.shape(plate.u))
        
        print('PDI Plate',end='\r')
        plate.draw(reduce_matrix=None)
        plt.show()


        # generate the source
        u_window = Scalar_source_XY(xs,ys,wavelength)
        u_window.plane_wave(A=1,theta=0,phi=0)

        t_window = Scalar_mask_XY(xs,ys,wavelength)
        t_window.circle(r0=window_pos,radius = beam_rad,angle=0)

        u_window *= t_window
        
        # define the source
        z_window = Scalar_source_XY(xs,ys,wavelength)
        z_window.zernike_beam(A=A,r0=window_pos,radius=beam_rad,n=zernike_ns,m=zernike_ms,c_nm=zernike_cnms)

        z_window *= u_window
        
        print('Window',end='\r')
        #z_window.draw(kind='intensity',logarithm = True)
        #z_window.draw(kind='phase',logarithm = True)
        #plt.show()

        # focus on the PDI plate
        lens_window = Scalar_mask_XY(xs,ys,wavelength)
        lens_window.lens_spherical(r0=window_pos,focal=z_dist,radius=length_xy,refraction_index=1.5)

        #z_window *= lens_window
            
        #window_prop = z_window.RS(z=z_dist,verbose=True,new_field=True)
        
        window_prop = z_window
        
        #window_far = window_prop.RS(z=3*z_dist,verbose=True,new_field=True)
        
        np.save('/home/ehold13/PhD/HCIPy_simulations/output_files/window_m'+str(zernike_ms[0])+'_n'+str(zernike_ns[0])+'.npy',window_prop.u)
       
        print('Window Propagated',end='\r')
        #window_prop.draw(kind='intensity')
        #window_prop.draw(kind='phase')
        #plt.show()
        
        # pinhole beam
        
        u_pinhole = Scalar_source_XY(xs,ys,wavelength)
        u_pinhole.plane_wave(A=1,theta=0,phi=0)

        t_pinhole = Scalar_mask_XY(xs,ys,wavelength)
        t_pinhole.circle(r0=pinhole_pos,radius = beam_rad,angle=0)

        u_pinhole *= t_pinhole
        
        # define the source
        z_pinhole = Scalar_source_XY(xs,ys,wavelength)
        z_pinhole.zernike_beam(A=A,r0=pinhole_pos,radius=beam_rad,n=zernike_ns,m=zernike_ms,c_nm=zernike_cnms)

        z_pinhole *= u_pinhole
        
        print('Pinhole',end='\r')
        #z_pinhole.draw(kind='intensity',logarithm = True)
        #z_pinhole.draw(kind='phase',logarithm = True)
        #plt.show()

        # focus on the PDI plate
        #lens_pinhole = Scalar_mask_XY(xs,ys,wavelength)
        #lens_pinhole.lens_spherical(r0=pinhole_pos,focal=z_dist,radius=length_xy,refraction_index=1.5)

        #z_pinhole *= lens_pinhole
            
        #pinhole_prop = z_pinhole.RS(z=z_dist,verbose=True,new_field=True)
        
        pinhole_prop = z_pinhole
       
        print('Pinhole Propagated',end='\r')
        #pinhole_prop.draw(kind='intensity')
        #pinhole_prop.draw(kind='phase')
        #plt.show()
        
        #del lens_pinhole, lens_window, u_pinhole, u_window, t_pinhole, t_window
        
        
        window_prop *= plate
        pinhole_prop *= plate

        # beam at focal point
        focal_beam = window_prop + pinhole_prop
        
        print('Focal Point',end='\r')
        #focal_beam.draw(kind='intensity',logarithm = True)
        #focal_beam.draw(kind='phase',logarithm = True)
        #plt.show()
        
        np.save('/home/ehold13/PhD/HCIPy_simulations/output_files/focal_m'+str(zernike_ms[0])+'_n'+str(zernike_ns[0])+'.npy',focal_beam.u)
        
        del window_prop, pinhole_prop, window, pinhole

        prop_pinhole = focal_beam.RS(z=z_dist,verbose= True,new_field=True)
        
        print('Final Propagated',end='\r')
        prop_pinhole.draw(kind='intensity',logarithm=True)
        prop_pinhole.draw(kind='phase')
        
        ap = Scalar_mask_XY(xs,ys,wavelength)
        ap.circle(r0=window_pos,radius = 1.22*wavelength*z_dist/window_radius,angle=0)
        
        #prop_pinhole *= ap
        
        np.save('/home/ehold13/PhD/HCIPy_simulations/output_files/final_m'+str(zernike_ms[0])+'_n'+str(zernike_ns[0])+'.npy',prop_pinhole.u)
        
        # lens to collimate the beam
        coll_lens = Scalar_mask_XY(xs,ys,wavelength)
        coll_lens.lens_spherical(r0=(0,0),focal=-z_dist,radius=length_xy/2,refraction_index=1.5)
        
        prop_pinhole *= coll_lens
        
        plt.show()


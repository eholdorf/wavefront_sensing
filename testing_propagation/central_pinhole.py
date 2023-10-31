import numpy as np
import matplotlib.pyplot as plt
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio import um
import matplotlib.animation as animation
import aotools
from multiprocessing import Pool
import astropy.io.fits as pyfits


def propagate_pinhole_out(Z_n,Z_m,C_nm,size=80,beam_rad=5,pin_rad=0.1,amp=0.9**0.5,dist=100,wavelength=0.589,n=2000):

    Z_n, Z_m, C_nm = [Z_n], [Z_m], [abs(C_nm)]

    x,y = np.linspace(-size/2,size/2,n),np.linspace(-size/2,size/2,n)

    beam = Scalar_source_XY(x,y,wavelength=wavelength)
    beam.zernike_beam(A=1,r0=(0,0),radius=beam_rad,n=Z_n,m=Z_m,c_nm=C_nm)

    beam_ap = Scalar_mask_XY(x,y,wavelength=wavelength)
    beam_ap.circle((0,0),radius = beam_rad, angle =0)

    beam.u = np.where(beam_ap.u==0,0,beam.u)

    pinhole = Scalar_mask_XY(x,y,wavelength=wavelength)
    pinhole.circle(r0=(0,0),radius=pin_rad,angle = 0)

    pinhole.u = np.where(pinhole.u==0,amp,1)

    #pinhole.u = np.fft.fftshift((pinhole.u+1)*0.5)

    #beam.u = np.fft.fft2(beam.u)

    beam *= pinhole

    beam.fast = False

    prop_beam = beam.RS(z=dist,new_field = True)

    # plt.figure()
    # plt.imshow(abs(prop_beam.u)**2)
    # plt.show()

    return abs(prop_beam.u)**2

if False:
    fig,ax= plt.subplots(4,4,figsize=[8,8])
    ax = ax.flatten()

    amplitudes = np.linspace(-1,1,10,endpoint=False).tolist()
    amplitudes.extend(np.linspace(1,-1,11).tolist())



    def func(i):
        a = amplitudes[i-1]
        Z_ns = [0,1,1,2,2,2,3,3,3,3,4,4,4,4,4,5]
        Z_ms = [0,1,-1,0,2,-2,1,-1,3,-3,0,2,-2,4,-4,-1]

        Z = zip(Z_ns,Z_ms,[a]*len(Z_ns))

        with Pool(20) as p:
                prop = p.starmap(propagate,Z)
        
        for i, img in enumerate(prop):
            ax[i].imshow(img)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].set_title(f"mode {i:d}, $a_{{{i:d}}}$: {a:4.2f}")

        return ax

    ani = animation.FuncAnimation(fig,func,len(amplitudes))
    ani.save('multi_amps.gif')
    plt.show()


def propagate(zern_current,a,frac,max_zerns = 16, wavelength=0.589,n=2**6, path ="/home/ehold13/PhD/testing_propagation/",see_plots=False):
    pup_width = n # number of pixels across the pupil
    fp_oversamp = 2**2 # number of pixels across the diffraction-limited FWHM
    pinhole_size = 1 # measured in multiples of FWHM, i.e., lambda/D
    wavelength = wavelength # in microns

    # generate the zernike functions
    # make a padded pupil so the focal plane smapled well
    padded_pupil = np.zeros((pup_width*fp_oversamp,pup_width*fp_oversamp),dtype=complex)
    zerns = aotools.zernikeArray(max_zerns,pup_width,norm="rms")
    # borrow piston mode as the pupil function
    pup = zerns[0]

    phi = zerns[zern_current] * a 
    beam = pup * np.exp(1j*2*np.pi/wavelength*phi)

    plt.figure()
    plt.imshow(abs(beam)**2)
    plt.title('Beam')
    if not see_plots:
        plt.close()
    


    mid_int = int(pup_width*fp_oversamp/2)
    padded_pupil[mid_int - int(pup_width/2):mid_int + int(pup_width/2),
                 mid_int - int(pup_width/2):mid_int + int(pup_width/2)] = beam

    plt.figure()
    plt.imshow(np.angle(padded_pupil))
    plt.title('Pupil')
    #plt.savefig(path + "pupil.png")
    if not see_plots:
        plt.close()

    # now need to propagate from the pupil plane to the focal plane
    focal = aotools.opticalpropagation.lensAgainst(padded_pupil, wavelength*1e-6, 1/(fp_oversamp*pup_width), 0.1)

    plt.figure()
    plt.imshow(abs(focal)**2)
    plt.title('Focal')
    #plt.savefig(path + "focal.png")
    if not see_plots:
        plt.close()

    # make the pinhole mask
    mask = aotools.circle(fp_oversamp*pinhole_size,fp_oversamp*pup_width)
    mask = np.where(mask == 0,frac,mask)

    plt.figure()
    plt.imshow(mask)
    plt.title('Mask')
    plt.colorbar()
    #plt.savefig(path + "mask.png")
    if not see_plots:
        plt.close()

    # pass the light through the pinhole
    focal *= mask

    plt.figure()
    plt.imshow(abs(focal)**2)
    plt.title('Focal with mask')
    #plt.savefig(path+"pinhole_and_beam.png")
    if not see_plots:
        plt.close()

    # propagate to the camera plane
    camera = aotools.opticalpropagation.oneStepFresnel(focal,wavelength*1e-6,2.44*wavelength*1e-6/(pup_width),0.1)

    plt.figure()
    plt.imshow(abs(camera)**2)
    plt.title('Camera')
    #plt.savefig(path+"camera.png")

    if not see_plots:
        plt.close()
    
    mid_int += 1
    camera = camera[mid_int - int(pup_width/2):mid_int + int(pup_width/2),
                 mid_int - int(pup_width/2):mid_int + int(pup_width/2)]


    plt.figure()
    plt.imshow(abs(camera)**2)
    plt.title('Camera cropped')

    if not see_plots:
        plt.close()
    
    if see_plots:
        plt.show()
    
    return abs(camera)**2

if True:
    fig,ax= plt.subplots(4,4,figsize=[8,8])
    ax = ax.flatten()

    amplitudes = np.linspace(-2,2,20,endpoint=False).tolist()
    amplitudes.extend(np.linspace(2,-2,21).tolist())

    def func(i):
        path ="/home/ehold13/PhD/testing_propagation/"
        fracs = np.linspace(0,1,11,endpoint=True)
        a = amplitudes[i-1]
        zs = np.arange(0,16,1)
        fracs = np.linspace(0,1,11,endpoint=True)
        for frac in fracs:
            Z = zip(zs,[a]*len(zs),[frac]*len(zs))

            with Pool(16) as p:
                    prop = p.starmap(propagate,Z)
            
            for i, img in enumerate(prop):
                ax[i].imshow(img)
                ax[i].set_xticks([])
                ax[i].set_yticks([])
                ax[i].set_title(f"mode {i:d}, $a_{{{i:d}}}$: {a:4.2f}")

                # save as a fits file
                hdu = pyfits.PrimaryHDU(np.transpose(img))
                hdul = pyfits.HDUList([hdu])
                hdul.writeto(path+'/data_comp_sims/erin_intensity'+str(i)+'_'+str(np.round(a,1))+'_'+str(np.round(frac,1))+'.fits',overwrite=True)   

        return ax
    path ="/home/ehold13/PhD/testing_propagation/"
    ani = animation.FuncAnimation(fig,func,len(amplitudes))
    ani.save(path+'multi_amps_new.gif', fps = 2)
    #plt.show()   

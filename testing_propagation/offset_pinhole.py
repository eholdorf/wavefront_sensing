import numpy as np
import matplotlib.pyplot as plt
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio import um
import matplotlib.animation as animation
import aotools
from multiprocessing import Pool
import astropy.io.fits as pyfits


def propagate(zern_current,a,max_zerns = 16, wavelength=0.589,n=2**6, path ="/home/ehold13/PhD/testing_propagation/",see_plots=False):
    pup_width = n # number of pixels across the pupil
    fp_oversamp = 2**2 # number of pixels across the diffraction-limited FWHM
    pinhole_size = 0.5/fp_oversamp**2  # measured in multiples of FWHM, i.e., lambda/D
    wavelength = wavelength # in microns

    # generate the zernike functions
    # make a padded pupil so the focal plane smapled well
    padded_pupil = np.zeros((pup_width*fp_oversamp,pup_width*fp_oversamp),dtype=complex)
    zerns = aotools.zernikeArray(max_zerns,pup_width,norm="rms")
    # borrow piston mode as the pupil function
    pup = zerns[0]
s
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
    if not see_plots:
        plt.close()

    # now need to propagate from the pupil plane to the focal plane
    focal = aotools.opticalpropagation.lensAgainst(padded_pupil, wavelength*1e-6, 1/(fp_oversamp*pup_width), 10)

    plt.figure()
    plt.imshow(abs(focal)**2)
    plt.title('Focal')
    if not see_plots:
        plt.close()

    # make the pinhole mask
    mask = aotools.circle(pinhole_size*pup_width*fp_oversamp/2,pup_width*fp_oversamp,circle_centre=(0,0))
    mask = np.where(mask == 0,(fp_oversamp*pinhole_size),mask)

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
    if not see_plots:
        plt.close()

    # propagate to the camera plane
    camera = aotools.opticalpropagation.oneStepFresnel(focal,wavelength*1e-6,2.44*wavelength*1e-6/(fp_oversamp*pup_width),0.1)

    plt.figure()
    plt.imshow(abs(camera)**2)
    plt.title('Camera')

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
    
propagate(0,0.5,see_plots=True)
if False:
    fig,ax= plt.subplots(4,4,figsize=[8,8])
    ax = ax.flatten()

    amplitudes = np.linspace(-2,2,20,endpoint=False).tolist()
    amplitudes.extend(np.linspace(2,-2,21).tolist())

    def func(i):
        path ="/home/ehold13/PhD/testing_propagation/"
        a = amplitudes[i-1]
        zs = np.arange(0,16,1)

        Z = zip(zs,[a]*len(zs))

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
            hdul.writeto(path+'/data_comp_sims/erin_intensity'+str(i)+'_'+str(np.round(a,1))+'.fits',overwrite=True)   

        return ax
    path ="/home/ehold13/PhD/testing_propagation/"
    ani = animation.FuncAnimation(fig,func,len(amplitudes))
    ani.save(path+'multi_amps_new.gif', fps = 2)
    #plt.show()   

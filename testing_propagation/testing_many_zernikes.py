import matplotlib.pyplot as plt#; plt.ion()
import aotools
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import numpy as np
import scipy.signal
from multiprocessing import Pool
import astropy.io.fits as pyfits

def propagate(frac):
    path ="/home/ehold13/PhD/testing_propagation/"
    pup_width = 2**6 # number of pixels across the pupil
    fp_oversamp = 2**2 # number of pixels across the diffraction-limited FWHM
    pinhole_size = 1 # measured in multiples of FWHM, i.e., lambda/D
    max_zerns = 16 # maximum number of zernikes to consider
    wavelength = 0.589 # in microns

    # generate the zernike functions
    zerns = aotools.zernikeArray(max_zerns,pup_width,norm="rms")
    # borrow piston mode as the pupil function
    pup = zerns[0]

    # pinhole mask function        
    mask = aotools.circle(fp_oversamp*pinhole_size,fp_oversamp*pup_width)
    # mask amplitude modulation function, (half of input light plus half
    # of masked light). fftshift to sort the quadrants properly for fft-ing
    mod_func = np.fft.fftshift((mask+1)*0.5)


    # subplots of response per mode
    fig,ax = plt.subplots(4,4,figsize=[8,8])
    ax = ax.flatten()

    # amplitude of mode
    a = 0.0
    def update(a):
        """Update the figure based on the amplitude value"""
        # loop over each mode
        for i,axi in enumerate(ax):
            # overly general way of pushing one zernike mode at a time:
            x = np.zeros(max_zerns)
            x[i] = a
            
            # build the phase function, phi, as linear combination of zernikes
            # though in this case only one is non-zero so this is overkill.
            phi = np.einsum("ijk,i->jk",zerns,x)
            
            # pupil-plane complex amplitude is pupil amplitude with a phase delay:
            psi = pup * np.exp(1j*2*np.pi/wavelength*(phi))

            # focal plane complex amplitude is FFT of pupil plane, modulated by
            # the mask modulation function (includes interference of two waves)
            fp_amp = np.fft.fft2(psi,s=[pup_width*fp_oversamp]*2)*mod_func

            # sensor-plane intensity is inverse FFT of FP amplitude, cropped and squared
            intensity = np.abs(np.fft.ifft2(fp_amp)[:pup_width,:pup_width])**2
            # save as a fits file
            path ="/home/ehold13/PhD/testing_propagation/"
            hdu = pyfits.PrimaryHDU(np.transpose(intensity))
            hdul = pyfits.HDUList([hdu])
            hdul.writeto(path+'data_comp_sims/jesse_intensity'+str(i)+'_'+str(np.round(a,1))+'.fits',overwrite=True)
            
            # plot it
            if len(axi.images)==0:
                # first time, format stuff
                axi.imshow(intensity)  
                axi.set_xticks([])
                axi.set_yticks([])
            axi.images[0].set_data(intensity)  
            # maximise colour range for visibility, but this obscures the limited 
            # signal in large aberration cases.
            axi.images[0].set_clim([intensity.min(),intensity.max()])
            axi.set_title(f"mode {i:d}, $a_{{{i:d}}}$: {a:4.2f}")
        return ax

    # run once to get figure initialised
    update(a)
    plt.tight_layout()

    # make animation for a "sawtooth" input of zernikes
    zernike_amplitude = tqdm(np.r_[np.linspace(-2,2,20,endpoint=False),np.linspace(2,-2,21)])
    anim = FuncAnimation(fig,update,zernike_amplitude)
    # run and save animation
    anim.save(path+f"test_{pup_width:04d}_{fp_oversamp:04d}.gif",fps=2)
    plt.show()

propagate(0.5)

# with Pool(20) as p:
#     prop = p.map(propagate,np.linspace(0,1,11,endpoint=True))
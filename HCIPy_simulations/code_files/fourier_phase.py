import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.fft
from skimage.restoration import unwrap_phase
import cmath

orig = np.load('/home/ehold13/PhD/HCIPy_simulations/output_files/initial_m-2_n2.npy') 
pdi = np.load('/home/ehold13/PhD/HCIPy_simulations/output_files/final_m-2_n2.npy') 
spherical = np.load('/home/ehold13/PhD/HCIPy_simulations/output_files/spherical_m-2_n2.npy')

if True:

    # take the fourier transform of pdi intensity

    I_pdi = np.abs(pdi)**2
    
    fft = scipy.fft.fftshift(scipy.fft.fft(I_pdi,axis=1))

    if True:
        plt.figure()
        plt.plot(abs(np.sum(fft,0)),'k')


        plt.figure()
        plt.imshow(np.abs(fft)**2,norm=LogNorm())
        plt.show()
    
    first_init = 1215
    last_init = 1245
    fft[:,:first_init] = 0
    fft[:,last_init:]=0
        
    fft_shifted = np.zeros_like(fft)
    
    mid_int = int(len(fft_shifted)/2)
    len_phase = last_init-first_init
    
    fft_shifted[:,int(mid_int-len_phase/2):int(mid_int+len_phase/2)] = fft[:,first_init:last_init]
    
    fft_shifted_ifft = scipy.fft.ifftshift(fft)
    

    ifft = scipy.fft.ifft(fft_shifted_ifft,axis=1)
    

    ifft = np.imag(np.log(ifft))
    
    np.save('/home/ehold13/PhD/HCIPy_simulations/output_files/left_background_phase.npy',ifft)
    

    plt.figure()
    plt.imshow(ifft)
    
    plt.figure()
    plt.imshow(np.angle(pdi))

    plt.figure()
    plt.imshow(np.angle(pdi)-np.angle(ifft))
    
    first_init = 1155
    last_init = 1185
    fft[:,:first_init] = 0
    fft[:,last_init:]=0
        
    fft_shifted = np.zeros_like(fft)
    
    mid_int = int(len(fft_shifted)/2)
    len_phase = last_init-first_init
    
    fft_shifted[:,int(mid_int-len_phase/2):int(mid_int+len_phase/2)] = fft[:,first_init:last_init]
    
    fft_shifted_ifft = scipy.fft.ifftshift(fft)
    

    ifft = scipy.fft.ifft(fft_shifted_ifft,axis=1)
    

    ifft = np.imag(np.log(ifft))
    

    plt.figure()
    plt.imshow(ifft)
    
    plt.figure()
    plt.imshow(np.angle(pdi)-np.angle(ifft))
    
    plt.show()






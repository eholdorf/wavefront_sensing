import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.fft
from skimage.restoration import unwrap_phase
import cmath

# helper functions


orig = np.load('/home/ehold13/PhD/HCIPy_simulations/output_files/initial_m-2_n2.npy') 
pdi = np.load('/home/ehold13/PhD/HCIPy_simulations/output_files/final_m-2_n2.npy') 
spherical = np.load('/home/ehold13/PhD/HCIPy_simulations/output_files/spherical_m-2_n2.npy')

diff = pdi-spherical

l = np.angle(diff)-np.angle(orig)
l = np.where(abs(l)>2,0,l)
print(np.var(l)/(2*np.pi) *589)
plt.figure()
plt.imshow(l)
plt.colorbar(label='Phase Error (nm)')
plt.show()

if False:

    # take the fourier transform of pdi intensity

    I_pdi = np.abs(pdi)**2
    
    fft = scipy.fft.fftshift(scipy.fft.fft(I_pdi,axis=1))

    if True:
        plt.figure()
        plt.plot(abs(np.sum(fft,0)),'k')


        plt.figure()
        plt.imshow(np.abs(fft)**2,norm=LogNorm())
        #plt.show()
    
    first_init = 0
    last_init = 1192
    fft[:,:first_init] = 0
    fft[:,last_init:]=0
        
    fft_shifted = np.zeros_like(fft)
    
    mid_int = int(len(fft_shifted)/2)
    len_phase = last_init-first_init
    
    fft_shifted[:,int(mid_int-len_phase/2):int(mid_int+len_phase/2)] = fft[:,first_init:last_init]
    
    plt.figure()
    plt.plot(abs(np.sum(fft_shifted,0)),'k')
    plt.show()
    
    fft_shifted_ifft = scipy.fft.ifftshift(fft)
    

    ifft = scipy.fft.ifft(fft_shifted_ifft,axis=1)
    

    ifft = np.imag(np.log(ifft))
    
    np.save('/home/ehold13/PhD/HCIPy_simulations/output_files/left_background_phase.npy',ifft)
    

    plt.figure()
    plt.imshow(ifft)
    plt.title('Recovered Phase')
    
    plt.figure()
    plt.imshow(np.angle(pdi))
    plt.title('True Phase')

    plt.figure()
    plt.imshow(-np.angle(pdi)+np.angle(spherical))
    plt.title('True-Spherical')
    
    plt.figure()
    plt.imshow(np.angle(orig))
    plt.title('Original')
    
    plt.show()






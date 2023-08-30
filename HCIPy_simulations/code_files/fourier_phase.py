import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.fft
from skimage.restoration import unwrap_phase

orig = np.load('/home/ehold13/PhD/HCIPy_simulations/output_files/initial_m1_n1.npy')
pdi = np.load('/home/ehold13/PhD/HCIPy_simulations/output_files/final_m1_n1.npy') 
spherical = np.load('/home/ehold13/PhD/HCIPy_simulations/output_files/spherical_m1_n1.npy')


plt.figure()
plt.imshow(np.angle(orig))

plt.figure()
plt.imshow(np.abs(pdi)**2)

plt.figure()
plt.imshow(np.abs(spherical)**2)
plt.show()

if True:

    # take the fourier transform of pdi intensity

    I_pdi = np.abs(pdi)**2

    fft = scipy.fft.fftshift(scipy.fft.fft(I_pdi,axis=1))

    if True:
        plt.figure()
        plt.plot(abs(np.sum(fft,0)),'k')


        plt.figure()
        plt.imshow(np.abs(fft))
        plt.show()

    fft[:,:1330] = 0
    fft[:,1530:]=0
    
    fft_shifted = scipy.fft.ifftshift(fft)
    

    ifft = scipy.fft.ifft(fft_shifted,axis=1)

    ifft = np.imag(np.log(ifft))

    plt.figure()
    plt.imshow(ifft)

    plt.figure()
    plt.imshow(np.angle(pdi))

    plt.figure()
    plt.imshow(np.angle(orig))
    plt.show()






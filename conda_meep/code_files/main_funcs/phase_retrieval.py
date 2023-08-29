import numpy as np
import scipy.fft
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cmath
from skimage.restoration import unwrap_phase

def make_circular(data):

    r_x,r_y = [np.shape(data)[0]/2,np.shape(data)[1]/2]
    m_x, m_y = [np.shape(data)[0]/2,np.shape(data)[1]/2] 
    for i in range(np.shape(data)[0]):
        for j in range(np.shape(data)[1]):
            if (i-m_x)**2/(r_x)**2 + (j-m_y)**2/(r_y)**2 > 1:
                data[i,j] = 0
    return data
    

if __name__=='__main__':

    # import the data to calculate the phase
    data = (np.load('/home/ehold13/PhD/conda_meep/output_files/ez_zernike_r_0.25_bw_2.945.npy')
    +np.load('/home/ehold13/PhD/conda_meep/output_files/ey_zernike_r_0.25_bw_2.945.npy'))
    wavelength = 0.589

    final_image = data[-10,:,:]
    
    final_image = make_circular(final_image)
    
    intensity = np.abs(final_image)**2
    
    plt.figure()
    plt.imshow(intensity)

    plt.figure()
    plt.imshow(np.angle(final_image))
    plt.show()
    
    
    fft = scipy.fft.fftshift(scipy.fft.fft2(intensity))

    plt.figure()
    plt.plot(abs(np.sum(fft,0)),'k.')
  
    
    plt.figure()
    plt.imshow(np.abs(fft), norm=LogNorm())
    plt.show()


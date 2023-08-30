import numpy as np
import math
import astropy.units as un
import matplotlib.pyplot as plt
import scipy.fft as sci
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
import scipy.signal
import scipy.ndimage
import scipy.fft


# generate Zernike wave
def zernike(x,y,m,n,c,A=1):
    if m < 0:
        sign = True
    else: 
        sign = False
    m = np.abs(m)

    Z = np.zeros((len(x),len(y)))
    R = np.zeros((len(x),len(y)))
    
    x,y = np.meshgrid(x,y,indexing='ij')

    rho = np.sqrt(x**2+y**2)/np.max(x)

    theta = np.arctan2(x,y)
    k = 0
    while k <= int((n-m)/2):
        numerator = (-1)**k * math.factorial(np.abs(n-k)) * rho**(n-2*k)
        denominator = math.factorial(abs(k)) * math.factorial(abs(int((n+m)/2)-k))\
                                * math.factorial(np.abs(int((n-m)/2)-k))
        R += numerator/denominator
        k += 1
    
    if sign == False :
        kron = 0
        if m==n:
            kron = 1
        N = np.sqrt((2*n+2)/(1+kron))
        Z = c*N*np.cos(m * theta) * R
    else:
        kron = 0
        if m==n:
            kron = 1
        N = np.sqrt((2*n+2)/(1+kron))
        Z = c*N*np.sin(m * theta) * R

    Z[rho>1] = 0  
    A = np.full_like(Z,A)
    A[rho>1] = 0      

    return A*np.exp(1j*np.real(Z))
#generate a plane wave
def plane_wave(A, theta, phi, x, y, wavelength, z0 = 0):
    '''
    Defining the field for an incoming plane wave.
    -------
    Inputs:
    -------
        A (float) - amplitude of the wave \n
        theta (float) - x axis rotation (unit: radians) \n
        phi (float) - y axis rotation (unit: radians) \n
        x (np.array) - list of x positions (unit: microns) \n
        y (np.array) - list of y positions (unit: microns) \n
        wavelength (float) - the wavelength of the light (unit: microns) \n
        z0 (float) - constant phase shift (default 0) (unit: microns) \n

    --------
    Outputs:
    --------
        u (np.array) - the field on the plane wave \n
    '''

    X,Y = np.meshgrid(x,y,indexing='ij')

    k = 2*np.pi/wavelength

    # translate into polar co-ords
    x_comp = X*np.sin(theta)*np.cos(phi)
    y_comp = Y*np.sin(theta)*np.sin(phi)
    z_comp = z0*np.cos(theta)

    u = A*np.exp(1j*k*(x_comp+y_comp+z_comp))

    return u
# apply the beam propagation method
def FFT_BPM(x,y,z,w,mask,wavelength, other = -1):
    '''
    Based on : https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1522306&tag=1
    https://www.researchgate.net/publication/236246940_Beam_Propagation_Method_Based_on_Fast_Fourier_Transform_and_Finite_Difference_Schemes_and_Its_Application_to_Optical_Diffraction_Grating
    '''
    # set the number of points and the stepping size in each direction
    delta_z = z[1]-z[0]
    num_z = len(z)
    num_x = len(x)
    range_x = x[-1]-x[0]
    num_y = len(y)
    range_y = y[-1]-y[0]

    # calculate the wave number
    k0 = 2 * np.pi/wavelength

    # change the input to be in a meshgrid
    #X,Y,Z = np.meshgrid(x,y,z,indexing ='ij')

    # set up the matrix to put the field values into
    u = np.zeros((num_x,num_y,num_z), dtype=complex)
    # set the initial value equal to the incoming field
    u[:,:,0] = w

    # generate the wave numbers in the x and y dimensions, on a field that 
    # spaced in 1's and includes 0
    kx1 = np.linspace(0,int(num_x/2)+1,int(num_x/2))
    kx2 = np.linspace(-int(num_x/2),-1,int(num_x/2))
    kx = 2*np.pi/range_x *np.concatenate((kx1,kx2))

    ky1 = np.linspace(0,int(num_y/2)+1,int(num_y/2))
    ky2 = np.linspace(-int(num_y/2),-1,int(num_y/2))
    ky = 2*np.pi/range_y * np.concatenate((ky1,ky2))

    # create a grid of the x and y pixels
    KX, KY = np.meshgrid(kx,ky)

    if False:
        T = np.ones_like(w, dtype=complex)
        f = z[other]
        X1,Y1 = np.meshgrid(x,y,indexing='xy')
        T = np.exp(-1j * (X1**2+Y1**2) * k0/(2*f))
        u[:,:,1] = u[:,:,0]*T

    for i in range(1,len(z)):
        print("BPM: {}/{}".format(i,len(z)), end="\r")
        if False: #i == other:
            # generate the pinhole mask
            mask = np.full((len(x),len(y)),1.51630,dtype=complex)
            r_pix_pinhole_x = int((r_sphere/(2*xs/num_x)))
            r_pix_pinhole_y = int((r_sphere/(2*ys/num_y)))
            
            if r_pix_pinhole_x <=0 or r_pix_pinhole_y <=0 :
                raise Exception("Grid Not Fine Enough, {}".format((r_sphere/(2*xs/num_x))))

            # for the points inside the pinhole, set the mask equal to 1.0003 (air)
            for j in range(int(n/2)-r_pix_pinhole_y, int(n/2)+r_pix_pinhole_y):
                for p in range(int(n/2)-r_pix_pinhole_x,int(n/2)+r_pix_pinhole_x):
                    l = (x[p])**2/(r_sphere)**2 + (y[j])**2/(r_sphere)**2
                    if l <= 1:
                        mask[p,j] = 1.000
        else:
            mask = np.full((len(x),len(y)),1.000,dtype=complex)
        # take the fourier transform of the step before
        fft = scipy.fft.fft2(u[:,:,i-1])
        fft *= np.exp(1j* delta_z*(KX**2 + KY**2)/(2*k0)) 
        ifft = scipy.fft.ifft2(fft)
        ifft *= np.exp(-1j*mask*k0*delta_z) 
        u[:,:,i] = ifft

        # absorbing boundary condition
        u[:10,:,i] = 0
        u[-10:,:,i] = 0
        u[:,:10,i] = 0
        u[:,-10:,i] = 0

    return u
#generate a spherical wave
def spherical_wave(A, x, y, r0, wavelength,theta= 0,phi=0, z0 = 0):
        X,Y = np.meshgrid(x,y,indexing='ij')
        X = X*np.sin(theta)*np.cos(phi)
        Y = Y*np.sin(theta)*np.sin(phi)
        z0 = z0*np.cos(theta)
        x0,y0 = r0
        k = 2 * np.pi/wavelength
        Rz = np.sqrt((X-x0)**2 + (Y-y0)**2 + z0**2)

        u = A * np.exp(-1j*np.sign(z0)*k*Rz)/Rz

        return u

if __name__ == '__main__':
    # set up x and y grid
    n = 2**9
    r_sphere = 0.25
    wavelength = 0.589

    # choose how far to see in each x, y, z direction
    xs = 1 
    ys = 1 
    zs = 10

    # initialise x,y grids
    x = np.linspace(-xs, xs, n)
    y = np.linspace(-ys, ys, n)

    z_i = 10 # index to place the mask
    z_len = 1 # number of iterations to include the mask

    # generate the pinhole mask
    mask = np.full((len(x),len(y)),1.000,dtype=complex)

    r_pix_pinhole_x = int((0.25/(2*xs/n)))
    r_pix_pinhole_y = int((0.25/(2*ys/n)))
    
    if r_pix_pinhole_x <=0 or r_pix_pinhole_y <=0 :
        raise Exception("Grid Not Fine Enough, current dx: {}".format((r_sphere/(2*xs/n))))

    # Zernike Polynomials to test (m,n,c)           
    mncs = [(-1,1,1)]
    n_z = 2**1 # as each run has 2**7 steps
    dz = zs/(n_z*2**7)
    for mnc in mncs:
        m,n_,c = mnc
        # generate the zernike components of the original wave 
        extent = 2*r_pix_pinhole_x
        w = zernike(x[int(n/2)-extent:int(n/2)+extent],y[int(n/2)-extent:int(n/2)+extent],m,n_,c,A=1)
        for k in range(n_z):
            print("                      Iteration: {}/{}".format(k,n_z-1), end="\r")
            
            k=128*k
            # set the inital wave to be the Zernike polynomial
            u0=np.zeros((len(x),len(y)),dtype = complex)
            # for the initial iteration, make the first frame Zernike
            if k==0:
                u0[int(n/2)-extent:int(n/2)+extent,int(n/2)-extent:int(n/2)+extent] = w
            # otherwise make it the last frame from the previous iteration
            else:
                u0 = u_out[:,:,-1]
            if k==0:
                z = np.linspace(k*dz,(k+127)*dz,128)
                #z = np.linspace(0,128*dz,128)
                u_out = FFT_BPM(x,y,z,w = u0,wavelength=wavelength, mask=mask,other=z_i)
            else:
                z = np.linspace(k*dz,(k+127)*dz,128)
                #z = np.linspace(0,128*dz,128)
                u_out = FFT_BPM(x,y,z,w = u0,wavelength=wavelength, mask=mask)
            if True:
                def func(i):
                    plt.clf()
                    print("Making Animation: {}/{}".format(i,128), end="\r")
                    a = plt.imshow(np.abs(u_out[:,i,:])**2)
                    plt.colorbar()
                    return a

                fig = plt.figure()
                anim = animation.FuncAnimation(fig,func,frames=128, interval = 100,repeat=True)
                anim.save('/home/ehold13/PhD/WFS_files/convolution_m'+str(m)+'_n'+str(n_)+'_c'+str(c)+'_iter'+str(k/128)+'.gif')
                
                def func2(i):
                    plt.clf()
                    print("Making Animation: {}/{}".format(i,128), end="\r")
                    a = plt.imshow(np.angle(u_out[:,i,:]))
                    plt.colorbar()
                    return a

                fig = plt.figure()
                anim = animation.FuncAnimation(fig,func2,frames=128, interval = 100,repeat=True)
                anim.save('/home/ehold13/PhD/WFS_files/angle_m'+str(m)+'_n'+str(n_)+'_c'+str(c)+'_iter'+str(k/128)+'.gif')
                


    if False:
        #set up x,y,z grid
        num_points = 2**9
        r_sphere = (0.25*un.micron).value # microns
        wavelength = (589*un.nm).to(un.micron).value # microns

        # distance in microns
        x_s = 50
        y_s = 50
        z_s = 500000 #500 mm

        # generate the grid of points
        # use: https://opg.optica.org/josa/fulltext.cfm?uri=josa-71-7-803&id=58040
        # to choose the delta z spacing, dz < 25 micron error < 10**-4
        x0 = np.linspace(-x_s, x_s, num_points)
        y0 = np.linspace(-y_s, y_s, num_points)
        z0 = np.linspace(0,z_s,num_points)

        # generate the pinhole mask
        mask = np.full((len(x0),len(y0),len(z0)),1.0003,dtype=complex)

        r_pix_pinhole_x = int((r_sphere/(2*x_s/num_points)))
        r_pix_pinhole_y = int((r_sphere/(2*y_s/num_points)))
        
        if r_pix_pinhole_x <=0 or r_pix_pinhole_y <=0 :
            raise Exception("Grid Not Fine Enough")

        plate_thickness = 1

        # location of the mask is in the middle of the mask
        mid_int = int(num_points/2)

        # set the points in the rectangle of mask equal to the mask refractive index
        mask[:,:,(mid_int-plate_thickness):(mid_int+plate_thickness)] = 1.51630

        # for the points inside the pinhole, set the mask equal to 1.0003
        for z in range(mid_int-plate_thickness,mid_int+plate_thickness):
            for y in range(int(num_points/2)-r_pix_pinhole_y, int(num_points/2)+r_pix_pinhole_y):
                for x in range(int(num_points/2)-r_pix_pinhole_x,int(num_points/2)+r_pix_pinhole_x):
                    l = (x0[x])**2/(r_sphere)**2 + (y0[y])**2/(r_sphere)**2
                    if l <= 1:
                        mask[x,y,z] = 1.0003

        # cycle through the first 6 Zernike polynomials to see the intensities
        #mns = [(0,0),(1,1),(-1,1),(0,2),(-2,2),(2,2)]
        mncs = [(-1,1,1)]
        for mnc in mncs:
            m,n,c = mnc
            # generate the zernike components of the original wave 
            move = 50
            w = zernike(x0[int(num_points/2)-move:int(num_points/2)+move],y0[int(num_points/2)-move:int(num_points/2)+move],m,n,c,A=1)
            # set the inital wave to be the Zernike polynomial
            u0=np.zeros((len(x0),len(y0)),dtype = complex)
            u0[int(num_points/2)-move:int(num_points/2)+move,int(num_points/2)-move:int(num_points/2)+move] = w

            u_out = FFT_BPM(x0,y0,z0,w = u0,wavelength=wavelength, mask=mask)

            fig,ax = plt.subplots(2,2)
            ax[0,0].imshow(np.angle(u_out[:,:,0]))
            ax[0,0].set_title('Initial Phase: Z$_{}^{}$'.format(str({n}),str({m})))
            
            ax[0,1].imshow(np.abs(u_out[:,:,0])**2)
            ax[0,1].set_title('Initial Intensity: Z$_{}^{}$'.format(str({n}),str({m})))

            ax[1,0].imshow(np.angle(u_out[:,:,-1]))
            ax[1,0].set_title('Final Phase: Z$_{}^{}$'.format(str({n}),str({m})))

            ax[1,1].imshow(np.abs(u_out[:,:,-1])**2)
            ax[1,1].set_title('Final Intensity: Z$_{}^{}$'.format(str({n}),str({m})))

            plt.savefig('/home/ehold13/PhD/WFS_files/Z'+str(m)+str(n)+str(c)+'.pdf')

            # animate how the intensity changes in the z direction (direction of propagation)
            if True:
                def func(i):
                    plt.clf()
                    print("Making Animation: {}/{}".format(i,int(num_points/2)), end="\r")
                    a = plt.imshow(np.abs(u_out[:,:,2*i])**2)
                    plt.colorbar()
                    return a
                def func2(i):
                    plt.clf()
                    print("Making Animation: {}/{}".format(i,int(num_points/2)), end="\r")
                    a = plt.imshow(np.angle(u_out[:,:,2*i]))
                    plt.colorbar()
                    return a
                fig = plt.figure()
                anim = animation.FuncAnimation(fig,func,frames=int(num_points/2), interval = 0.001,repeat=True)
                anim.save('/home/ehold13/PhD/WFS_files/convolution_m'+str(m)+'_n'+str(n)+'_c'+str(c)+'.gif')

                fig = plt.figure()
                anim = animation.FuncAnimation(fig,func,frames=int(num_points/2), interval = 0.001,repeat=True)
                anim.save('/home/ehold13/PhD/WFS_files/angle_m'+str(m)+'_n'+str(n)+'_c'+str(c)+'.gif')



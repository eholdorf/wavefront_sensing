# import statements
import numpy as np
import scipy.fft
import matplotlib.pyplot as plt
import math
from matplotlib.colors import LogNorm
from skimage.restoration import unwrap_phase
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio import um
import matplotlib.animation as animation
import poppy
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from multiprocessing import Pool
import time
#helper functions
def linear_line(x,m,c):
    return m*x+c
if False:
    dists = np.linspace(100,200,50)
    tilts = []
    printing = False
    for k,p in enumerate(dists):
        print(end='\x1b[2K')
        print("Progress, Initialising... Dist {}/".format(k+1)+str(len(dists)),end='\r')
        n = 5000
        # object wave
        d = 100
        x = np.linspace(-d/2,d/2,n)
        y = np.linspace(-d/2,d/2,n)

        ms = [-2]
        ns = [2]
        cnms = [0.5]

        rad = 0.5
        
        ref_rad = 0.25
        ref_loc = p*um
        dist = 4000*um
        print(end='\x1b[2K')
        print("Progress, Generating Ref Beam... Dist {}/".format(k+1)+str(len(dists)),end='\r')
        ref1 = Scalar_source_XY(x,y,0.589)
        ref1.zernike_beam(A=1,r0=(ref_loc,0),radius=ref_rad,n=[4],m=[0],c_nm=[1])

        aperture = Scalar_mask_XY(x,y,0.589)
        aperture.circle(r0=(ref_loc,0),radius = ref_rad)

        ref1 *= aperture
        print(end='\x1b[2K')
        print("Progress, Generating Object Beam... Dist {}/".format(k+1)+str(len(dists)),end='\r')
        obj_middle = Scalar_source_XY(x,y,0.589)
        obj_middle.zernike_beam(A=0.5,r0=(0,0),radius=rad,n=ns,m=ms,c_nm=cnms)

        aperture = Scalar_mask_XY(x,y,0.589)
        aperture.circle(r0=(0,0),radius = rad)

        obj_middle *= aperture

        inter = ref1 + obj_middle
        print(end='\x1b[2K')
        print("Progress, Propagating Beam... Dist {}/".format(k+1)+str(len(dists)),end='\r')

        prop = inter.RS(z=dist,new_field=True)
        print(end='\x1b[2K')
        print("Progress, Beam Propagated... Dist {}/".format(k+1)+str(len(dists)),end='\r')
        ap_dist = 2*(1.22*0.589*dist/(2*rad))
        aperture = Scalar_mask_XY(x,y,0.589)
        aperture.circle(r0=(0,0),radius = ap_dist)

        intf = np.abs(prop.u)**2

        if printing:
            plt.figure()
            plt.imshow(intf)
            plt.show()


        fft = scipy.fft.fftshift( scipy.fft.fft(intf,axis=1) )

        m = scipy.signal.find_peaks(np.sum(abs(fft),axis=0),0.01*max(np.sum(abs(fft),axis=0)))
        m = min(m[0])
        w = scipy.signal.peak_widths(np.sum(abs(fft),axis=0),[m])

        f = int(w[2][0])-0
        l = int(w[3][0])+1

        #print(f,m,l)

        if printing:
            plt.figure()
            plt.imshow(abs(fft))
            plt.colorbar()

            plt.figure()
            plt.plot(np.sum(abs(fft),axis=0))
            plt.ylabel('F[e(x,y)]')
            plt.xlabel('Spatial Frequency')
            plt.show()

        print(end='\x1b[2K')
        print("Progress, Isolating The Phase In Fourier... Dist {}/".format(k+1)+str(len(dists)),end='\r')
        new_fft = np.zeros_like(fft)
        diff = l-f
        new_fft[:,int(len(fft)/2-diff/2):int(len(fft)/2+diff/2)] = fft[:,f:l]
        fft[:,f:l] = 0

        fft = new_fft

        ifft = scipy.fft.ifft( scipy.fft.ifftshift(fft),axis=1)
        print(end='\x1b[2K')
        print("Progress, Extracting the Phase... Dist {}/".format(k+1)+str(len(dists)),end='\r')

        angle = np.imag(np.log(ifft))  
        angle -= angle[int(len(angle)/2),int(len(angle)/2)]

        ap_dist /=2
        aperture = Scalar_mask_XY(x,y,0.589)
        aperture.circle(r0=(0,0),radius = ap_dist)

        angle *= aperture.u

        if printing:
            plt.figure()
            plt.title('Retrieved Phase Wrapped')
            plt.imshow(angle)
            plt.colorbar()

        print(end='\x1b[2K')
        print("Progress, Unwraping the Phase... Dist {}/".format(k+1)+str(len(dists)),end='\r')
        angle = unwrap_phase(angle,wrap_around=(True, True))
        print(end='\x1b[2K')
        print("Progress, Decomposing Phase Into Components... Dist {}/".format(k+1)+str(len(dists)),end='\r')

        zerns = poppy.zernike.decompose_opd_nonorthonormal_basis(angle,nterms=20)
        print(zerns)

        nn = []
        mm = []
        for i in range(1,11):
            nc,mc= poppy.zernike.noll_indices(i)
            nn.append(nc)
            mm.append((-1)**nc * mc)

        rec = Scalar_source_XY(x,y,0.589)
        rec.zernike_beam(A=1,r0=(0,0),radius=ap_dist,n=nn,m=mm,c_nm=zerns)

        tilts.append(zerns[1])

        tilt_piston_spherical = Scalar_source_XY(x,y,0.589)
        tilt_piston_spherical.zernike_beam(A=1,r0=(0,0),radius=ap_dist,n = [1],m=[-1],
                                            c_nm = [zerns[1]])

        obj_middle = Scalar_source_XY(x,y,0.589)
        obj_middle.zernike_beam(A=1,r0=(0,0),radius=ap_dist,n=ns,m=ms,c_nm=cnms)

        if printing:
            plt.figure()
            plt.title("Retrieved Phase Unwrapped")
            plt.imshow(angle)
            plt.colorbar()

            plt.figure()
            plt.title("Desired Phase Unwrapped")
            plt.imshow(np.angle(obj_middle.u))
            plt.colorbar()

            plt.figure()
            plt.title("Difference")
            plt.imshow(angle - np.angle(obj_middle.u))
            plt.colorbar()

            plt.figure()
            plt.title("Recovered Zernike")
            plt.imshow(np.angle(rec.u))
            plt.colorbar()

            plt.show()

    plt.figure()
    plt.plot(dists,tilts,'k.')
    plt.show()


if False:
    mini = 20*np.pi
    maxi = 22*np.pi
    interval = 0.1*np.pi
    r = np.linspace(mini,maxi,int((maxi-mini)/interval)) 
    final_zerns = []
    rmse = []
    start = time.perf_counter()
    for j,tip in enumerate(r):
        print('Current Index: {}'.format(j),end='\r')
        n = 2000
        # object wave
        rad = 10
        wavelength = 0.589
        x = np.linspace(-rad,rad,n)
        y = np.linspace(-rad,rad,n)

        aperture = Scalar_mask_XY(x,y,wavelength)
        aperture.circle(r0=(0,0),radius = rad)

        ref1 = Scalar_source_XY(x,y,wavelength)
        ref1.zernike_beam(A=1,r0=(0,0),radius=rad,n=[0,1],m=[2,-1],c_nm=[1,tip])

        ref1 *= aperture

        obj_middle = Scalar_source_XY(x,y,wavelength)
        obj_middle.zernike_beam(A=1,r0=(0,0),radius=rad,n=[2,4],m=[-2,-4],c_nm=[0.6,0.1])

        obj_middle *= aperture

        intf = np.abs(ref1.u+obj_middle.u)**2


        fft = scipy.fft.fftshift( scipy.fft.fft(intf,axis=1) )

        m = scipy.signal.find_peaks(np.sum(abs(fft),axis=0),0.2*max(np.sum(abs(fft),axis=0)))
        m = min(m[0])
        w = scipy.signal.peak_widths(np.sum(abs(fft),axis=0),[m])

        f = int(w[2][0])-10
        l = int(w[3][0])+11

        if False:
            plt.figure()
            # plt.plot(np.linspace(0,1/589e-9,len(fft)),np.sum(abs(fft),axis=0))
            plt.plot(np.sum(abs(fft),axis=0))
            plt.ylabel('F[e(x,y)]')
            plt.xlabel('Spatial Frequency')
            plt.show()

        new_fft = np.zeros_like(fft)
        diff = l-f
        new_fft[:,int(len(fft)/2-diff/2):int(len(fft)/2+diff/2)] = fft[:,f:l]
        fft[:,f:l] = 0

        fft = new_fft

        ifft = scipy.fft.ifft( scipy.fft.ifftshift(fft),axis=1)

        angle = np.imag(np.log(ifft))  
        angle = unwrap_phase(angle,wrap_around=(True, True))

        angle *= aperture.u

        zerns = poppy.zernike.decompose_opd_nonorthonormal_basis(angle - np.angle(obj_middle.u),nterms=20)

        til = Scalar_source_XY(x,y,wavelength)
        til.zernike_beam(A=1,r0=(0,0),radius=rad,n=[1],m=[-1],c_nm=[zerns[1]])

        residual = abs(angle-np.angle(obj_middle.u))#-np.angle(til.u)
        # plt.figure()
        # plt.imshow(angle)
        # plt.colorbar()
        # plt.figure()
        # plt.imshow(np.angle(obj_middle.u))
        # plt.colorbar()
        # plt.figure()
        # plt.imshow(residual)
        # plt.colorbar()


        

        # calculate the RMS error in the phase
        rms =np.std(residual)/(2*np.pi) * wavelength*1e3
        rmse.append(rms)
        print(rms)
        plt.show()
    
        nn = []
        mm = []
        for i in range(1,21):
            nc,mc= poppy.zernike.noll_indices(i)
            nn.append(nc)
            mm.append((-1)**nc*mc)

        rec = Scalar_source_XY(x,y,wavelength)
        rec.zernike_beam(A=1,r0=(0,0),radius=rad,n=nn,m=mm,c_nm=zerns)

        final_zerns.append(zerns)

        del x,y,ref1,obj_middle,intf,fft,ifft,angle,zerns,nn,mm,rec

    #ani = animation.ArtistAnimation(fig, ims, interval=1000,blit=True)
    #ani.save('figure.gif')

    plt.figure()
    cols = ['black','grey','lightcoral','red','darkred','coral','peru','darkorange',
            'olive','greenyellow','turquoise','deepskyblue','royalblue','fuchsia',
            'blueviolet','hotpink','pink']
    stds = []
    
    for j in range(len(final_zerns[0])):
        zern_w  = []
        for i in range(len(final_zerns)):
            zern_w.append(final_zerns[i][j])

        # characterise the error in the phase decomposition
        if j==1:
            tilt = zern_w
            # plt.figure()
            popt, pcov = scipy.optimize.curve_fit(linear_line, r[0:20], tilt[0:20])
            # plot the tilt co-effs
            plt.figure()
            plt.plot(np.linspace(mini,maxi,int((maxi-mini)/interval)),tilt,'k.')
            plt.plot([mini,mini],[0,-2],'r--')
            plt.plot([np.mean([mini,maxi]),np.mean([mini,maxi])],[0,-2],'r--')
            plt.plot([mini,np.mean([mini,maxi])],[tilt[0],tilt[0]],'r--')
            plt.plot(r[0:40],linear_line(r[0:40],popt[0],popt[1]),'c--')
            plt.xlabel('Introduced Tilt (rad)')
            plt.ylabel("Zernike Co-eff")
            plt.show()
        if j>1:
            stds.append(np.std(zern_w[1:]))
    end = time.perf_counter()
    print(end-start)
    plt.figure()
    plt.plot(r,rmse,'k.')
    plt.xlabel('Introduced Tilt (rad)')
    plt.ylabel('RMS Error in Phase (nm)')
    plt.show()


if False:
    mini = 10*np.pi
    maxi = 100*np.pi
    interval = 0.5*np.pi
    r = np.linspace(mini,maxi,int((maxi-mini)/interval)) 
    final_zerns = []
    rmse = []
    def calc_rms(tip):
        n = 1000 
        # object wave
        rad = 0.5
        wavelength = 0.589
        x = np.linspace(-rad,rad,n)
        y = np.linspace(-rad,rad,n)

        aperture = Scalar_mask_XY(x,y,wavelength)
        aperture.circle(r0=(0,0),radius = rad)

        ref1 = Scalar_source_XY(x,y,wavelength)
        ref1.zernike_beam(A=1,r0=(0,0),radius=rad,n=[0],m=[4],c_nm=[0.1])

        ref1 *= aperture

        obj_middle = Scalar_source_XY(x,y,wavelength)
        obj_middle.zernike_beam(A=1,r0=(0,0),radius=rad,n=[2],m=[-2],c_nm=[0.1])

        obj_middle *= aperture

        intf = np.abs(ref1.u+obj_middle.u)**2


        fft = scipy.fft.fftshift( scipy.fft.fft(intf,axis=1) )

        m = scipy.signal.find_peaks(np.sum(abs(fft),axis=0),0.2*max(np.sum(abs(fft),axis=0)))
        m = min(m[0])
        w = scipy.signal.peak_widths(np.sum(abs(fft),axis=0),[m])

        f = int(w[2][0])-3
        l = int(w[3][0])+4

        new_fft = np.zeros_like(fft)
        diff = l-f
        new_fft[:,int(len(fft)/2-diff/2):int(len(fft)/2+diff/2)] = fft[:,f:l]
        fft[:,f:l] = 0

        fft = new_fft

        ifft = scipy.fft.ifft( scipy.fft.ifftshift(fft),axis=1)

        angle = np.imag(np.log(ifft))  
        angle = unwrap_phase(angle,wrap_around=(True, True))

        angle *= aperture.u

        zerns = poppy.zernike.decompose_opd_nonorthonormal_basis(angle - np.angle(obj_middle.u),nterms=20)

        til = Scalar_source_XY(x,y,wavelength)
        til.zernike_beam(A=1,r0=(0,0),radius=rad,n=[1],m=[-1],c_nm=[zerns[1]])

        residual = abs(angle-np.angle(obj_middle.u)-np.angle(til.u))
        residual -= residual[int(len(residual)/2),int(len(residual)/2)]
        residual *= aperture.u

        # calculate the RMS error in the phase
        rms =np.std(residual)/(2*np.pi) * wavelength*1e3

        plt.figure()
        plt.imshow(np.angle(intf))
        plt.show()

        return rms,tip,5
    

    with Pool(20) as p:
        rmse = p.map(calc_rms, [10,20,30,40,50])

    print(rmse)
    

    # plt.figure()
    # plt.plot(r,rmse,'k.')
    # plt.xlabel('Introduced Tilt (rad)')
    # plt.ylabel('RMS Error in Phase (nm)')
    # plt.show()
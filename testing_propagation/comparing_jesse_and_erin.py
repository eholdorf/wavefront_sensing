import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import aotools
import poppy

# import fits files from data_comp_sims
# fits files are 2D arrays of intensity
# fits files are named with the following convention:
#     "name_intensityZ_amp.fits"
# where Z is the Zernike mode and amp is the amplitude of the phase aberration and name is either erin or jesse

# create a list of the fits files
path ="/home/ehold13/PhD/testing_propagation/"
amps = np.linspace(-2,2,21,endpoint=True).tolist()
fracs = np.linspace(0,1,11,endpoint=True)

if False:
    fits_files_erin = []
    for i in range(0,16):
        for j in amps:
            for k in [0]:# fracs:
                file = pyfits.open(path+'data_comp_sims/erin_intensity'+str(i)+'_'+str(np.round(j,1))+'.fits')
                fits_files_erin.append(file)
                file.close()

    fits_files_jesse = []
    for i in range(0,16):
        for j in amps:
            file = pyfits.open(path+'data_comp_sims/jesse_intensity'+str(i)+'_'+str(np.round(j,1))+'.fits')
            fits_files_jesse.append(file)
            file.close()

plt.figure()

if False:
    count_0,count_1,count_2,count_3,count_4,count_5,count_6,count_7,count_8,count_9,count_10,count_11,count_12,count_13,count_14,count_15 = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    pup_width = 2**6 # number of pixels across the pupil
    mask = aotools.circle(pup_width/2,pup_width)
    mask = np.where(mask==0,np.nan,mask)
    for i in range(0,16):
        for j in np.linspace(-1,1,11,endpoint=True):
            file = pyfits.open(path+'data_comp_sims/erin_intensity'+str(i)+'_'+str(np.round(j,1))+'.fits')
            dat_erin = file[0].data
            dat_erin *= mask
            dat_erin -= np.nanmean(dat_erin)
            dat_erin /= np.nanmax(dat_erin) 
            file.close()
            file = pyfits.open(path+'data_comp_sims/jesse_intensity'+str(i)+'_'+str(np.round(j,1))+'.fits')
            dat_jesse = file[0].data
            dat_jesse *= mask
            dat_jesse -= np.nanmean(dat_jesse)
            dat_jesse /= np.nanmax(dat_jesse)
            file.close()

            if False:
                plt.figure()
                plt.imshow(dat_erin)
                plt.colorbar()

                plt.figure()
                plt.imshow(dat_jesse)
                plt.colorbar()

                plt.figure()
                plt.imshow(dat_jesse-dat_erin)
                plt.colorbar()
                plt.show()
            # plot the median fractional difference between the two sets of images
            frac_diff = np.nanmedian( abs(dat_jesse - dat_erin)/abs(dat_erin) ) *100
            if i==0 and count_0==0:
                plt.plot([j],[frac_diff],'ko',label=str(i+1))
                count_0 += 1
            elif i==0 and count_0==1:
                plt.plot([j],[frac_diff],'ko')
            elif i==1 and count_1==0:
                plt.plot([j],[frac_diff],'ro',label=str(i+1))
                count_1+=1
            elif i==1 and count_1==1:
                plt.plot([j],[frac_diff],'ro')
            elif i==2 and count_2==0:
                plt.plot([j],[frac_diff],'bo',label=str(i+1))
                count_2+=1
            elif i==2 and count_2==1:
                plt.plot([j],[frac_diff],'bo')
            elif i==3 and count_3==0:
                plt.plot([j],[frac_diff],'go',label=str(i+1))
                count_3+=1
            elif i==3 and count_3==1:
                plt.plot([j],[frac_diff],'go')
            elif i==4 and count_4==0:
                plt.plot([j],[frac_diff],'co',label=str(i+1))
                count_4+=1
            elif i==4 and count_4==1:
                plt.plot([j],[frac_diff],'co')
            elif i==5 and count_5==0:
                plt.plot([j],[frac_diff],'mo',label=str(i+1))
                count_5+=1
            elif i==5 and count_5==1:
                plt.plot([j],[frac_diff],'mo')
            elif i==6 and count_6==0:
                plt.plot([j],[frac_diff],'yo',label=str(i+1))
                count_6+=1
            elif i==6 and count_6==1:
                plt.plot([j],[frac_diff],'yo')


    plt.xlabel('Zernike Amplitude')
    plt.ylabel('Median Fractional Percent Difference')
    plt.legend()
    plt.show()




if False:
    for k in range(len(fits_files_erin)):
        # plot the difference and save the image
        diff = fits_files_erin[k][0].data - fits_files_jesse[k][0].data
        

        plt.figure()
        plt.imshow(diff)
        plt.colorbar()
        plt.savefig(path+"diff_images/diff_"+str(k)+".png")
        plt.clf()
        plt.close()

if False:
    pup_width = 2**6 # number of pixels across the pupil
    mask = aotools.circle(pup_width/2,pup_width)
    mask = np.where(mask==0,np.nan,mask)

    e = pyfits.open(path+'data_comp_sims/erin_intensity4_-1.0.fits')[0].data
    e *= mask
    e -= np.nanmean(e)
    e /= np.nanmax(e) 

    j = pyfits.open(path+'data_comp_sims/jesse_intensity4_-1.0.fits')[0].data
    j*= mask
    j -= np.nanmean(j)
    j /= np.nanmax(j) 


    plt.figure()
    plt.imshow(e)
    plt.colorbar()
    plt.savefig(path+"diff_images/diff_e.png")

    plt.figure()
    plt.imshow(j)
    plt.colorbar()
    plt.savefig(path+"diff_images/diff_j.png")

    print(np.nanmedian(abs(e-j)/abs(e))*100)
    plt.figure()
    plt.imshow((e-j)/e)
    plt.colorbar()
    plt.savefig(path+"diff_images/diff.png")

    plt.show()

if False:
    for k in fracs:
        e = pyfits.open(path+'data_comp_sims/erin_intensity6_0.8_'+str(np.round(k,1))+'.fits')[0].data
        j = pyfits.open(path+'data_comp_sims/jesse_intensity6_0.8.fits')[0].data

        e *= mask
        e -= np.nanmean(e)
        e /= np.nanmax(e) 

        j*= mask
        j -= np.nanmean(j)
        j /= np.nanmax(j)

        plt.figure()
        plt.imshow(abs(e-j))
        plt.show()

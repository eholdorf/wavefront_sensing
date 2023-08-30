import poppy
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

n_pix = 2**10
wavelength = 0.589*u.um
beam_rad = 10*u.um
over_samp = 20
pinhole_rad = 0.5*u.um
pad_factor = 20

# generate the initial wavefront
wf = poppy.FresnelWavefront(beam_radius=beam_rad,wavelength=wavelength,
                            npix=n_pix,oversample=over_samp)

aperature = poppy.CircularAperture(radius=beam_rad,pad_factor=pad_factor)
wf *= aperature

# generate the pinhole plate
trans = np.ones((n_pix*over_samp,n_pix*over_samp)) * 0.0

n_pix_pinhole = (pinhole_rad/(2*beam_rad) * n_pix).value

for i in range(int(n_pix*over_samp/2 - n_pix_pinhole),int(n_pix*over_samp/2 + n_pix_pinhole)):
    for j in range(int(n_pix*over_samp/2 - n_pix_pinhole),int(n_pix*over_samp/2 + n_pix_pinhole)):
        if (i-n_pix*over_samp/2)**2 + (j-n_pix*over_samp/2)**2 <=n_pix_pinhole**2:
            trans[i,j] = 1

pinhole = poppy.ArrayOpticalElement(name="PDI Plate",transmission=trans,pixelscale=2*beam_rad/(n_pix*u.pixel))
wf *= pinhole

dist = 100*u.um
wf.propagate_fresnel(dist)

plt.figure()
plt.imshow(wf.amplitude**2)
plt.show()
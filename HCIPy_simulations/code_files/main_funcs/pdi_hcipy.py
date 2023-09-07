import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from hcipy import *

class PDI(OpticalElement):
    def __init__(self,transmittance):
        self.transmittance = transmittance
    
    def forward(self,wavefront):
        wf = wavefront.copy()
        
        wf.electic_field *= np.sqrt(self.transmittance)
        
        return wf


pupil_diameter = 6e-3 #m
wavelength = 589e-9 #m
npix = 1024
oversample = 8
beam_diam = 2*1.22*wavelength*14.8
pinhole_rad = 0.5e-6 #m
focal_length = 0.1 #m
plate_diameter = 50e-6
spatial_resolution = focal_length/plate_diameter*wavelength


pupil_grid = make_pupil_grid(npix,1.2*plate_diameter)
aperture_circ = evaluate_supersampled(make_circular_aperture(beam_diam),pupil_grid,oversample)

wf = Wavefront(aperture_circ,wavelength)

pdi_plate = evaluate_supersampled(make_circular_aperture(2*pinhole_rad),pupil_grid,oversample)

for i,elem in enumerate(pdi_plate):
    if elem < 0.1:
        pdi_plate[i] = 0.00
    else:
        pdi_plate[i] = 1

plt.figure()
imshow_field(pdi_plate)

wf.electric_field *= np.sqrt(pdi_plate)

focal_grid = make_focal_grid(oversample,num_airy=3,spatial_resolution=spatial_resolution,pupil_diameter=pupil_diameter)

Fraun = FraunhoferPropagator(pupil_grid,focal_grid,focal_length=focal_length)

img = Fraun(wf)
plt.figure()
imshow_field(img.intensity)
plt.xlabel('Distance (m)')
plt.ylabel('Distance (m)')
#plt.show()

pupil_grid = make_pupil_grid(npix,1.2*pupil_diameter)
aperture_circ = evaluate_supersampled(make_circular_aperture(beam_diam),pupil_grid,oversample)

wf = Wavefront(aperture_circ,wavelength)

 
wf.electric_field *= pdi_plate
        
prop_dist = focal_length #m
fresnel=FresnelPropagator(pupil_grid,prop_dist)

img = fresnel(wf)

plt.figure()
imshow_field(img.intensity)
plt.xlabel('Distance (m)')
plt.ylabel('Distance (m)')
plt.show()

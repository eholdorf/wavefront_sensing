import matplotlib.pyplot as plt
from main_funcs import FDTD_PDI
import numpy as np 
import astropy.constants as c
import matplotlib.animation 
from tqdm import tqdm
from matplotlib.colors import LogNorm

# want dx < lambda/10
field = FDTD_PDI.FDTD([2**9,2**9,2**9,2**8],[3e-6,3e-6,2e-6])

# add a source of light
field.add_pinhole([int(field.nx/2),int(field.ny/2),1],1.52,10,2,0.0)

bar = tqdm(range(field.nt),desc="FDTD")
for t in bar:
    field.update_E()

    field.add_source(20,'z')
    
    field.update_H()

def func(i):
    plt.clf()
    a = plt.imshow(np.abs(field.Ex[1:,1:-1,i-1]+field.Ey[1:-1,1:,i-1] + field.Ez[1:-1,1:-1,i-1])**2)
    plt.text(5,5,str(i),fontsize=12)
    plt.colorbar()
    return a

E = field.Ex[:,1:,1:] + field.Ey[1:,:,1:] + field.Ez[1:,1:,:]
np.save('/home/ehold13/PhD/output_files/r_025_type_plane_abs_100_intensity.npy', E)

fig = plt.figure()
anim = matplotlib.animation.FuncAnimation(fig,func,field.nz,interval=100)
plt.show()
#anim.save('/home/ehold13/PhD/output_files/r_025_type_plane_abs_100_intensity_anim.mp4')

plt.figure()
plt.imshow(np.angle(field.Ex[1:,int(field.ny/2),1:-1]+field.Ey[1:-1,int(field.ny/2),1:-1] + field.Ez[1:-1,int(field.ny/2),1:]))

plt.figure()
plt.imshow(np.abs(field.Ex[1:,int(field.ny/2),1:-1]+field.Ey[1:-1,int(field.ny/2),1:-1] + field.Ez[1:-1,int(field.ny/2),1:])**2)

plt.figure()
plt.imshow(abs(field.e0x[1:,int(field.ny/2),1:-1]+field.e0y[1:-1,int(field.ny/2),1:-1] + field.e0z[1:-1,int(field.ny/2),1:]))

plt.figure()
plt.imshow(abs(field.absx[1:,int(field.ny/2),1:-1]+field.absy[1:-1,int(field.ny/2),1:-1] + field.absz[1:-1,int(field.ny/2),1:]))
plt.show()


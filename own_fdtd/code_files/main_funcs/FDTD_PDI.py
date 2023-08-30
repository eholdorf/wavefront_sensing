# https://eecs.wsu.edu/~schneidj/ufdtd/chap9.pdf

import numpy as np 
import matplotlib.pyplot as plt
import astropy.constants as c
import matplotlib.animation 
from tqdm import tqdm

class FDTD:
    def __init__(self,n,d):
        self.wavelength = 0.589
        # set the size of the simulation
        self.nx, self.ny, self.nz, self.nt = n

        # set constants
        self.c0 = c.c.value
        self.e0x = np.ones((self.nx-1,self.ny,self.nz)) * 8.854e-12
        self.e0y = np.ones((self.nx,self.ny-1,self.nz)) * 8.854e-12
        self.e0z = np.ones((self.nx,self.ny,self.nz-1)) * 8.854e-12
        self.u0 = 4*np.pi*1e-7

        # set the distance in the simulation
        self.d_x, self.d_y, self.d_z = d
        # set the step sizes
        self.dx, self.dy, self.dz= [self.d_x/self.nx,self.d_y/self.ny, self.d_z/self.nz]
        self.dt = 0.99/(self.c0 * np.sqrt(1/self.dx**2 + 1/self.dy**2+1/self.dz**2))

        self.Ex = np.zeros((self.nx-1,self.ny,self.nz),dtype=complex)
        self.Ey = np.zeros((self.nx,self.ny-1,self.nz),dtype=complex)
        self.Ez = np.zeros((self.nx,self.ny,self.nz-1),dtype=complex)

        self.Hx = np.zeros((self.nx,self.ny-1,self.nz-1),dtype=complex)
        self.Hy = np.zeros((self.nx-1,self.ny,self.nz-1),dtype=complex)
        self.Hz = np.zeros((self.nx-1,self.ny-1,self.nz),dtype=complex)

        # set up absorbing objects
        self.absx = np.ones_like(self.Ex)
        self.absy = np.ones_like(self.Ey)
        self.absz = np.ones_like(self.Ez)
    
    def update_E(self):
        # update the electric field
        self.Ex[:,1:-1,1:-1] += self.absx[:,1:-1,1:-1]*(self.dt/(self.e0x[:,1:-1,1:-1]*self.dy)*np.diff(self.Hz[:,:,1:-1],1,1) - self.dt/(self.e0x[:,1:-1,1:-1]*self.dz)* np.diff(self.Hy[:,1:-1,:],1,2))
        self.Ey[1:-1,:,1:-1] += self.absy[1:-1,:,1:-1]*(self.dt/(self.e0y[1:-1,:,1:-1]*self.dz)* np.diff(self.Hx[1:-1,:,:],1,2) - self.dt/(self.e0y[1:-1,:,1:-1]*self.dx)* np.diff(self.Hz[:,:,1:-1],1,0))
        self.Ez[1:-1,1:-1,:] += self.absz[1:-1,1:-1,:]*(self.dt/(self.e0z[1:-1,1:-1,:]*self.dx)* np.diff(self.Hy[:,1:-1,:],1,0) - self.dt/(self.e0z[1:-1,1:-1,:]*self.dy)* np.diff(self.Hx[1:-1,:,:],1,1))
        
        self.Ex[:5,-5:,:] = 0
        self.Ey[:5,-5:,:] = 0
        self.Ez[:5,-5:,:] = 0

    
    def update_H(self):
        self.Hx = self.Hx + self.dt/(self.u0*self.dz)* np.diff(self.Ey,1,2)  - self.dt/(self.u0*self.dy)* np.diff(self.Ez,1,1)
        self.Hy = self.Hy + self.dt/(self.u0*self.dx)* np.diff(self.Ez,1,0) - self.dt/(self.u0*self.dz)* np.diff(self.Ex,1,2)
        self.Hz = self.Hz + self.dt/(self.u0*self.dy)* np.diff(self.Ex,1,1) - self.dt/(self.u0*self.dx)* np.diff(self.Ey,1,0)
        
        self.Hx[:5,-5:,:] = 0
        self.Hy[:5,-5:,:] = 0
        self.Hz[:5,-5:,:] = 0
       
    def add_source(self,b_w,direction='z'):
        X, Y = np.meshgrid(np.linspace(-self.d_x/2,self.d_x/2,self.nx),
                                     np.linspace(-self.d_y/2,self.d_y/2,self.ny)
                                     ,indexing='ij')
        
        k = 2 * np.pi/self.wavelength
        phi = 0
        theta = np.pi/4
        # beam doesn't have a component in the direction of travel
        if direction=='z':
            # from the middle of grid out the beam width on either side in x,y plane and just first step of z             
            self.Ex[int(self.nx/2)-b_w:int(self.nx/2)+b_w,int(self.ny/2)-b_w:int(self.ny/2)+b_w,0] = 1e7*np.exp(1j * k * 
            (X[int(self.nx/2)-b_w:int(self.nx/2)+b_w,int(self.ny/2)-b_w:int(self.ny/2)+b_w]*np.sin(theta)*np.cos(phi)+Y[int(self.nx/2)-b_w:int(self.nx/2)+b_w,int(self.ny/2)-b_w:int(self.ny/2)+b_w]*np.sin(theta)*np.cos(phi)))
            
            self.Ey[int(self.nx/2)-b_w:int(self.nx/2)+b_w,int(self.ny/2)-b_w:int(self.ny/2)+b_w,0] = 1e7*np.exp(1j * k * 
            (X[int(self.nx/2)-b_w:int(self.nx/2)+b_w,int(self.ny/2)-b_w:int(self.ny/2)+b_w]*np.sin(theta)*np.cos(np.pi/2-phi)+Y[int(self.nx/2)-b_w:int(self.nx/2)+b_w,int(self.ny/2)-b_w:int(self.ny/2)+b_w]*np.sin(theta)*np.cos(np.pi/2-phi)))
        
        
            
            
    
    def add_pinhole(self,ns,n,r,w,a):
        pos_x, pos_y, pos_z = ns
        # for the z points which are in the wanted plate width
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx-1):
                    if (i-pos_x)**2/r**2 + (j-pos_y)**2/r**2 > 1 and pos_z-w<=k and k<=pos_z+w:
                        self.e0x[i,j,k] *= n**2
                        self.absx[i,j,k] = a
                    else:
                        self.e0x[i,j,k] *= 1.0003**2
        
        for k in range(self.nz):
            for j in range(self.ny-1):
                for i in range(self.nx):
                    if (i-pos_x)**2/r**2 + (j-pos_y)**2/r**2 > 1 and pos_z-w<=k and k<=pos_z+w:
                        self.e0y[i,j,k] *= n**2
                        self.absy[i,j,k] = a
                    else:
                        self.e0y[i,j,k] *= 1.0003**2

        for k in range(self.nz-1):
            for j in range(self.ny):
                for i in range(self.nx):
                    if (i-pos_x)**2/r**2 + (j-pos_y)**2/r**2 > 1 and pos_z-w<=k and k<=pos_z+w:
                        self.e0z[i,j,k] *= n**2
                        self.absz[i,j,k] = a
                    else:
                        self.e0z[i,j,k] *= 1.0003**2

if __name__ == "__main__":     
    # initialise the field
    field = FDTD([2**9,2**9,2**9,1000],[25e-6,25e-6,25e-6])

    # add a source of light
    field.add_pinhole([int(field.nx/2),int(field.ny/2),2],1.52,10,1,0.05)

    bar = tqdm(range(field.nt),desc="FDTD")
    for t in bar:
        field.update_E()

        field.add_source(5)
        
        field.update_H()

    def func(i):
        plt.clf()
        a = plt.imshow(np.abs(field.Ex[:,1:,i-1]+field.Ey[1:,:,i-1] + field.Ez[1:,1:,i-1])**2)
        plt.colorbar()
        return a

    fig = plt.figure()
    anim = matplotlib.animation.FuncAnimation(fig,func,field.ny,interval=100)
    anim.save('/home/ehold13/PhD/fdtd_anim.gif')
    plt.show()

    plt.figure()
    plt.imshow(np.abs(field.Ex[:,10,1:]+field.Ey[1:,10,1:] + field.Ez[1:,10,:])**2)
    plt.show()

    plt.figure()
    plt.imshow(np.abs(field.Ex[:,1:,-1]+field.Ey[1:,:,-1] + field.Ez[1:,1:,-1])**2)
    plt.show()

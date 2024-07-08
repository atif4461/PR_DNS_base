from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from numpy import pi, cosh, exp, round, zeros, identity, arange, real, cos, sin, multiply, outer
from numpy.fft import fft,ifft, fftfreq, fft2, ifft2
from matplotlib.pyplot import figure

import numpy as np  # Import numpy
import math
from numpy.linalg import inv

import torch
import matplotlib.pyplot as plt
import h5py
import sys
import numpy as np
# import plotly.graph_objects as go


module = torch.jit.load('./output_tensor.pt').to('cpu')
print(f'{module}')
print("Parameters:")

# Print the shape of each buffer
print("\nBuffers:")
for name, buffer in module.named_buffers():
    print(f"{name}: {buffer.shape}")


###########################################
Nx = 256
Ny = 256

for name, param in module.named_parameters():
    print(f"{name}: {param.dtype} {param} {param.shape} 0,0->{param[0,0,255,0]},{param[0,1,255,0]} and 255,255->{param[0,0,0,255]},{param[0,1,0,255]}")
  
    with torch.no_grad():
        
        for index in range(0,1):
            
            # Plotting a sample vorticity heat map
            #vortmax = (np.max(vortFD))
            #vortmin = (np.min(vortFD))
            #umax = (torch.max(u))
            #umin = (torch.min(u))
            #vmax = (torch.max(v))
            #vmin = (torch.min(v))
             
          
            fig, axs = plt.subplots(figsize=(150, 40))
            
            for a in range (0,5):
                ax2 = fig.add_subplot(3,10,a+1)
                im  = ax2.imshow(param[0,2*a  ,:,:], cmap='bwr', interpolation='nearest')
                plt.axis('off')
                ax2.set_xticks([])
                ax2.set_yticks([])
            
                ax2 = fig.add_subplot(3,10,a+11)
                im  = ax2.imshow(param[0,2*a+1,:,:], cmap='bwr', interpolation='nearest')
                plt.axis('off')
                ax2.set_xticks([])
                ax2.set_yticks([])
     
                vortFD = zeros((Ny, Nx))
                u = param[0,2*a  ,:,:]
                v = param[0,2*a+1,:,:]
                for ii in range(1, Ny-1):
                    for jj in range(1, Nx-1):
                        vortFD[ii, jj] = v[ii,jj+1] - v[ii,jj-1] - u[ii-1,jj] + u [ii+1,jj]  
     
                ax2 = fig.add_subplot(3,10,a+21)
                im  = ax2.imshow(torch.from_numpy(vortFD[:,:]), cmap='bwr', interpolation='nearest')
                plt.axis('off')
                ax2.set_xticks([])
                ax2.set_yticks([])
     
            
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.86, 0.15, 0.01, 0.7])
            cbar_ax.tick_params(labelsize=20)
            fig.colorbar(im,cax=cbar_ax)
            
            fig.savefig(f'test_{index}_3d_channels_uv.png')
            plt.close()
        
    

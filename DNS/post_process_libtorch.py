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
from utils import get_rel_l1_error, get_rel_l2_error
from utils.compute_gradients import *


       
module  = torch.jit.load('/home/atif/PR_DNS_base/DNS/input_tensor_50.pt').to('cpu')
print(f'{module}')
print("Parameters:")

# Print the shape of each buffer
print("\nBuffers:")
for name, buffer in module.named_buffers():
    print(f"{name}: {buffer.shape}")


###########################################
Nx = 256
Ny = 256


velx = zeros((5,Nx,Ny))
vely = zeros((5,Nx,Ny))
vel = zeros((2,Nx,Ny))

for name, param in module.named_parameters():
    print(f"{name}: {param.dtype} {param.shape} 0,0->{param[0,0,255,0]},{param[0,1,255,0]} and 255,255->{param[0,0,0,255]},{param[0,1,0,255]}")
  
    with torch.no_grad():
        
        for index in range(0,1):
            
            velx = param[0,:,0:256,:]
            vely = param[0,:,256:512,:]
            
            fig, axs = plt.subplots(figsize=(10, 40))
        
            for a in range(0,5):

                vel[0,:,:] = velx[a,:,:]
                vel[1,:,:] = vely[a,:,:]
                vortFD = calculate_vorticity_2d( vel, 1, 1) 
 
                ax2 = fig.add_subplot(3,1,1)
                im  = ax2.imshow(velx[a,:,:], cmap='BrBG', interpolation='nearest')
                plt.axis('off')
                ax2.set_xticks([])
                ax2.set_yticks([])
            
                ax2 = fig.add_subplot(3,1,2)
                im  = ax2.imshow(vely[a,:,:], cmap='BrBG', interpolation='nearest')
                plt.axis('off')
                ax2.set_xticks([])
                ax2.set_yticks([])
         
                ax2 = fig.add_subplot(3,1,3)
                im  = ax2.imshow(vortFD[:,:], cmap='BrBG', interpolation='nearest')
                plt.axis('off')
                ax2.set_xticks([])
                ax2.set_yticks([])
         
            
                fig.subplots_adjust(right=0.85)
                cbar_ax = fig.add_axes([0.86, 0.15, 0.01, 0.7])
                cbar_ax.tick_params(labelsize=20)
                fig.colorbar(im,cax=cbar_ax)
                
                fig.savefig(f'test_input_{a}_3d_channels_uv.png')
                plt.close()
            
     





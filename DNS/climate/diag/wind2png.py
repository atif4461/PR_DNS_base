import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

file_dir = "../out-entrainment2dm/record-yvel/"
file_dir_out =  "out-entrainment2dm/yvel/"
file_list =  os.listdir(file_dir)

for i in range(300):
    f = "yvel-"+f'{i:07d}'+".h5"
    print(f)
    fid = h5py.File(file_dir+f,"r")
    key_list = list(fid.keys())[0]
    data = np.array(fid[key_list])

    xx = np.arange(data.shape[0])
    yy = np.arange(data.shape[1])

    plt.contourf(xx,yy,data)
    plt.savefig(file_dir_out+f+".png")




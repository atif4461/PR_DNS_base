import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

T = 20
var = 'vapor'
file_dir = "record-"+var+"/"
file_list =  os.listdir(file_dir)

dataset = []

for i in range(420):
    f = var+"-"+f'{i:07d}'+".h5"
    fid = h5py.File(file_dir+f,"r")
    key_list = list(fid.keys())[0]
    data = np.array(fid[key_list])

    dataset.append(data)

dataset = np.array(dataset)
print(dataset.shape)

dataset_4d = np.zeros((dataset.shape[0]-T,dataset.shape[1],dataset.shape[2],T))

for i in range(dataset_4d.shape[0]):
    for j in range(T):
        dataset_4d[i,:,:,j] = dataset[i+j,:,:]


np.save("pr_dns_"+var+".npy", dataset_4d)

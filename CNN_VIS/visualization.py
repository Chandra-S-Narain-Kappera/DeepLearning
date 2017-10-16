import torch
import torchvision.models as models
from matplotlib import pyplot as plt
import numpy as np

def plot_kernels(tensor, filename, num_cols=6):
    plt.clf()
    if not tensor.ndim==4:
        raise Exception("assumes a 4D tensor")

    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(tensor.shape[0]):
        imdata = tensor[i].copy()
        if (tensor.shape[1] is not 3):
            imdata = imdata.mean(axis=0)
        else:
            imdata = tensor[i].reshape(tensor.shape[2],tensor.shape[3],tensor.shape[1])

        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(imdata)
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(filename)

def plot_activations(tensor, label, data, num_cols=6):
    label = 'Activations/'+label

    if not tensor.ndim==4:
        raise Exception("assumes a 4D tensor")

    num_kernels = tensor.shape[1]+1
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    data = data.numpy()
    for i in range(tensor.shape[0]): # looping through the batch
        plt.clf()
        ax1 = fig.add_subplot(num_rows,num_cols,1)
        imdata = data[i]
        imdata = imdata/2+0.5
        imdata = np.transpose(imdata,(1,2,0))
        ax1.imshow(imdata)
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        for j in range(tensor.shape[1]): # looping through the layers
            imdata = tensor[i][j].copy()

            ax1 = fig.add_subplot(num_rows,num_cols,j+2)
            ax1.imshow(imdata)
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
        filename = label+'_'+str(i)+'.png'
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.savefig(filename)
    

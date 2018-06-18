import numpy as np
import os
import sys
import time

def net2deeper(bias,noise_std):

    #assumes that this is a linear layer, idea is to capture function preserving transformation
    #which may be captured by model itsel in future but as of now it is not possible hence we
    #do it manually hence this verssion can be applied to only linear layers

    new_weight=np.matrix(np.eye(bias.shape[0],dtype=bias.dtype))
    new_bias=np.zeros(bias.shape[0],dtype=bias.dtype)

    if noise_std:
        new_weight+=np.random.normal(scale=noise_std,size=new_weight.shape)
        new_bias+=np.random.normal(scale=noise_std,size=new_bias.shape)

    return new_weight,new_bias


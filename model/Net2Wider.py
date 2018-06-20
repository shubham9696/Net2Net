import numpy as np
import os
import sys
import time

def net2wider(weight,bias,next_layer_weight,noise_std=0.01,new_size=None,split_max=True):

    if weight.shape[1]!=bias.shape[0]:
        raise ValueError("weight must have second dimension=%s equal to the dimension of bias=%s"%(weight.shape[1],bias.shape[0]))

    if bias.shape[0]!=next_layer_weight.shape[0]:
        raise ValueError("incompatible dimensions")

    if new_size is None:
        new_size=bias.shape[0]+1
    elif new_size<=bias.shape[0]:
        raise ValueError("Model not suitable for thinner networks")

    while bias.shape[0]<new_size:
        weight,bias,next_layer_weight=wider_by_one(weight,bias,next_layer_weight,noise_std,split_max)

    return weight,bias,next_layer_weight



def wider_by_one(weight,bias,next_layer,noise_std=0.01,split_max=True):

    if split_max:
        node_split=np.argmax(np.dot(np.ones(weight.shape[0]),weight))
    else:
        node_split=np.random.randint(0,bias.shape[0])


    new_weight_layer=weight[:,node_split]

    next_layers_new_weights=next_layer[node_split,:]*0.5

    new_bias=np.r_[bias,[bias[node_split]]]

    if next_layers_new_weights.ndim==1:
        next_layers_new_weights=np.reshape(next_layers_new_weights,(1,next_layers_new_weights.shape[0]))

    if noise_std:
        weight_noise=np.random.normal(scale=noise_std,size=new_weight_layer.shape)
        new_weight_layer+=weight_noise

        bias_noise=np.random.normal(scale=noise_std)
        bias[-1]+=bias_noise

        output_weight_noise=np.random.normal(scale=noise_std,size=next_layers_new_weights.shape)
        next_layers_new_weights+=output_weight_noise

    new_weight=np.c_[weight,new_weight_layer]

    new_next_layer_weight=np.r_[next_layer,next_layers_new_weights]

    new_next_layer_weight[node_split,:]*=0.5

    return new_weight,new_bias,new_next_layer_weight







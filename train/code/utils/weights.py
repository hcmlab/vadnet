import os, csv, time, math

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np
import matplotlib.pyplot as plt


def get_layers(path):

    if os.path.isdir(path):
        path = tf.train.get_checkpoint_state(path).model_checkpoint_path

    reader = pywrap_tensorflow.NewCheckpointReader(path)    
    return reader.get_variable_to_shape_map()


def get_weights(path, name):
    
    if os.path.isdir(path):
        path = tf.train.get_checkpoint_state(path).model_checkpoint_path
    
    reader = pywrap_tensorflow.NewCheckpointReader(path)    
    return reader.get_tensor(name)
    

def save_weights(weights, path):

    weights.tofile(os.path.join(path))


def plot_weights(weights):
    
    n_filter = weights.shape[-1]
    n_rowcol = int(math.ceil(math.sqrt(n_filter)))
    
    for i in range(n_filter):        
        filter = weights[:,i]
        plt.subplot(n_rowcol, n_rowcol, i+1)
        plt.plot(filter)

    plt.show()


if __name__ == '__main__':    

    layer = 1
    dim = 0

    path = r'X:\Mediathek\nets\UrbanSound_Model[16000,8000,4000,128,None]_ConvNoRelu3Rnn[2,64,gru]_SceAdam[1e-04,0.9,0.999,1e-08]\ckpt'
    #path = r'X:\Mediathek\nets\vad_Model[44100,22050,11025,128,None]_ConvNoRelu3Rnn[2,64,gru]_SceAdam[1e-04,0.9,0.999,1e-08]\ckpt'
    #path = r'..\test\ckpt'
    weights = get_weights(path, 'layers/conv{}/conv1d/kernel'.format(layer))           
    weights = weights[:,dim,:].squeeze()

    plot_weights(weights)
    save_weights(weights.squeeze(), os.path.join(r'utils\conv{}.weights'.format(layer)))
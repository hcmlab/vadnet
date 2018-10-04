import enum, functools, operator

import tensorflow as tf


class RnnCell(enum.Enum): 
    GRU, LSTM = range(2)


def convolution(inputs:tf.Tensor, filters:int, kernel_size:int, strides:int=1) -> tf.Tensor:

    inputs = tf.layers.conv1d(inputs, filters, kernel_size, strides=strides, padding='SAME', kernel_initializer=tf.truncated_normal_initializer(0.0, stddev=0.01), bias_initializer=tf.constant_initializer(0.0))
        
    #w_conv = tf.get_variable('weights', (kernel_size, inputs.shape[2], filters), initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))                    
    #b_conv = tf.get_variable('biases', (filters,), initializer=tf.constant_initializer(0.0))                
    #inputs = tf.nn.conv1d(inputs, w_conv, strides, padding=padding.upper(), name='z') + b_conv

    return inputs


def norm(inputs:tf.Tensor) -> tf.Tensor:
    return tf.layers.batch_normalization(inputs)


def relu(inputs:tf.Tensor) -> tf.Tensor:
    return tf.nn.relu(inputs)


def maxpool(inputs:tf.Tensor, pool_size:int, strides:int) -> tf.Tensor:
    return tf.layers.max_pooling1d(inputs, pool_size, strides)


def avgpool(inputs:tf.Tensor, pool_size:int, strides:int, absolute:bool=False) -> tf.Tensor:
    if absolute:
        inputs = tf.abs(inputs, name='abs')
    return tf.layers.average_pooling1d(inputs, pool_size, strides)


def rnn_cell(units:int, type:RnnCell) -> tf.Tensor:
    if type == RnnCell.GRU:
        cell = tf.nn.rnn_cell.GRUCell(units)        
    elif type == RnnCell.LSTM:
        cell = tf.nn.rnn_cell.LSTMCell(n_rnn_hidden_units, use_peepholes=True, cell_clip=100, state_is_tuple=True)
    #cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.5)        
    return cell


def rnn(inputs:tf.Tensor, layers:int, units:int, type:RnnCell) -> tf.Tensor:    
    stacked_cells = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(units, type) for _ in range(layers)], state_is_tuple=True)
    inputs, _ = tf.nn.dynamic_rnn(stacked_cells, inputs, dtype=tf.float32)    
    return inputs 


def dense(inputs:tf.Tensor, units:int, activation=None) -> tf.Tensor:    
    return tf.layers.dense(inputs[:,-1,:], units, activation=activation)


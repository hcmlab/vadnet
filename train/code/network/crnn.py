import argparse
from typing import Optional, List

import tensorflow as tf

import network.ops as opts
from network.base import NetworkBase


class ConvXRnn(NetworkBase):

    def get_args(self, parser:argparse.ArgumentParser):

        super().get_args(parser)

        parser.add_argument('--conv_stride',
                default=2,
                type=int,
                help='stride of conv layer')

        parser.add_argument('--conv_apply_norm',
                default=True,
                type=lambda x: str(x).lower() == 'true',
                help='add a batch norm after each conv layer')

        parser.add_argument('--conv_apply_relu',
                default=False,
                type=lambda x: str(x).lower() == 'true',
                help='add a relu after each conv layer')        

        parser.add_argument('--conv_pool_size',
                default=4,
                type=int,
                help='pooling size after each conv layer')

        parser.add_argument('--conv_pool_stride',
                default=4,
                type=int,
                help='pooling stride after each conv layer')

        parser.add_argument('--rnn_apply',
                default=False,
                type=lambda x: str(x).lower() == 'true',
                help='add rnn layers after conv layers')

        parser.add_argument('--n_rnn_layers',
                default=2,
                type=int,
                help='number of rnn layers')

        parser.add_argument('--n_rnn_units',
                default=32,
                type=int,
                help='number of hidden rnn units')

        parser.add_argument('--rnn_cell_type',
                default='gru',
                type=str,
                choices=[i.name.lower() for i in opts.RnnCell],
                help='rnn cell type')

        return parser


    def get_name(self, args) -> str :
        name = type(self).__name__
        if args.rnn_apply:
            return '{}[{},{},{},{},{},{},{},{}]'.format(name, 
                args.conv_stride, 
                args.conv_apply_norm,
                args.conv_apply_relu, 
                args.conv_pool_size, 
                args.conv_pool_stride,
                args.n_rnn_layers, 
                args.n_rnn_units, 
                args.rnn_cell_type)
        else:
            return '{}[{},{},{},{},{}]'.format(name.replace('Rnn', ''), 
                args.conv_stride, 
                args.conv_apply_norm,
                args.conv_apply_relu, 
                args.conv_pool_size, 
                args.conv_pool_stride)


class ConvLayer:
    def __init__(self):
        self.conv_layer_count = 0
    def __call__ (self, inputs, layers, args, filters, kernel_size, pool=True):                
        self.conv_layer_count += 1
        with tf.variable_scope('conv{}'.format(self.conv_layer_count)):
            inputs = opts.convolution(inputs, filters, kernel_size, args.conv_stride); layers.append(inputs)
            if args.conv_apply_norm:
                inputs = opts.norm(inputs); layers.append(inputs)
            if args.conv_apply_relu:
                inputs = tf.nn.relu(inputs); layers.append(inputs)
        if pool:
            with tf.variable_scope('pool{}'.format(self.conv_layer_count)):
                inputs = opts.maxpool(inputs, args.conv_pool_size, args.conv_pool_stride); layers.append(inputs)                                     
        return inputs


class RnnLayer:
    def __call__ (self, inputs, layers, args): 
        with tf.variable_scope('rnn'):
            inputs = opts.rnn(inputs, args.n_rnn_layers, args.n_rnn_units, opts.RnnCell[args.rnn_cell_type.upper()]); layers.append(inputs)         
        return inputs


class DenseLayer:
    def __call__ (self, inputs, layers, n_targets, args): 
        with tf.variable_scope('logits'):
            inputs = opts.dense(inputs, n_targets, activation=tf.nn.softmax); layers.append(inputs)
        return inputs


class Conv3Rnn(ConvXRnn):
    
    def get_layers(self, args, inputs:tf.Tensor, n_targets:int) -> List[tf.Tensor] :
                
        conv_layer = ConvLayer()
        rnn_layer = RnnLayer()
        dense_layer = DenseLayer()
        layers = []

        inputs = tf.reshape(inputs, (-1, inputs.shape[1], 1))
        
        with tf.variable_scope('layers'): 
                   
            inputs = conv_layer(inputs, layers, args, 16, 64)
            inputs = conv_layer(inputs, layers, args, 32, 32)
            inputs = conv_layer(inputs, layers, args, 64, 16)
            if args.rnn_apply:
                inputs = rnn_layer(inputs, layers, args)  
            inputs = dense_layer(inputs, layers, n_targets, args)            

        return layers


class Conv5Rnn(ConvXRnn):


    def get_layers(self, args, inputs:tf.Tensor, n_targets:int) -> List[tf.Tensor] :
        
        conv_layer = ConvLayer()
        rnn_layer = RnnLayer()
        dense_layer = DenseLayer()
        layers = []

        inputs = tf.reshape(inputs, (-1, inputs.shape[1], 1))
        
        with tf.variable_scope('layers'):

            inputs = conv_layer(inputs, layers, args, 4, 256)
            inputs = conv_layer(inputs, layers, args, 8, 128)
            inputs = conv_layer(inputs, layers, args, 16, 64)
            inputs = conv_layer(inputs, layers, args, 32, 32)
            inputs = conv_layer(inputs, layers, args, 64, 16)
            if args.rnn_apply:
                inputs = rnn_layer(inputs, layers, args)  
            inputs = dense_layer(inputs, layers, n_targets, args)   
        
        return layers


class Conv7Rnn(ConvXRnn):


    def get_layers(self, args, inputs:tf.Tensor, n_targets:int) -> List[tf.Tensor] :
        
        conv_layer = ConvLayer()
        rnn_layer = RnnLayer()
        dense_layer = DenseLayer()
        layers = []

        inputs = tf.reshape(inputs, (-1, inputs.shape[1], 1))
        
        with tf.variable_scope('layers'):

            inputs = conv_layer(inputs, layers, args, 4, 256)
            inputs = conv_layer(inputs, layers, args, 8, 128)
            inputs = conv_layer(inputs, layers, args, 16, 64)
            inputs = conv_layer(inputs, layers, args, 32, 32)
            inputs = conv_layer(inputs, layers, args, 64, 16)
            inputs = conv_layer(inputs, layers, args, 128, 8)                       
            inputs = conv_layer(inputs, layers, args, 256, 4)   
            if args.rnn_apply:  
                inputs = rnn_layer(inputs, layers, args)  
            inputs = dense_layer(inputs, layers, n_targets, args)    

        return layers


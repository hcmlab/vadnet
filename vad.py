'''
ssivad.py
author: Johannes Wagner <wagner@hcm-lab.de>
created: 2018/05/04
Copyright (C) University of Augsburg, Lab for Human Centered Multimedia

Returns energy of a signal (dimensionwise or overall)
'''

import sys, os, json

import tensorflow as tf
import numpy as np


def getOptions(opts,vars):

    opts['path'] = ''
    opts['n_classes'] = 2


def getSampleDimensionOut(dim, opts, vars):

    return opts['n_classes']


def getSampleTypeOut(type, types, opts, vars): 

    if type != types.FLOAT:  
        print('types other than float are not supported') 
        return types.UNDEF

    return type


def load_model(opts, vars):

    ckeckpoint_path = opts['path']

    graph = tf.Graph()

    with graph.as_default():

        print('loading model {}'.format(ckeckpoint_path)) 
        saver = tf.train.import_meta_graph(ckeckpoint_path + '.meta')
        with open(ckeckpoint_path + '.json', 'r') as fp:
            vocab = json.load(fp)

        x = graph.get_tensor_by_name(vocab['x'])
        y = graph.get_tensor_by_name(vocab['y'])            
        init = graph.get_operation_by_name(vocab['init'])
        logits = graph.get_tensor_by_name(vocab['logits'])            
        ph_n_shuffle = graph.get_tensor_by_name(vocab['n_shuffle'])
        ph_n_repeat = graph.get_tensor_by_name(vocab['n_repeat'])
        ph_n_batch = graph.get_tensor_by_name(vocab['n_batch'])

        sess = tf.Session()    
        saver.restore(sess, ckeckpoint_path)

        vars['sess'] = sess
        vars['x'] = x
        vars['y'] = y    
        vars['ph_n_shuffle'] = ph_n_shuffle
        vars['ph_n_repeat'] = ph_n_repeat
        vars['ph_n_batch'] = ph_n_batch
        vars['init'] = init
        vars['logits'] = logits


def transform_enter(sin, sout, sxtra, board, opts, vars): 

    load_model(opts, vars)


def transform(info, sin, sout, sxtra, board, opts, vars): 
  
    sess = vars['sess']
    x = vars['x']
    y = vars['y']
    ph_n_shuffle = vars['ph_n_shuffle']
    ph_n_repeat = vars['ph_n_repeat']
    ph_n_batch = vars['ph_n_batch']
    init = vars['init']
    logits = vars['logits']

    input = np.asmatrix(sin).reshape(-1, x.shape[1])
    dummy = np.zeros((input.shape[0],), dtype=np.int32)
    sess.run(init, feed_dict = { x : input, y : dummy, ph_n_shuffle : 1, ph_n_repeat : 1, ph_n_batch : input.shape[0] })    
    output = sess.run(logits)    
    output = np.mean(output, axis=0)
    
    sout[0] = output[0]
    sout[1] = output[1]


def transform_flush(sin, sout, sxtra, board, opts, vars): 

    sess = vars['sess']
    sess.close()
'''
author: Johannes Wagner <wagner@hcm-lab.de>
created: 2018/05/04
Copyright (C) University of Augsburg, Lab for Human Centered Multimedia
'''

import sys, os, json, glob

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import numpy as np


def getOptions(opts,vars):

    opts['path'] = ''    


def getSampleDimensionOut(dim, opts, vars):

    vars['loaded'] = False
    vars['n_classes'] = 0

    try:
        load_model(opts, vars)					
        vars['loaded'] = True
    except Exception as ex:
        print(ex)

    return vars['n_classes']


def getSampleTypeOut(type, types, opts, vars): 

    if type != types.FLOAT:  
        print('types other than float are not supported') 
        return types.UNDEF

    return type


def load_model(opts, vars):

    print('load model ', opts['path'])

    if os.path.isdir(opts['path']):
        files = glob.glob(os.path.join(opts['path'], 'model.ckpt-*.meta'))
        if files:
            files.sort()                
            checkpoint_path, _ = os.path.splitext(files[-1])
    else:
        checkpoint_path = opts['path']        

    if not all([os.path.exists(checkpoint_path + x) for x in ['.data-00000-of-00001', '.index', '.meta']]):
        print('ERROR: could not load model')
        raise FileNotFoundError

    vocabulary_path = checkpoint_path + '.json'
    if not os.path.exists(vocabulary_path):
        vocabulary_path = os.path.join(os.path.dirname(checkpoint_path), 'vocab.json')
    if not os.path.exists(vocabulary_path):
        print('ERROR: could not load vocabulary')
        raise FileNotFoundError

    graph = tf.Graph()

    with graph.as_default():

        print('loading model {}'.format(checkpoint_path)) 
        saver = tf.train.import_meta_graph(checkpoint_path + '.meta')
        with open(vocabulary_path, 'r') as fp:
            vocab = json.load(fp)

        x = graph.get_tensor_by_name(vocab['x'])
        y = graph.get_tensor_by_name(vocab['y'])            
        init = graph.get_operation_by_name(vocab['init'])
        logits = graph.get_tensor_by_name(vocab['logits'])            
        ph_n_shuffle = graph.get_tensor_by_name(vocab['n_shuffle'])
        ph_n_repeat = graph.get_tensor_by_name(vocab['n_repeat'])
        ph_n_batch = graph.get_tensor_by_name(vocab['n_batch'])
        vars['n_classes'] = len(vocab['targets'])

        sess = tf.Session()    
        saver.restore(sess, checkpoint_path)

        vars['sess'] = sess
        vars['x'] = x
        vars['y'] = y    
        vars['ph_n_shuffle'] = ph_n_shuffle
        vars['ph_n_repeat'] = ph_n_repeat
        vars['ph_n_batch'] = ph_n_batch
        vars['init'] = init
        vars['logits'] = logits


def transform_enter(sin, sout, sxtra, board, opts, vars): 	

    pass


def transform(info, sin, sout, sxtra, board, opts, vars): 
     
    if vars['loaded']:	

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

        for i in range(sout.dim):
            sout[i] = output[i]        


def transform_flush(sin, sout, sxtra, board, opts, vars): 

    if vars['loaded']:

        sess = vars['sess']
        sess.close()
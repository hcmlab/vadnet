'''
model.py
author: Johannes Wagner <wagner@hcm-lab.de>
created: 2018/05/04
Copyright (C) University of Augsburg, Lab for Human Centered Multimedia

Returns energy of a signal (dimensionwise or overall)
'''

import sys, os, json, glob

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import numpy as np


def getOptions(opts,vars):

    pass


def getSampleDimensionOut(dim, opts, vars):
    
    return 5


def getSampleTypeOut(type, types, opts, vars): 

    if type != types.FLOAT:  
        print('types other than float are not supported') 
        return types.UNDEF

    return type


def transform_enter(sin, sout, sxtra, board, opts, vars): 	

    pass


def transform(info, sin, sout, sxtra, board, opts, vars): 
       
    voiced = sin[1] > sin[0]    

    for i in range(sout.dim):
        sout[i] = 0
   
    sout[0] = sin[0] # noise
    sout[1] = sin[1] # voice    
        
    if voiced:
        sout[2] = sin[2] # male
        sout[3] = sin[3] # female                
        sout[4] = sin[6] # laugh
     

def transform_flush(sin, sout, sxtra, board, opts, vars): 

    pass
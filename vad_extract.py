'''
ssivad.py
author: Johannes Wagner <wagner@hcm-lab.de>
created: 2018/05/04
Copyright (C) University of Augsburg, Lab for Human Centered Multimedia

Returns energy of a signal (dimensionwise or overall)
'''

import sys, os, json, argparse

import tensorflow as tf
import numpy as np
import librosa as lr


def audio_from_file(path, sr=None, ext=''):
    return lr.load('{}{}'.format(path, ext), sr=sr, mono=True, offset=0.0, duration=None, dtype=np.float32, res_type='kaiser_best')                


def audio_to_file(path, x, sr):    
    lr.output.write_wav(path, x.reshape(-1), sr, norm=False)   


def audio_to_frames(x, n_frame, n_step=None):    

    if n_step is None:
        n_step = n_frame

    if len(x.shape) == 1:
        x.shape = (-1,1)

    n_overlap = n_frame - n_step
    n_frames = (x.shape[0] - n_overlap) // n_step       
    n_keep = n_frames * n_step + n_overlap

    strides = list(x.strides)
    strides[0] = strides[1] * n_step

    return np.lib.stride_tricks.as_strided(x[0:n_keep,:], (n_frames,n_frame), strides)


def extract_voice(ckeckpoint_path, files, sr=44100, n_batch=256):

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

        with tf.Session() as sess:

            saver.restore(sess, ckeckpoint_path)

            for file in files:

                print('processing {}..'.format(file), end='')

                if os.path.exists(file):                
                    sound, _ = audio_from_file(file, sr=sr)
                    input = audio_to_frames(sound, x.shape[1])
                    dummy = np.zeros((input.shape[0],), dtype=np.int32)
                    sess.run(init, feed_dict = { x : input, y : dummy, ph_n_shuffle : 1, ph_n_repeat : 1, ph_n_batch : n_batch })                        
                    prediction = sess.run(logits)    
                    winner = np.argmax(prediction, axis=1)
                    noise = input[np.argwhere(winner==0),:].reshape(-1,1)
                    speech = input[np.argwhere(winner==1),:].reshape(-1,1)
                    name, ext = os.path.splitext(file)                    
                    audio_to_file(os.path.join(name + '.speech' + ext), speech, sr)                    
                    audio_to_file(os.path.join(name + '.noise' + ext), noise, sr)                    

                    print('ok')

                else:
                    print('skip [file not found]')


parser = argparse.ArgumentParser()

parser.add_argument('--model',
                default=r'models\model.ckpt-47072',
                help='path to model')  

parser.add_argument('--files', 
                nargs='+', 
                default=[r'data\noise.wav', r'data\speech.wav'],
                help='list of files')


if __name__ == '__main__':

    args = parser.parse_args()

    extract_voice(args.model, args.files)
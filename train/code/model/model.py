import argparse, time, os, random, datetime, json, glob, re
from typing import Optional, List

import tensorflow as tf
import numpy as np

from source.utils import anno_from_file, audio_from_file, audio_to_frames, BalanceMethod
from utils.printy import print_prog, print_info, print_star, print_err

from network.base import NetworkBase
from trainer.base import TrainerBase
from source.base import SourceBase
from source.loader import Loader
from model.base import ModelBase

class Model(ModelBase):


    def get_args(self, parser:argparse.ArgumentParser):

        parser.add_argument('--retrain',
                        default=True,
                        type=lambda x: str(x).lower() == 'true',
                        help='continue training from last checkpoint if possible')
        parser.add_argument('--output_dir',
                        default='.',
                        help='folder to store summaries and checkpoints')        
        parser.add_argument('--exp_name',
                        default=None,
                        help='name of experiment (if None will be created)')        
        parser.add_argument('--sample_rate',
                        default=44100,
                        type=int,
                        help='sample rate in hz')
        parser.add_argument('--n_frame',
                        default=22050,
                        type=int,
                        help='frame size in samples')
        parser.add_argument('--n_step',
                        default=11025,
                        type=int,
                        help='step size in samples')
        parser.add_argument('--balance', 
                        default=None,
                        type=str,
                        choices=[i.name for i in BalanceMethod],
                        help='balance method')
        parser.add_argument('--n_fetch_threads',
                        default=5,
                        type=int,
                        help='number of threads fetching data')
        parser.add_argument('--n_batch',
                        default=128,
                        type=int,
                        help='batch size')
        parser.add_argument('--n_shuffle',
                        default=1000,
                        type=int,
                        help='shuffle size')
        parser.add_argument('--n_epochs',
                        default=2,
                        type=int,
                        help='number of epochs')
        parser.add_argument('--n_repeat',
                        default=1,
                        type=int,
                        help='number of repetitions')
        parser.add_argument('--n_log_steps',
                        default=1,
                        type=int,
                        help='create summary every n steps')
        parser.add_argument('--n_eval_files',
                        default=1,
                        type=int,
                        help='run evaluation after n files')
        parser.add_argument('--log_filename',
                        default=False,
                        type=lambda x: str(x).lower() == 'true',
                        help='include filename in summary')
        parser.add_argument('--n_save_secs',
                        default=300,
                        type=int,
                        help='save model every n seconds')
        parser.add_argument('--eval_audio_file',
                        default=None,                        
                        help='use this audio file to evaluate the model')
        parser.add_argument('--eval_anno_file',
                        default=None,                        
                        help='use this annotation file to evaluate the model')
        parser.add_argument('--eval_blacklist_thres',
                        default=None,                        
                        type=float,
                        help='blacklist files if evaluation accuracy is below threshold (will be excluded in next iteration)')

        return parser


    def get_name(self, args) -> str :
        return '{}[{},{},{},{},{}]'.format(type(self).__name__, args.sample_rate, args.n_frame, args.n_step, args.n_batch, args.balance)


    def train(self, args, source:SourceBase, network:NetworkBase, trainer:TrainerBase):               

        targets = source.get_targets(args)
        n_targets = len(targets)

        do_eval = False
        if args.eval_audio_file and args.eval_anno_file:
            eval_audio = audio_from_file(args.eval_audio_file, args.sample_rate)
            eval_frames = audio_to_frames(eval_audio, args.n_frame, n_step=args.n_step)
            eval_labels = anno_from_file(args.eval_anno_file, eval_frames.shape[0], args.sample_rate/args.n_step)
            eval_predict = np.zeros((eval_frames.shape[0],), dtype=np.int32)
            do_eval = True

        graph = tf.Graph()

        with graph.as_default():
    
            with tf.name_scope('ph'): # place holder

                ph_frames = tf.placeholder(dtype=tf.float32, shape=(None, args.n_frame), name='frames')
                ph_labels = tf.placeholder(dtype=tf.int32, shape=(None,), name='labels')
                ph_n_shuffle = tf.placeholder(dtype=tf.int64, shape=(), name='n_shuffle')
                ph_n_repeat = tf.placeholder(dtype=tf.int64, shape=(), name='n_repeat')
                ph_n_batch = tf.placeholder(dtype=tf.int64, shape=(), name='n_batch')

            with tf.name_scope('ds'): # data set

                ds_frames = tf.data.Dataset.from_tensor_slices(ph_frames)
                ds_labels = tf.data.Dataset.from_tensor_slices(ph_labels).map(lambda z : tf.one_hot(z, n_targets))
                ds = tf.data.Dataset.zip((ds_frames, ds_labels)) \
                    .shuffle(ph_n_shuffle) \
                    .repeat(ph_n_repeat) \
                    .batch(ph_n_batch)            
                ds_iter = ds.make_initializable_iterator()
                ds_init = ds_iter.make_initializer(ds, name='initializer')
                ds_next = ds_iter.get_next(name='next')                        
    
            with tf.name_scope('net'): # network  
                                  
                layers = network.get_layers(args, ds_next[0], n_targets) 
                for layer in layers:
                    print_info('{} -> {}'.format(layer.name, layer.shape))           

                logits = layers[-1]
                one_hot_labels = ds_next[1]        

            with tf.name_scope('train'): # loss, optimizer, train ops

                global_frames_counter = tf.get_variable('global_frames_counter', initializer=tf.constant(0, dtype=tf.int32))
                global_frames = tf.assign(global_frames_counter, global_frames_counter + tf.shape(ds_next[0])[0])        
                global_step = tf.train.get_or_create_global_step()
                  
                loss = trainer.get_loss(args, logits, one_hot_labels)                         
                optimize = trainer.get_optimize(args, loss, global_step)                  
                accuracy = trainer.get_accuracy(args, logits, one_hot_labels)     

            with tf.name_scope('log'): # summary
        
                tf.summary.scalar('loss', loss)
                tf.summary.scalar('accuracy', accuracy)
                   
                merged_summaries = tf.summary.merge_all()   
                summary_writer = tf.summary.FileWriter(args.log_dir, graph=graph)            
                summary_hook = tf.train.SummarySaverHook(summary_writer=summary_writer, summary_op=merged_summaries, save_steps=args.n_log_steps)

            with tf.name_scope('save'): # saver
        
                saver_hook = tf.train.CheckpointSaverHook(args.checkpoint_dir, save_secs=args.n_save_secs)        

            # customize scaffold
            
            scaffold = tf.train.Scaffold(init_fn = lambda scaffold, session: print_info('train from scratch'))

            with tf.name_scope('vocab'): # create vocabulary
                vocab = {
                    'x' : ph_frames.name,
                    'y' : ph_labels.name,
                    'init' : ds_init.name,
                    'logits' : logits.name,
                    'n_shuffle' : ph_n_shuffle.name,
                    'n_repeat' : ph_n_repeat.name,
                    'n_batch' : ph_n_batch.name,
                    'targets' : targets,
                    'sample_rate' : args.sample_rate
                    #'args' : vars(args)
                }
                if not os.path.exists(args.checkpoint_dir):
                    os.makedirs(args.checkpoint_dir)
                with open(os.path.join(args.checkpoint_dir, 'vocab.json'), 'w') as fp:
                    json.dump(vocab, fp)
                

            # training starts here..

            print_star('start training')

            blacklist = []

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            with tf.train.SingularMonitoredSession(config=config, scaffold=scaffold, hooks=[saver_hook, summary_hook], checkpoint_dir=args.checkpoint_dir) as sess:                      

                # prepare timing

                global_start_time = time.time()     
                batch_dur = args.n_batch * (args.n_step / args.sample_rate)                                                  

                # loop epochs
                
                for epoch in range(args.n_epochs):
                                                                       
                    loader = Loader(args, source, n_threads=args.n_fetch_threads) 
                    for iter, sample in loader:
            
                        file, frames, labels = sample

                        if file in blacklist:
                            print_info('skip {} [blacklisted]'.format(file))
                            continue

                        print_info('processing {} [{}]'.format(file, frames.shape[0]))                                                

                        try:                 
                            
                            print_prog('initializing dataset...')      
                            # https://github.com/tensorflow/tensorflow/issues/12859                     
                            sess._coordinated_creator.tf_sess.run(ds_init,
                                    feed_dict = {
                                    ph_frames : frames,
                                    ph_labels : labels,
                                    ph_n_shuffle : args.n_shuffle,
                                    ph_n_repeat: args.n_repeat,
                                    ph_n_batch: args.n_batch,
                                })
          
                            print_prog('running training step...')                                                  
                            log_print = lambda: \
                                print_prog('epoch={}/{} | progress={:.1f}% | step={} [{}] | time={} | loss={:.5f} | acc={:.2f}'.format(
                                    epoch+1, args.n_epochs, 
                                    100 * (iter / len(loader)),
                                    step, 
                                    datetime.timedelta(seconds=int(count*(args.n_step/args.sample_rate))), 
                                    datetime.timedelta(seconds=int(time.time()-global_start_time)), 
                                    step_loss_sum/local_step,
                                    step_accuracy_sum/local_step))

                            step_loss_sum = 0
                            step_accuracy_sum = 0
                            local_step = 0      

                            # common way, but does not work.. why?
                            #while not sess.should_stop():                               
                            while True:

                                try:                             
                                    
                                    step, count, step_loss, step_accuracy, _ = sess.run((global_step, global_frames, loss, accuracy, optimize))

                                    step_loss_sum += step_loss
                                    step_accuracy_sum += step_accuracy
                                    local_step += 1
                                      
                                    if step % args.n_log_steps == 0:                                                                                    
                                        log_print()                                        
                        
                                except tf.errors.OutOfRangeError:   
                                    break                                

                            log_print()  
                            print()

                            # log filename

                            if args.log_filename:                            
                                filename_tensor = tf.make_tensor_proto(file, dtype=tf.string)
                                filename_meta = tf.SummaryMetadata()
                                filename_meta.plugin_data.plugin_name = "text"
                                filename_summary = tf.Summary()
                                filename_summary.value.add(tag='log/filename', metadata=filename_meta, tensor=filename_tensor)
                                summary_writer.add_summary(filename_summary, step)
                            
                            # evaluation

                            if do_eval and iter % args.n_eval_files == 0:
                               
                                print_prog('initializing dataset...')      
                                # https://github.com/tensorflow/tensorflow/issues/12859    
                                sess._coordinated_creator.tf_sess.run(ds_init,
                                            feed_dict = {
                                            ph_frames : eval_frames,
                                            ph_labels : eval_labels,
                                            ph_n_shuffle : 1,
                                            ph_n_repeat: 1,
                                            ph_n_batch: args.n_batch,
                                        })
                            
                                print_prog('running evaluation step...')
                                count = 0                            
                                while True:
                                    try:                    
                                        output = sess.run(logits) 
                                        eval_predict[count:count+output.shape[0]] = np.argmax(output, axis=1)                                
                                        count += output.shape[0]                                    
                                    except tf.errors.OutOfRangeError:                                                                                
                                        break
                                eval_acc = np.sum(eval_labels == eval_predict) / len(eval_labels)
                                print_prog('evaluation accuracy is {:.2f}%'.format(100 * eval_acc))
                                print()

                                eval_summary = tf.Summary(value=[
                                    tf.Summary.Value(tag='log/evaluation', simple_value=eval_acc)
                                ])                                                                          
                                summary_writer.add_summary(eval_summary, step)

                                if args.eval_blacklist_thres and eval_acc < args.eval_blacklist_thres:
                                    print_info('add to blacklist'.format(file))
                                    blacklist.append(file)
                
                        except Exception as ex:
                            print_err('\n' + str(ex))
                                              
            summary_writer.close()



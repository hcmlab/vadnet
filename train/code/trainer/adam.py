import argparse
from typing import Optional, List

import tensorflow as tf

from enum import Enum
from trainer.base import TrainerBase


class SceAdam(TrainerBase):

    def get_args(self, parser:argparse.ArgumentParser):

        parser.add_argument('--learning_rate',
                default=0.0001,
                type=float,
                help='initial learning rate')

        parser.add_argument('--adam_epsilon',
                default=1e-08,
                type=float,
                help='constant for numerical stability')

        parser.add_argument('--adam_beta1',
                default=0.9,
                type=float,
                help='exponential decay rate for the 1st moment estimates')

        parser.add_argument('--adam_beta2',
                default=0.999,
                type=float,
                help='exponential decay rate for the 2nd moment estimates')

        return parser


    def get_name(self, args) -> str :
        return '{}[{:.0e},{},{},{}]'.format(type(self).__name__, args.learning_rate, args.adam_beta1, args.adam_beta2, args.adam_epsilon)


    def get_loss(self, args, logits:tf.Tensor, one_hot_labels:tf.Tensor) -> tf.Tensor:

        #https://stackoverflow.com/questions/47034888/how-to-choose-cross-entropy-loss-in-tensorflow
                
        #loss = tf.nn.weighted_cross_entropy_with_logits(one_hot_labels, logits, pos_weight=1)
        #loss = slim.losses.compute_weighted_loss(loss)               
        #total_loss = tf.losses.get_total_loss()               
                
        #loss = tf.losses.sigmoid_cross_entropy(one_hot_labels, logits)
        #loss = tf.losses.compute_weighted_loss(loss)     

        #loss = tf.nn.weighted_cross_entropy_with_logits(one_hot_labels, logits, pos_weight=1)
        #loss = slim.losses.compute_weighted_loss(loss)   
        #loss = tf.losses.get_total_loss()
        
        #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_labels, logits=logits))        

        #ratio = 0.7
        #class_weight = tf.constant([ratio, 1.0 - ratio])
        #loss = tf.losses.softmax_cross_entropy(one_hot_labels, logits)
        #loss = tf.losses.compute_weighted_loss(loss) 
        #loss = tf.losses.get_total_loss()

        #prediction = tf.nn.softmax(logits)
        loss = tf.losses.softmax_cross_entropy(one_hot_labels, logits)
        loss = tf.losses.compute_weighted_loss(loss) 
        loss = tf.losses.get_total_loss()

        return loss


    def get_accuracy(self, args, logits:tf.Tensor, one_hot_labels:tf.Tensor) -> tf.Tensor:

        correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(one_hot_labels,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return accuracy


    def get_optimize(self, args, loss:tf.Tensor, global_step:tf.Tensor) -> tf.Tensor:
                
        optimizer = tf.train.AdamOptimizer(args.learning_rate, beta1=args.adam_beta1, beta2=args.adam_beta2, epsilon=args.adam_epsilon)
        optimize = optimizer.minimize(loss, global_step=global_step)

        #beta1_power, beta2_power = optimizer._get_beta_accumulators()
        #update_lr = learning_rate * (1-beta2_power) ** 0.5 / (1-beta1_power)

        return optimize


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    trainer = SceAdam()
    trainer.get_args(parser)    
    args = parser.parse_args()

    print(args)
    print(trainer.get_name(args))     

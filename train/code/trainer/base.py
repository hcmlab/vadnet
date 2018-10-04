import abc, argparse
from typing import Optional, List

import tensorflow as tf

from utils.base import Base


class TrainerBase(Base):

    __metaclass__ = abc.ABCMeta


    @abc.abstractmethod
    def get_loss(self, args, logits:tf.Tensor, targets:tf.Tensor) -> tf.Tensor:
        pass


    @abc.abstractmethod
    def get_optimize(self, args, loss:tf.Tensor, global_step:tf.Tensor) -> tf.Tensor:
        pass


    @abc.abstractmethod
    def get_accuracy(self, args, logits:tf.Tensor, targets:tf.Tensor) -> tf.Tensor:
        pass
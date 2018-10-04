import abc
from typing import Optional, List

import tensorflow as tf

from utils.base import Base


class NetworkBase(Base):

    __metaclass__ = abc.ABCMeta


    @abc.abstractmethod
    def get_layers(self, args, inputs:tf.Tensor, n_targets:int) -> List[tf.Tensor]: 
        pass



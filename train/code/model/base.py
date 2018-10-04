import abc
from typing import Optional, List

import tensorflow as tf

from utils.base import Base


class ModelBase(Base):

    __metaclass__ = abc.ABCMeta


    @abc.abstractmethod
    def train(self, args, source:Base, network:Base, trainer:Base):
        pass

    @abc.abstractmethod
    def predict(self, args, path:str, network:Base) -> List:
        pass



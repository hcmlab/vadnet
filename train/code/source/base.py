import abc
from typing import Optional, List, Any

from utils.base import Base


class SourceBase(Base):

    __metaclass__ = abc.ABCMeta


    @abc.abstractmethod
    def setup(self, args) -> bool:
        pass


    @abc.abstractmethod
    def init(self, args) -> bool:
        pass


    @abc.abstractmethod
    def get_size(self, args) -> int:
        pass


    @abc.abstractmethod
    def get_targets(self, args) -> List:
        pass


    # return None if no more items are available
    # return [] if not a valid item and should be skipped
    # otherwise return item
    @abc.abstractmethod
    def next(self, args) -> Any:
        pass

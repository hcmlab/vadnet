import abc, argparse
from typing import Optional, List

import colorama
colorama.init(autoreset=True)


class Base(object):

    __metaclass__ = abc.ABCMeta


    @abc.abstractmethod
    def get_args(self, parser:argparse.ArgumentParser) -> argparse.ArgumentParser:
        pass


    @abc.abstractmethod
    def get_name(self, args) -> str:
        pass

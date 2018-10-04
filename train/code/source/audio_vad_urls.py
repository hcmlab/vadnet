import os, argparse, glob, threading, random
from typing import Optional, List, Any

from source.base import SourceBase
from source.loader import Loader
from utils.printy import print_err
from source.utils import sample_from_url, get_urls

                                                                                                                                             
class AudioVadUrls(SourceBase):

    
    lock = threading.Lock()
    files = []


    def get_args(self, parser:argparse.ArgumentParser) -> argparse.ArgumentParser:
        
        parser.add_argument('--urls_root',
                default='.',
                type=str,
                help='file with urls')

        parser.add_argument('--urls_filter',
                default='*',
                type=str,
                help='search filter')
        
        parser.add_argument('--urls_max',
                default=None,
                type=int,
                help='maximum number of urls')

        parser.add_argument('--urls_shuffle',
                default=True,
                type=lambda x: str(x).lower() == 'true',
                help='shuffle urls')

        return parser


    def setup(self, args) -> bool:
        pass


    def get_name(self, args) -> str:
        return 'vad'


    def get_size(self, args) -> int:
        return self.n_urls


    def get_targets(self, args) -> List:
        return ('noise', 'voice')


    def init(self, args) -> bool:        
        self.urls = get_urls(args.urls_root, args.urls_filter)                        
        if args.urls_shuffle:
            random.shuffle(self.urls)
        self.n_urls = len(self.urls)
        if args.urls_max and self.n_urls > args.urls_max:
            self.n_urls = args.urls_max
        self.counter = 0
        return self.counter <= self.n_urls


    def next(self, args) -> Any:

        with self.lock:
            if self.counter >= self.n_urls:
                return None
            url = self.urls[self.counter]
            self.counter += 1       
                    
        result = sample_from_url(url, sr=args.sample_rate, n_frame=args.n_frame, n_step=args.n_step, balance=args.balance)

        if not result:
            return []
        (audio, anno) = result

        return (url, audio, anno)



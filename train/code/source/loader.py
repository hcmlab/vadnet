import argparse, time, random, threading
from multiprocessing.pool import ThreadPool
from typing import Optional, List, Any

from source.base import SourceBase


class Loader(object):


    def __init__(self, args, source:SourceBase, n_threads=1):

        self.args = args
        self.source = source
        self.counter = 0       
        self.n_threads = n_threads     
        self.results = [None] * n_threads
        self.pool = ThreadPool(processes=n_threads)


    def __len__(self):
        return self.source.get_size(self.args)


    def __iter__(self):

        self.source.init(self.args)
        self.counter = 0
        self.fetch_counter = 0  
        for _ in range(self.n_threads): # fill queue
            self.fetch()
        
        return self


    def fetch(self):

        self.results[self.fetch_counter] = self.pool.apply_async(self.source.next, (self.args,))                     
        self.fetch_counter = (self.fetch_counter + 1) % self.n_threads

    
    def __next__(self):  
       
        while True:           
            counter = self.counter                                  
            result = self.results[self.counter % self.n_threads].get()
            if result is None: # no more items
                raise StopIteration                               
            self.fetch() # fetch next item
            self.counter += 1
            if result: # found valid item     
                break

        return counter, result


        
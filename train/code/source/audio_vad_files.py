import os, argparse, glob, threading, random
from typing import Optional, List, Any

from source.base import SourceBase
from source.loader import Loader
from utils.printy import print_err
from source.utils import sample_from_file

                                                                                                                                             
class AudioVadFiles(SourceBase):

    
    lock = threading.Lock()
    files = []


    def get_args(self, parser:argparse.ArgumentParser) -> argparse.ArgumentParser:
        
        parser.add_argument('--files_root',
                default='.',
                type=str,
                help='root folder containing the files')

        parser.add_argument('--files_filter',
                default='*',
                type=str,
                help='search filter')
        
        parser.add_argument('--files_audio_ext',
                default='.wav',
                type=str,
                help='audio file extension')

        parser.add_argument('--files_anno_ext',
                default='.annotation',
                type=str,
                help='annotation file extension')

        parser.add_argument('--files_max',
                default=None,
                type=int,
                help='maximum number of files')

        parser.add_argument('--files_shuffle',
                default=True,
                type=lambda x: str(x).lower() == 'true',
                help='shuffle files')

        return parser


    def setup(self, args) -> bool:
        pass


    def get_name(self, args) -> str:
        return 'vad'


    def get_size(self, args) -> int:
        return self.n_files


    def get_targets(self, args) -> List:
        return ('noise', 'voice')


    def init(self, args) -> bool:        
        self.files = glob.glob(os.path.join(args.files_root, args.files_filter))                        
        if args.files_shuffle:
            random.shuffle(self.files)
        self.n_files = len(self.files)
        if args.files_max and self.n_files > args.files_max:
            self.n_files = args.files_max
        self.counter = 0
        return self.counter <= self.n_files


    def next(self, args) -> Any:

        with self.lock:            
            if self.counter >= self.n_files:
                return None
            path = self.files[self.counter]
            self.counter += 1       
            
        path, _ = os.path.splitext(path)
        audio_path = path + args.files_audio_ext
        anno_path = path + args.files_anno_ext        

        result = sample_from_file(path, sr=args.sample_rate, n_frame=args.n_frame, n_step=args.n_step, audio_ext=args.files_audio_ext, anno_ext=args.files_anno_ext, balance=args.balance)

        if not result:
            return []
        (audio, anno) = result

        return (path, audio, anno)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()    
    
    source = AudioVadFiles()
    source.get_args(parser)     

    args = parser.parse_args()
    args.files_root = 'data'
    args.files_filter = '*.m4a'
    args.files_audio_ext = '.m4a'
    args.files_anno_ext = '.voiceactivity.annotation'
    args.files_shuffle = True
    args.sr = 44100
    args.n_frame = args.sr // 2
    args.n_step = args.n_frame // 2
    args.balance = None

    print(args)
    print(source.get_name(args))
    
    for i, result in Loader(args, source, n_threads=5):
        print('{} : {}'.format(i, result))
        


import os, argparse, glob, threading, random, time, json, csv, pickle
from typing import Optional, List, Any

import numpy as np
import librosa as lr

from source.base import SourceBase
from source.loader import Loader
from utils.printy import print_err, print_info
from source.utils import audio_from_files, audio_to_frames


def create_mapping(root, segments_file, exts=('.opus', '.m4a')):
    
    segments = {}

    with open(segments_file, 'r', newline='') as fp:
        reader = csv.reader(fp, delimiter=' ')                        
        for _ in range(3):
            next(reader)
        for i, row in enumerate(reader):                
            ids = row[3].split(',')
            for id in ids:
                if not id in segments:
                    segments[id] = set()
                for ext in exts:
                    file = os.path.join(root, '{}{}'.format(i, ext))         
                    if os.path.exists(file):
                        segments[id].add('{}{}'.format(i, ext))        
                        break

            if i % 1000 == 0:
                print('\r{:.2f}%'.format(100 * (i/2041792)), end='', flush=True)

    return segments


class AudioSetHelper:

    def __init__(self, root, ontology_file, quality_file, mapping_file):
        
        self.root = root
                
        with open(ontology_file, 'r') as fp:
            self.ontology = json.load(fp)

        self.quality = {}
        with open(quality_file, 'r', newline='') as fp:
            reader = csv.reader(fp, delimiter=',')                        
            next(reader)
            for row in reader:
                id = row[0]
                num_rated, num_true = int(row[1]), float(row[2])
                acc = 0 if num_rated == 0 else num_true / num_rated
                self.quality[id] = acc

        with open(mapping_file, 'rb') as fp:
            self.mapping = pickle.load(fp)                   


    def get_id_from_name(self, name):
        for entry in self.ontology:
            if entry['name'] == name:
                return entry['id']
        return None    


    def get_name_from_id(self, id):
        for entry in self.ontology:
            if entry['id'] == id:
                return entry['name']
        return None    


    def get_full_ids(self, id):
        ids = [id]
        for entry in self.ontology:
            if entry['id'] == id:
                child_ids = entry['child_ids']
                for child_id in child_ids:
                    ids.extend(self.get_full_ids(child_id))
        return set(ids)
        

    def filter_by_quality(self, ids, thres):
        new_ids = set()
        for id in ids:
            if id in self.quality and self.quality[id] > thres:
                new_ids.add(id)
        return new_ids
        

    def remove_by_names(self, ids, names):
        new_ids = set()
        for id in ids:
            if not self.get_name_from_id(id) in names:
                new_ids.add(id)
        return new_ids


    def get_files_from_ids(self, ids):
        if not ids:
            return None
        valid_ids = []
        for id in ids:
            if id in self.mapping:
                valid_ids.append(id)
        return set.union(*[self.mapping[x] for x in valid_ids])


    def get_ids_from_file(self, file):
        ids = set()
        for id, files in self.mapping.items():
            if file in files:
                ids.add(id)
        return ids


    def ids_from_confentry(self, entry):
        label = entry[0]
        names = entry[1]
        args = entry[2] if len(entry) > 2 else None                    
        ids = { self.get_id_from_name(name) for name in names }
        if args:
            if 'children' in args and args['children']:
                ids = set.union(*[self.get_full_ids(id) for id in ids])
            if 'threshold' in args and args['threshold']:                
                ids = set.union(*[self.filter_by_quality(ids, args['threshold'])])
            if 'blacklist' in args and args['blacklist']:                
                ids = self.remove_by_names(ids, args['blacklist'])
        return label, ids


    def read_conf(self, conf):

        task = conf[0]
        for entry in conf[1]:
            label, ids = self.ids_from_confentry(entry)                
            yield task, label, ids
        return 


    def get_audio(self, ids, sr):
        audios = []
        for file in files:
            audio_from_file(files[0], sr)


                                                                                                                                             
class AudioSet(SourceBase):

    
    lock = threading.Lock()


    def get_args(self, parser:argparse.ArgumentParser) -> argparse.ArgumentParser:
        
        parser.add_argument('--source_root',
                default='.',
                type=str,
                help='root folder containing the files')

        parser.add_argument('--source_config',
                default='audioset.conf',
                type=str,
                help='config file')
        
        parser.add_argument('--source_n',
                default=1000,
                type=int,
                help='number of samples that will be generated')

        parser.add_argument('--source_dur',
                default=3*60,
                type=int,
                help='seconds added from each class to a sample')
        
        parser.add_argument('--source_shuffle',
                default=True,
                type=lambda x: str(x).lower() == 'true',
                help='shuffle the frames within a sample')

        return parser


    def setup(self, args) -> bool:
        
        helper = AudioSetHelper(os.path.join(args.source_root, 'files'),
                os.path.join(args.source_root, 'ontology.json'),
                os.path.join(args.source_root, 'qa_true_counts.csv'),
                os.path.join(args.source_root, 'mapping.pickle'))
        self.targets = []   
        self.files = []   

        with open(args.source_config, 'r') as fp:
            config = json.load(fp)
     
        for task, name, ids in helper.read_conf(config):    

            self.task = task    
            self.targets.append(name) 
            self.files.append(helper.get_files_from_ids(ids))     

            print_info(name)
            for id in ids:
                print_info('\t{}\t{}'.format(id, helper.get_name_from_id(id)))

        self.n_targets = len(self.targets)


    def get_name(self, args) -> str:
        return self.task


    def get_size(self, args) -> int:
        return args.source_n


    def get_targets(self, args) -> List:
        return self.targets


    def init(self, args) -> bool: 
               
        self.counter = 0
        self.targets = self.get_targets(args)
        self.n_targets = len(self.targets)        
        self.n_chunk = (args.sample_rate*args.source_dur)//args.n_step
        self.n_sample = self.n_chunk*self.n_targets
        
        return self.counter <= args.source_n


    def get_chunk(self, files, label, args):

        audio = audio_from_files(files, args.sample_rate, root=os.path.join(args.source_root, 'files'), duration=args.source_dur, shuffle=True)
        frames = audio_to_frames(audio, args.n_frame, args.n_step)
        labels = np.full((frames.shape[0],), label, dtype=np.int32)

        return frames, labels


    def next(self, args) -> Any:

        with self.lock:            
            if self.counter >= args.source_n:
                return None            
            self.counter += 1       

        frames = []
        labels = []
        for label, files in enumerate(self.files):
            chunk = self.get_chunk(files, label, args)
            frames.append(chunk[0])
            labels.append(chunk[1])
        
        frames = np.concatenate(frames)
        labels = np.concatenate(labels)
                
        if args.source_shuffle:
            perm = np.random.permutation(len(labels))
            frames = frames[perm,:]
            labels = labels[perm]        
        
        return ('randomly generated sample', frames, labels)

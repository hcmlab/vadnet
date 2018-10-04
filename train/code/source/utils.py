import sys, os, csv, glob, random, threading, time, enum

import numpy as np
import librosa as lr
import youtube_dl

from utils.printy import print_err, print_prog

#sys.path.insert(0, '../../download')
from playlist import parse_list, Header
from annotation import write_voiceactivity 
from subtitles import parse_subtitles


def anno_from_file(path, n, sr, ext=''):        
    
    labels = np.zeros((n,), dtype=np.int32)

    with open('{}{}~'.format(path, ext), newline='') as fp:
        rows = csv.reader(fp, delimiter=';')      
        for row in rows:
            start = int(round(float(row[0]) * sr))
            stop = int(round(float(row[1]) * sr))
            labels[start:stop] = 1

    return labels


def anno_to_file(path, labels, sr):
   
    time = 0
    step = 1/sr

    in_label = labels[0] == 1        
    onset = 0
    count = 0

    with open('{}~'.format(path), 'w', encoding='latin-1') as fp:        

        for i in range(len(labels)):    
            if in_label and labels[i] == 0:                  
                count += 1              
                fp.write('{};{};0;1.0\n'.format(onset, time)) 
                in_label = False
            elif not in_label and labels[i] == 1:
                onset = time
                in_label = True
            time += step

        if in_label:
            count += 1              
            fp.write('{};{};0;1.0\n'.format(onset, time)) 

    with open('{}'.format(path), 'w', encoding='latin-1') as fp:
        fp.write('<?xml version="1.0" ?>\n<annotation ssi-v="3">\n\t<info ftype="ASCII" size="{}" />\n\t<meta role="subtitles" annotator="system" />\n\t<scheme name="voiceactivity" type="DISCRETE" color="#FFDDD9C3">\n\t\t<item name="VOICE" id="0" color="#FF494429" />\n\t</scheme>\n</annotation>\n'.format(count))           


def audio_dur(path, ext='', root=''):
    path = os.path.join(root, '{}{}'.format(path, ext))
    try:
        return lr.get_duration(filename=path)
    except Exception as ex:        
        print_err('could not read {}\n{}'.format(path, ex))
        return 0


def audio_from_file(path, sr, ext='', root='', offset=0.0, duration=None):
    path = os.path.join(root, '{}{}'.format(path, ext))
    try:
        audio, _ = lr.load(path, sr=sr, mono=True, offset=offset, duration=duration, dtype=np.float32, res_type='kaiser_fast') 
        audio.shape = (-1,1)
        return audio
    except ValueError as ex:
        print_err('value error {}\n{}'.format(path, ex))
        return []
    except Exception as ex:        
        print_err('could not read {}\n{}'.format(path, ex))
        return None


def audio_from_files(paths, sr, ext='', root='', duration=None, shuffle=False):
    
    if shuffle:
        paths = list(paths)
        random.shuffle(paths)

    audios = []
    for path in paths:
        if duration:
            dur = audio_dur(path, ext=ext, root=root)  
            audio = audio_from_file(path, sr, ext=ext, root=root, duration=duration if dur > duration else None)              
            if audio is None:
                continue
            if len(audio):
                audios.append(audio)
            duration -= dur            
            if duration <= 0:                
                break
        else:
            audio = audio_from_file(path, sr, ext=ext, root=root)
            if audio is None:
                continue
            if len(audio):
                audios.append(audio)

    return np.concatenate(audios) if audios else None
   

def audio_to_file(path, x, sr):    
    lr.output.write_wav(path, x.reshape(-1), sr, norm=False)   
    

def audio_to_frames(x, n_frame, n_step=None):    

    if n_step is None:
        n_step = n_frame

    if len(x.shape) == 1:
        x.shape = (-1,1)

    n_overlap = n_frame - n_step
    n_frames = (x.shape[0] - n_overlap) // n_step       
    n_keep = n_frames * n_step + n_overlap

    strides = list(x.strides)
    strides[0] = strides[1] * n_step

    return np.lib.stride_tricks.as_strided(x[0:n_keep,:], (n_frames,n_frame), strides)
    

def sample_from_file(path, sr, n_frame, n_step, audio_ext='.m4a', anno_ext='.annotation', balance=None):

    audio_path = '{}{}'.format(path, audio_ext)
    anno_path = '{}{}'.format(path, anno_ext)

    if not os.path.exists(audio_path) or not os.path.exists(anno_path):
        print_err('file not found {}[{},{}]'.format(path, audio_ext, anno_ext))
        return None

    audio = audio_from_file(audio_path, sr)
    if audio is None or audio.size == 0:
        return None

    frames = audio_to_frames(audio, n_frame, n_step)
    labels = anno_from_file(anno_path, frames.shape[0], sr/n_step)  
    
    if balance:
        select = get_balance_indices(labels, balance)
        frames = frames[select,:]
        labels = labels[select]  

    return frames, labels


def sample_from_url(url, sr, n_frame, n_step, balance=None):

    class MyLogger(object):
        def debug(self, msg):
            pass
        def warning(self, msg):
            pass
        def error(self, msg):
            print_err(msg)

    ydl_opts = {
        'ignoreerrors' : True,
        'outtmpl': r'tmp\%(id)s.%(ext)s',
        'format': 'bestaudio/best',    
        'writesubtitles' : True,
        'logger' : MyLogger()
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:

        info_dict = ydl.extract_info(url, download=False)
        if not info_dict:
            print_prog('skip {} [not found]'.format(url))
            return None
        if not 'subtitles' in info_dict or not info_dict['subtitles']:
            print_prog('skip {} [no subtitles]')
            return None
        
        audio_ext = '.' + info_dict['ext']
        anno_ext = '.annotation'
        path = os.path.join('tmp', info_dict['id'])
        
        print_prog('downloading {}'.format(path))        
        ydl.download((url,)) 

        result = None

        if os.path.exists(path + audio_ext):

            sub_path = glob.glob(r'{}.*.*'.format(path))            
            if sub_path:                
            
                print_prog('parsing {}'.format(path)) 
                subs = parse_subtitles(sub_path[0])
                if subs is not None:
                    write_voiceactivity(path, subs)                                 
                    result = sample_from_file(path, sr, n_frame, n_step, audio_ext, anno_ext, balance)
            
        for file in glob.glob('{}.*'.format(path)):
            try:              
                os.remove(file)                    
            except Exception as ex:
                print_err(ex)

        return result


def get_files(root, filter='*', max_files=None):

    files = []
    
    count = 0
    for file in glob.glob(os.path.join(root, filter)):
        if max_files is not None and count >= max_files:
            break
        path, _ = os.path.splitext(file)
        files.append(path)
        count = count + 1
        
    return files


def get_urls(path, filter='', max_urls=None):

    table = parse_list(path)
    
    urls = []
    for item in table:
        if item and filter in item[Header.Url_Untertitel]:                            
            urls.append(item[Header.Website])

    if max_urls:
        urls = urls[:min(len(urls), max_urls)]

    return urls


class BalanceMethod(enum.Enum): 
    Down, Up = range(2)

def get_balance_indices(labels,method):

    n_classes = np.max(labels)+1
    select = None
    
    if method is not None:

        initial_n = [np.sum(labels==i) for i in range(n_classes)]
        if np.min(initial_n) == 0:
            return []
        target_n = np.max(initial_n) if method == BalanceMethod.Up else np.min(initial_n)        
        select = np.zeros((n_classes * target_n,), dtype=np.int64)
        for i in range(n_classes):                        
            select[i*target_n:i*target_n+target_n] = np.random.choice(np.where(labels==i)[0], target_n, replace=method == BalanceMethod.Up)   
        np.random.shuffle(select)                
        
    return select


if __name__ == '__main__':
    

    # framing

    x = np.arange(0,10)
    
    print(x)
    print(audio_to_frames(x, 3))    
    print(audio_to_frames(x, 3, n_step=2))
    print(audio_to_frames(x, 4, n_step=2))

    # sample from file

    sr = 44100
    n_frame = 44100
    n_step = 44100 // 2

    files = get_files('data', filter='*.m4a')   
    frames, labels = sample_from_file(files[0], sr, n_frame, n_step, audio_ext='.m4a', anno_ext='.voiceactivity.annotation')    
    audio_to_file(r'data\test1.wav', frames[labels == 1], sr)    

    # sample from url
    
    urls = get_urls(r'..\youtube\filme.json', filter='ZDF')  
    frames, labels = sample_from_url(urls[0], sr, n_frame, n_step)    
    audio_to_file(r'data\test2.wav', frames[labels == 1], sr)    

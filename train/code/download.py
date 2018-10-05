from __future__ import unicode_literals
import youtube_dl
import shutil
import os
import glob
import json
import multiprocessing

from playlist import parse_list, download_list, Header

#from subtitles import parse_subtitles
#from annotation import write_annotation


DOWNLOAD_DIR = 'data'
INFO_EXT = 'info'


def is_list(object):
    return type(object) is list or type(object) is set or type(object) is tuple        


class MyLogger(object):

    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        print(msg)


def my_hook(d):

    id, _ = os.path.splitext(d['filename'])
    
    if d['status'] == 'finished':                
        #subs_path = glob.glob('{}.*.*'.format(id))
        #if subs_path:
        #    print('{}: extracting subtitles'.format(id))
        #    subs = parse_subtitles(subs_path[0])
        #    write_annotation(id, subs)
        print('{}: saving info'.format(id))
        os.rename('{}.{}~'.format(id, INFO_EXT), '{}.{}'.format(id, INFO_EXT))
        print('{}: converting audio'.format(id))        
    else:
        print('{}: {}\r'.format(id, d['_percent_str']), end="", flush=True)


default_opts = {
    'ignoreerrors' : True,
    'outtmpl': '{}\\%(extractor_key)s_%(id)s.%(ext)s'.format(DOWNLOAD_DIR),
    'format': 'bestaudio/best',    
    'writesubtitles' : True,    
    #'writeautomaticsub' : True,
    'download_archive' : 'archive',
    'logger': MyLogger(),
    'progress_hooks': [my_hook],    
    #'postprocessors': [{
    #    'key': 'FFmpegExtractAudio',
    #    'preferredcodec': 'mp3',
    #    'preferredquality': '192',          
    #}]
    #'postprocessors': [{
    #    'key': 'FFmpegExtractAudio',
    #    'preferredcodec': 'flac',        
    #}]
}


def chunks(l, n):    
    for i in range(0, len(l), n):
        yield l[i:i + n]


def download(urls, n_parallel=0, ydl_opts=default_opts, subtitles_only=False):
    
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    if not is_list(urls):
        urls = (urls,)

    if n_parallel <= 1 or len(urls) < n_parallel:
        _download(urls, ydl_opts=ydl_opts, subtitles_only=subtitles_only)
    else:
        pool = multiprocessing.Pool(n_parallel)
        tasks = []
        n_per_task = len(urls) // n_parallel
        for sub_urls in chunks(urls, n_per_task):
            tasks.append((sub_urls, ydl_opts, subtitles_only))
        results = [pool.apply_async(_download, t) for t in tasks]
        output = [p.get() for p in results]
                    
    print('finished!')    


def _download(urls, ydl_opts=default_opts, subtitles_only=False):

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:                      

        print('checking urls [{}]'.format(len(urls)))

        valid_urls = []
        for url in urls:
            print(url.encode('latin-1'), end="", flush=True)
            info_dict = ydl.extract_info(url, download=False)
            if not info_dict:
                print(' skip [not found]')
                continue
            if subtitles_only:
                if not 'subtitles' in info_dict or not info_dict['subtitles']:
                    print(' skip [no subtitles]')
                    continue
            print(' ok')
            with open(os.path.join(DOWNLOAD_DIR, '{}_{}.{}~'.format(info_dict['extractor_key'], info_dict['id'], INFO_EXT)), 'w') as fp:
                json.dump(info_dict, fp)
            valid_urls.append(url)
    
        print('start downloading [{}]'.format(len(valid_urls)))

        if valid_urls:                   
            ydl.download(valid_urls)    


def download_zdf_serien():

    urls = []

    download_list()

    for item in parse_list('filme.json'):
        if item and 'serien' in item[Header.Website] and 'utstreaming.zdf' in item[Header.Url_Untertitel]:            
            urls.append(item[Header.Website])

    print('found {}'.format(len(urls)))

    download(urls, n_parallel=10, subtitles_only=True)

    

if __name__ == '__main__':


    download_zdf_serien()

    #urls = ['https://www.youtube.com/watch?v=5AfEBjvfDYc',
    #    'https://www.youtube.com/watch?v=FyH6vOMYSnY',
    #    'https://www.youtube.com/watch?v=P9x0wt7P1-0',
    #    'https://www.youtube.com/watch?v=T2a-xIN6FKk',
    #    'https://www.youtube.com/watch?v=rFwxhKrxthA',
    #    'https://www.youtube.com/watch?v=0PVuEGlPI7U',
    #    'https://www.youtube.com/watch?v=2tZLUxJ8Lqs',
    #    'https://www.youtube.com/watch?v=wq9Uax7taWA',
    #    'https://www.youtube.com/watch?v=eNuIzMZ_SSs']
    #download(urls, n_parallel=len(urls), subtitles_only=False)

    
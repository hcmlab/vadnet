from __future__ import unicode_literals
import urllib.request
import urllib.parse
import re
import json
from pprint import pprint
import ast
import lzma
import urllib.request


class Header:
    Size = 20
    Sender, Thema, Titel, Datum, Zeit, Dauer, Groesse, Beschreibung, Url, Website, Url_Untertitel, RTMP, Url_Klein, Url_RTMP_Klein, Url_HD, Url_RTMP_HD, DatumL, Url_History, Geo, neu = range(20)


def download_list(url='http://verteiler5.mediathekview.de/Filmliste-akt.xz', path='filme.json', tmp_path='filme.xz'):        
    
    print('download {}'.format(url))

    with urllib.request.urlopen(url) as response, \
        open(tmp_path, 'wb') as tmp_fp, \
        open(path, 'wb') as fp:

        data = response.read()
        tmp_fp.write(data)
        data = lzma.decompress(data)
        fp.write(data)
            

def parse_list(path):

    print('parsing {}'.format(path))

    table = None

    with open(path, 'r', encoding='latin-1') as fp:                
        content = fp.read()
        entries = content[:-1].split('"X":') # get rid of last }
        entries.pop(0) # get rid of header
        table = [None,]*len(entries)
        for idx, entry in enumerate(entries):                             
            table[idx] = ast.literal_eval(entry)[0]
            
    return table


if __name__ == '__main__':

    download()

    table = parse_list('filme.json')
    
    urls = []
    for item in table:
        if item and 'zdf' in item[Header.Url_Untertitel]:                            
            urls.append(item[Header.Website])
        
    print(urls)
    print(len(urls))

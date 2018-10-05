import os

from pycaption import detect_format, WebVTTReader, DFXPReader


def parse_subtitles(path):

#    print('parsing {}'.format(path))

    caps = None

    if os.path.exists(path):
        with open(path, 'r', encoding='latin-1') as fp:
            content = fp.read().replace('<tt:', '<').replace('</tt:', '</')
            reader = detect_format(content)
            if reader:
                try:
                    set = reader().read(content)        
                    caps = set.get_captions(set.get_languages()[0])
                except Exception as ex:
                    print('ERROR: {}'.format(ex))
            
    return caps


if __name__ == '__main__':    

    for path in ('yXPrLGUGZsw.en.vtt', '51466206.de.ttml'):    
        caps = parse_subtitles(path)
        for cap in caps:
            print('{} {} {}'.format(cap.format_start(), cap.format_end(), cap.get_text().encode('latin-1')))            
    
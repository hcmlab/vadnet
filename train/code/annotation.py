from subtitles import parse_subtitles
from datetime import datetime, timedelta
import html
import re
import os
import glob


def convert_timestamp_to_s(str):

    if re.match('\d\d\d:', str):  # handle strings like 000:00:00.000
        str = str[1:]
    if re.match('.*\.\d\d$', str): # handle strings like 00:00:00.00
        str = str + '0'
    if str[0] != '0': # handle strings like 10:00:00.000
        str = '0' + str[1:]
    dt = datetime.strptime(str, '%H:%M:%S.%f')
    ms = (dt - datetime(1900, 1, 1)) // timedelta(milliseconds=1)
    s = ms / 1000.0

    return s


def write_transcription(path, subs):
        
    if not path.endswith('.annotation'):
        path += '.annotation'

#    print('writing {}'.format(path))

    count = 0
    
    with open(path + '~', 'w', encoding='latin-1') as fp:        
        for sub in subs:    
            if not sub:
                continue
            count += 1
            start = convert_timestamp_to_s(sub.format_start())
            end = convert_timestamp_to_s(sub.format_end())  
            cap = html.unescape(sub.get_text().replace('\n', ' ').replace(';', ','))
            fp.write('{};{};{};1.0\n'.format(start, end, cap)) 

    with open(path, 'w', encoding='latin-1') as fp:
        fp.write('<?xml version="1.0" ?>\n<annotation ssi-v="3">\n\t<info ftype="ASCII" size="{}" />\n\t<meta role="youtube" annotator="system" />\n\t<scheme name="transcription" type="FREE"/>\n</annotation>\n'.format(count))   
            

def write_voiceactivity(path, subs):

    if not path.endswith('.annotation'):
        path += '.annotation'

#    print('writing {}'.format(path))

    count = 0

    with open(path + '~', 'w', encoding='latin-1') as fp:        
        for sub in subs:      
            if not sub:
                continue
            count += 1
            start = convert_timestamp_to_s(sub.format_start())
            end = convert_timestamp_to_s(sub.format_end())  
            cap = html.unescape(sub.get_text().replace('\n', ' ').replace(';', ','))
            if not re.match('\s*\*.*\*\s*', cap) \
                and not re.match('\s*\[.*\]\s*', cap):                
                fp.write('{};{};0;1.0\n'.format(start, end)) 

    with open(path, 'w', encoding='latin-1') as fp:
        fp.write('<?xml version="1.0" ?>\n<annotation ssi-v="3">\n\t<info ftype="ASCII" size="{}" />\n\t<meta role="subtitles" annotator="system" />\n\t<scheme name="voiceactivity" type="DISCRETE" color="#FFDDD9C3">\n\t\t<item name="VOICE" id="0" color="#FF494429" />\n\t</scheme>\n</annotation>\n'.format(count))   



if __name__ == '__main__':    

    force = False
    files = glob.glob(os.path.join(r'data', '*.info'))

    for file in files:

        print(file)

        name = os.path.basename(file)
        path, _ = os.path.splitext(file)
        sub_path = glob.glob(r'{}.*.*'.format(path))[0]
        vad_path = path + '.voiceactivity'
        transcr_path = path + '.transcription'

        if not force and os.path.exists(vad_path + '.annotation') and os.path.exists(transcr_path + '.annotation'):        
            continue

        subs = parse_subtitles(sub_path)

        if force or not os.path.exists(vad_path + '.annotation'):        
            write_voiceactivity(vad_path, subs)
        if force or not os.path.exists(transcr_path + '.annotation'):        
            write_transcription(transcr_path, subs)

        #break
    
import glob
import os
import re
import subprocess


if __name__ == '__main__':    
    
    files = glob.glob(os.path.join('data', '*.mp4'))

    for file in files:
        print(file)

        with subprocess.Popen('ffprobe -show_format {}'.format(file), shell=False, stdout=subprocess.PIPE,  stderr=subprocess.PIPE) as proc:
            info = proc.stderr.read()
        
        #  Stream #0:0(und): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 96 kb/s (default)'
        for line in str(info).split(r'\r\n'):                    
            match = re.match('\s*Stream.*Audio', line)
            if match:
                props = line.split(',')
                print(props[1].lstrip().rstrip())
        


import os
import glob
import shutil


if __name__ == '__main__': 

    source_dir = r'downloads\serien\ZDF'
    target_dir = r'X:\nova\data\mediathek'

    exts = ('m4a', 'transcription.annotation', 'transcription.annotation~', 'voiceactivity.annotation', 'voiceactivity.annotation~')
    user = 'clip'    
    prefix = 'serie_'

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    files = glob.glob(os.path.join(source_dir, '*.info'))
    
    for file in files:
        print(file)

        id, _ = os.path.splitext(os.path.basename(file))
        for ext in exts:
            source_path = os.path.join(source_dir, '{}.{}'.format(id, ext))
            target_path = os.path.join(target_dir, '{}{}'.format(prefix, id), '{}.{}'.format(user, ext))
            if not os.path.exists(os.path.dirname(target_path)):
                os.makedirs(os.path.dirname(target_path))
            shutil.copy(source_path, target_path)            



import platform, os, shutil, argparse
from importlib import import_module

#from source.audio_urbansound import UrbanSound as Source
#from source.audio_vad_files import AudioVadFiles as Source
#from source.audioset import AudioSet as Source

from source.utils import anno_to_file, BalanceMethod
from utils.printy import print_info


parser = argparse.ArgumentParser(allow_abbrev=False)

parser.add_argument('--postfix',
        default='',
        type=str,
        help='an optional string appended to the name of the experiment')

parser.add_argument('--network',
        default='',
        type=str,
        help='name of network class')

parser.add_argument('--model',
        default='',
        type=str,
        help='name of model class')

parser.add_argument('--trainer',
        default='',
        type=str,
        help='name of trainer class')

parser.add_argument('--source',
        default='',
        type=str,
        help='name of source class')


def instance(name):
    tokens = name.split('.')
    module_name = '.'.join(tokens[:-1])
    class_name = tokens[-1]
    module = import_module(module_name)
    return getattr(module, class_name)()


def main():    

    args, _ = parser.parse_known_args()

    source = instance(args.source)
    source.get_args(parser)    

    model = instance(args.model)
    model.get_args(parser)

    network = instance(args.network)
    network.get_args(parser)

    trainer = instance(args.trainer)
    trainer.get_args(parser)

    args = parser.parse_args()
    source.setup(args)
    
    if not args.exp_name:
        args.exp_name = '{}_{}_{}_{}{}'.format(
            source.get_name(args),
            model.get_name(args),         
            network.get_name(args),
            trainer.get_name(args),
            args.postfix)
    
    print_info(args)

    args.log_dir = os.path.join(args.output_dir, args.exp_name, 'log')
    args.checkpoint_dir = os.path.join(args.output_dir, args.exp_name, 'ckpt')

    if not args.retrain and os.path.exists(args.log_dir): shutil.rmtree(args.log_dir)
    if not args.retrain and os.path.exists(args.checkpoint_dir): shutil.rmtree(args.checkpoint_dir)
         
    model.train(args, source, network, trainer)


if __name__ == '__main__':
    main()
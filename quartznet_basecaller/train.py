import os
import torch
import toml

from .util import init, MODEL, load_model
from .model import Model
def train(training_directory: str, pretrained_weights_dir:str=None, model_config_file: str=None):
    workdir = os.path.expanduser(training_directory)

    if not os.path.exists(workdir):
        print(f"[ERROR] {workdir} does not exists. Training stopped.")
        exit(-1)
    
    # GPU and randomness initialization
    init(42, 'cuda')
    device = torch.device('cuda')

    # checking pretraining options
    if not pretrained:
        # get the model toml 
        config = toml.load(model_config_file)
    else:
        dirname = pretrained
        # check if the weights folder exists
        if not os.path.isdir(dirname):
            print(f"[ERROR]: the specified directory {workdir}, with the pretrained weights does not exists. Training stopped.")
            exit(-2)
        config = toml.load(os.path.join(dirname, f'{MODEL}.toml'))

        if 'lr_scheduler' in config:
            print(f"[ignoring 'lr_scheduler' in --pretrained config]")
            del config['lr_scheduler']

        if pretrained_weights_dir:
            print(f"[INFO]: using pretrained model {pretrained_weights_dir}.")
            model = load_model(pretrained_weights_dir, device, half=False)
        else:
            model = Model(config)
        
        print("[loading data]")

        



import torch
import numpy as np
import os
import toml

from torch.cuda import get_device_capability
from .model import Model
from collections import OrderedDict

MODEL = 'dna_r9.4.1@v1'

def init(seed, device, deterministic=True):
    """
    Initialise random libs and setup cudnn

    https://pytorch.org/docs/stable/notes/randomness.html
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cpu": 
        print("[ERROR] no GPU available. The proccess will be stopped")
        return
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = (not deterministic)
    assert(torch.cuda.is_available())


def set_config_defaults(config, chunksize=None, batchsize=None, overlap=None, quantize=False):
    basecall_params = config.get("basecaller", {})
    # use `value or dict.get(key)` rather than `dict.get(key, value)` to make
    # flags override values in config
    basecall_params["chunksize"] = chunksize or basecall_params.get("chunksize", 4000)
    basecall_params["overlap"] = overlap if overlap is not None else basecall_params.get("overlap", 500)
    basecall_params["batchsize"] = batchsize or basecall_params.get("batchsize", 64)
    basecall_params["quantize"] = basecall_params.get("quantize") if quantize is None else quantize
    config["basecaller"] = basecall_params
    return config

def load_model(dirname, device, weights=None, half=None, chunksize=None, batchsize=None, overlap=None, quantize=False, use_koi=False):
    if not os.path.isfile(os.path.join(dirname, f'config.toml')):
        print(f"[ERROR] no toml file found at {dirname}.")
        return(-1)
    weights = os.path.join(dirname, 'weights_1.tar')
    config = toml.load(os.path.join(dirname, 'config.toml'))
    config = set_config_defaults(config, chunksize, batchsize, overlap, quantize)

    return _load_model(weights, config, device, half, use_koi)

def _load_model(weights_file, config, device, half=None, use_koi=False):
    device = torch.device(device)
    model = Model(config)

    config["basecaller"]["chunksize"] -= config["basecaller"]["chunksize"] % model.stride
    # overlap must be even multiple of stride for correct stitching
    config["basecaller"]["overlap"] -= config["basecaller"]["overlap"] % (model.stride * 2)

    if use_koi:
        model.use_koi(
            batchsize=config["basecaller"]["batchsize"],
            chunksize=config["basecaller"]["chunksize"],
            quantize=config["basecaller"]["quantize"],
        )

    state_dict = torch.load(weights_file, map_location=device)
    state_dict = {k2: state_dict[k1] for k1, k2 in match_names(state_dict, model).items()}
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    if half is None:
        half = half_supported()

    if half: 
        model = model.half()

    model.eval()
    model.to(device)
    return model

def match_names(state_dict, model):
    keys_and_shapes = lambda state_dict: zip(*[
        (k, s) for s, i, k in sorted([(v.shape, i, k)
        for i, (k, v) in enumerate(state_dict.items())])
    ])
    k1, s1 = keys_and_shapes(state_dict)
    k2, s2 = keys_and_shapes(model.state_dict())
    assert s1 == s2
    remap = dict(zip(k1, k2))
    return OrderedDict([(k, remap[k]) for k in state_dict.keys()])

def half_supported():
    """
    Returns whether FP16 is support on the GPU
    """
    try:
        return get_device_capability()[0] >= 7
    except:
        return False

import os
import time
import torch
import numpy as np

from itertools import starmap
from pathlib import Path

from .data import load_numpy
from .util import init, load_model, concat, permute
from .util import half_supported, accuracy, decode_ref
from .util import SEED, DEVICE

from torch.utils.data import DataLoader


def evaluate(model_directory:str, directory:Path, weights:str="1", chunks:int=1000, batch_size=96, poa_:bool=False, min_covarage:float=0.5):
    poas = []
    # initialize random and cuda
    init()
    
    print("*\tLoading data...")
    try: 
        _, valid_loader_kwargs = load_numpy()
    except FileNotFoundError:
        print(f"[ERROR] cannot load data from {directory}.")
        return None

    dataloader = DataLoader(
        batch_size=batchsize,
        num_workers=4,
        pin_memory=True,
        **valid_loader_kwargs
    )

    accuracy_with_cov = lambda ref, seq: accuracy(ref, seq, min_coverage=min_covarage)

    for w in [int(i) for i in weights.split(',')]:
        seqs = []

        print("*\tLoading model", w)
        model = load_model(model_directory, DEVICE, weights=w)

        print("*\tCalling")
        t0 = time.perf_counter()

        targets = []

        with torch.no_grad():
            for data, target, *_ in dataloader:
                target.extend(torch.unbind(target, 0))
                if half_supported():
                    data = data.type(torch.float(16)).to(DEVICE)
                else:
                    data = data.to(DEVICE)
                
                log_probs = model(data)

                if hasattr(model, 'decode_batch'):
                    seqs.extend(model.decode_batch(log_probs))
                else:
                    seqs.extend([model.decode(p) for p in permute(log_probs, 'TNC', 'NTC')])
        duration = time.perf_counter() - t0

        refs = [decode_ref(target, model.alphabet) for target in targets]
        accuracies = [accuracy_with_cov(ref, seq) if len(seq) else 0. for ref, seq in zip(refs, seqs)]

        if poa_:
            poas.append(sequences)
        print(f"*\t\tmean\t:{np.mean(accuracies):.2f}")
        print(f"\t\tmedian\t:{np.median(accuracies):.2f}")
        print(f"*\t\ttime\t:{duration:.2f}")
        print(f"*\t\tsamples:\t{(args.chunks * data.shape[2] / duration):.2f}")
    if poa_:
        print("*\tDoing poa")
        t0 = time.perf_counter()
        # group each sequence prediction per model together
        poas = [list(seq) for seq in zip(*poas)]
        consensuses = poa(poas)
        duration = time.perf_counter() - t0
        accuracies = list(starmap(accuracy_with_coverage_filter, zip(references, consensuses)))

        print("* mean      %.2f%%" % np.mean(accuracies))
        print("* median    %.2f%%" % np.median(accuracies))
        print("* time      %.2f" % duration)
        
        
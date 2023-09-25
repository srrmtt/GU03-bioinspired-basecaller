import torch
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from bonitosnn.model.snn_model import BonitoSNNModel, BonitoSpikeConv, BonitoSpikeLin #as Model
from bonito.model import BonitoModel

#FLOP estimation
from fvcore.nn import FlopCountAnalysis

from classes import BaseNanoporeDataset
from constants import NON_RECURRENT_DECODING_DICT, NON_RECURRENT_ENCODING_DICT, RECURRENT_DECODING_DICT, RECURRENT_ENCODING_DICT
from torch.utils.data import DataLoader

if __name__=="__main__":
    """
    Experimental script that estimate the memory, flops, sops and energy footprint
    of all the proposed models
    """

    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    
    decoding_dict = NON_RECURRENT_DECODING_DICT
    encoding_dict = NON_RECURRENT_ENCODING_DICT
    dataset = BaseNanoporeDataset(
        data_dir = "/root/Acinetobacter_baumannii_AYP-A2/train_numpy_resquiggled", 
        decoding_dict = decoding_dict, 
        encoding_dict = encoding_dict, 
        split = 1, 
        shuffle = True, 
        seed = 1,
        s2s = False,
    )
    dataloader_train = DataLoader(
        dataset, 
        batch_size = 4, 
        sampler = dataset.train_sampler, 
        num_workers = 1
    )
    
    model_snn = BonitoSNNModel(
        load_default = True,
        device = device,
        dataloader_train = dataloader_train, 
        dataloader_validation = "", 
        scaler = None,
        use_amp = False,
        nlstm=0,
        conv_threshold=0.05,
        slstm_threshold=0.05
    ).to(device)
    #model = model.to(device)

    model_snn.load("/root/checkpoint_trial_3.pt", initialize_lazy = True)
    model_snn.to(device)

    param_size = 0
    for param in model_snn.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model_snn.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model slstm size: {:.3f}MB'.format(size_all_mb))

    model_bonito = BonitoModel(
        load_default = True,
        device = device,
        dataloader_train = "", 
        dataloader_validation = "", 
        scaler = None,
        use_amp = False,
        nlstm=0,
        conv_threshold=0.05,
        slstm_threshold=0.05
    )

    param_size = 0
    for i,param in enumerate(model_bonito.parameters()):
        print(i,") ",param.shape )
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model_bonito.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('bonito model size: {:.3f}MB'.format(size_all_mb))

    model_spikeconv = BonitoSpikeConv(
        load_default = True,
        device = device,
        dataloader_train = "", 
        dataloader_validation = "", 
        scaler = None,
        use_amp = False,
        nlstm=0,
        conv_threshold=0.05,
        slstm_threshold=0.05
    ).to(device)

    model_spikeconv.load("/root/test_nni_spikeconv/trial_1/checkpoints/checkpoint_56876.pt", initialize_lazy = True)
    model_spikeconv.to(device)

    param_size = 0
    for param in model_spikeconv.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model_spikeconv.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model spikeconv size: {:.3f}MB'.format(size_all_mb))

    model_spikelin= BonitoSpikeLin(
        load_default = True,
        device = device,
        dataloader_train = "", 
        dataloader_validation = "", 
        scaler = None,
        use_amp = False,
        nlstm=0,
        conv_threshold=0.01,
        slstm_threshold=0.05
    )

    param_size = 0
    for i,param in enumerate(model_spikelin.parameters()):
        #print(i,") ",param.shape )
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model_spikelin.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model spikelin size: {:.3f}MB'.format(size_all_mb))

    for train_batch_num, train_batch in enumerate(model_snn.dataloader_train):
        flops_bonito = FlopCountAnalysis(model_bonito, train_batch['x'].unsqueeze(1))
        print("bonito flops: ",flops_bonito.total())
        print("bonito flops by operator: ",flops_bonito.by_module_and_operator())
        
        flops_snn = FlopCountAnalysis(model_snn, train_batch['x'].unsqueeze(1))
        print("bonitosnn flops: ",flops_snn.total())
        print("bonitosnn flops by operator: ",flops_snn.by_module_and_operator())
        
        flops_spikeconv = FlopCountAnalysis(model_spikeconv, train_batch['x'].unsqueeze(1))
        print("bonitospikeconv flops: ",flops_spikeconv.total())
        print("bonitospikeconv flops by operator: ",flops_spikeconv.by_module_and_operator())

        flops_spikelin = FlopCountAnalysis(model_spikelin, train_batch['x'].unsqueeze(1))
        print("bonitospikelin flops: ",flops_spikelin.total())
        print("bonitospikelin flops by operator: ",flops_spikelin.by_module_and_operator())

        break
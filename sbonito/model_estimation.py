import torch
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from bonitosnn.model.snn_model import BonitoSNNModel, BonitoSpikeConv #as Model
from bonito.model import BonitoModel

if __name__=="__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_snn = BonitoSNNModel(
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
    for param in model_bonito.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model_bonito.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('bonito model size: {:.3f}MB'.format(size_all_mb))

    model_snn = BonitoSpikeConv(
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
    for param in model_snn.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model_snn.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model spikeconv size: {:.3f}MB'.format(size_all_mb))
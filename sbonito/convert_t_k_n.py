import sys
import os
#import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import torch
import torch.nn as nn
import numpy as np
#from bonitosnn.model import BonitoSNNModel as Model
from bonito.model import BonitoModel as Model

import onnx
from onnx2keras import onnx_to_keras

import nengo
import nengo_dl
import tensorflow as tf

if __name__ =="__main__":
    """
    script that tries to convert a pytorch trained model to keras and then to the
    spiking equivalent through nengo.
    """

    checkpoint_file="./sbonito/trained/bonito_trained/inter_2000/checkpoint.pt"

    #device = torch.device("cpu")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler=None
    use_amp=False
    nlstm=0
    
    model = Model(
        load_default = True,
        device = device,
        dataloader_train = None, 
        dataloader_validation = None, 
        scaler = scaler,
        use_amp = use_amp,
        #nlstm=nlstm
    )
    model = model.to(device)
    model.load(checkpoint_file, initialize_lazy = False)
    print(model)
    x = torch.randn( 16, 1, 2000, requires_grad=True).to(device)
    extractor=model.convolution

    extractor.eval()
    
    #x = torch.randn( 16, 1, 2000, requires_grad=True).to(device)
    #conv0=nn.Conv1d(1, 4, kernel_size=(5,), stride=(1,), padding=(2,)).to(device)
    """extractor=nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=(5,), stride=(1,), padding=(2,)),
            nn.SiLU(),
            nn.Conv1d(4, 16, kernel_size=(5,), stride=(1,), padding=(2,)),
            nn.SiLU(),
            nn.Conv1d(16, 384, kernel_size=(19,), stride=(5,), padding=(9,)),
    ).to(device)"""
    #torch_out = conv0(x) #check output shape

    torch.onnx.export(extractor, x, "./checkpoint_43546.onnx",
                    export_params=True, verbose=True, do_constant_folding=True,
     input_names = ['input'], output_names = ['output'])
    
    onnx_model = onnx.load('./checkpoint_43546.onnx')

    #print(onnx.checker.check_model(onnx_model))
    k_model = onnx_to_keras(onnx_model, ['input'], name_policy='renumerate')

    k_model.summary()

    converter = nengo_dl.Converter(k_model, allow_fallback=True, max_to_avg_pool=True, inference_only=True)
    
    """
    inp = tf.keras.Input(shape=( 1, 2000))

    conv0 = tf.keras.layers.Conv1D(
        filters=1000,
        kernel_size=5,
        strides=1,
        activation=tf.nn.silu,
    )(inp)

    conv1 = tf.keras.layers.Conv1D(
        filters=16,
        kernel_size=5,
        strides=1,
        activation=tf.nn.silu,
    )(conv0)

    conv2 = tf.keras.layers.Conv1D(
        filters=384,
        kernel_size=19,
        strides=5,
        activation=tf.nn.silu,
    )(conv1)

    k_model2 = tf.keras.Model(inputs=inp, outputs=conv2)

    converter = nengo_dl.Converter(k_model2, allow_fallback=True, max_to_avg_pool=True, inference_only=True)
    """


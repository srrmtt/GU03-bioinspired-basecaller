"""Implementation of the Bonito-CTC model

Based on: 
https://github.com/nanoporetech/bonito
"""

import os
import sys
from bonitosnn.layers.layers import BonitoLSTM
from torch import nn

import torch
import snntorch as snn
from snntorch import surrogate,utils

from bonitosnn.classes import BaseModelImpl
from bonitosnn.layers import BonitoSLSTM


class BonitoSNNModel(BaseModelImpl):
    """Bonito Model
    """
    def __init__(self, convolution = None, encoder = None, decoder = None, reverse = True,
                 load_default = False,nlstm=0,slstm_threshold=0.05, conv_threshold=0, *args, **kwargs):
        super(BonitoSNNModel, self).__init__(*args, **kwargs)
        """
        Args:
            convolution (nn.Module): module with: in [batch, channel, len]; out [batch, channel, len]
            encoder (nn.Module): module with: in [len, batch, channel]; out [len, batch, channel]
            decoder (nn.Module): module with: in [len, batch, channel]; out [len, batch, channel]
            reverse (bool): if the first rnn layer starts with reverse 
        """
    
        self.convolution = convolution
        self.encoder = encoder
        self.decoder = decoder
        self.reverse = reverse
        self.nlstm = nlstm

        #iperparametri
        self.slstm_threshold=slstm_threshold
        
        if load_default:
            self.load_default_configuration()

    def forward(self, x):
        """Forward pass of a batch
        
        Args:
            x (tensor) : [batch, channels (1), len]
        """
        #x.shape: [batch_size,1,2000]

        x = self.convolution(x)
        #x.shape: [batch_size,384,400]

        x = x.permute(2, 0, 1) # [len, batch, channels] [400,batch_size,384]
        
        x = self.encoder(x)

        x = self.decoder(x)

        return x

    def build_cnn(self):

        cnn = nn.Sequential(
            nn.Conv1d(
                in_channels = 1, 
                out_channels = 4, 
                kernel_size = 5, 
                stride= 1, 
                padding=5//2, 
                bias=True),
            nn.SiLU(),
            nn.Conv1d(
                in_channels = 4, 
                out_channels = 16, 
                kernel_size = 5, 
                stride= 1, 
                padding=5//2, 
                bias=True),
            nn.SiLU(),
            nn.Conv1d(
                in_channels = 16, 
                out_channels = 384, 
                kernel_size = 19, 
                stride= 5, 
                padding=19//2, 
                bias=True),
            nn.SiLU()
        )
        return cnn

    def build_encoder(self, input_size, reverse,nlstm=0):
        if reverse: #mi sa che a sto punto non servono proprio
            reverse=True
        else:
            reverse=False
        
        modules=[] #BonitoLSTM(input_size, 384, reverse = True)]
        
        for i in range(nlstm):
            reverse= not reverse
            modules.append(BonitoLSTM(input_size, 384, reverse = True))
        for j in range(5-nlstm):
            reverse= not reverse
            modules.append(BonitoSLSTM(384, 384, reverse = True,threshold=self.slstm_threshold))

        encoder=nn.Sequential(*modules)
        return encoder    

    def get_defaults(self):
        defaults = {
            'cnn_output_size': 384, 
            'cnn_output_activation': 'silu',
            'encoder_input_size': 384,
            'encoder_output_size': 384,
            'cnn_stride': 5,
        }
        return defaults
        
    def load_default_configuration(self):
        """Sets the default configuration for one or more
        modules of the network
        """

        self.convolution = self.build_cnn()
        self.cnn_stride = self.get_defaults()['cnn_stride']
        self.encoder = self.build_encoder(input_size = 384, reverse = True,nlstm=self.nlstm)
        self.decoder = self.build_decoder(encoder_output_size = 384, decoder_type = 'crf')
        self.decoder_type = 'crf'

class BonitoSpikeConv(BonitoSNNModel):
    def __init__(self, convolution=None, encoder=None, decoder=None, reverse=True, load_default=False,slstm_threshold=0.05,conv_threshold=0.05,nlstm=0, *args, **kwargs):
        super().__init__(convolution, encoder, decoder, reverse, load_default,nlstm,slstm_threshold, *args, **kwargs)
        self.convolution=self.build_spike_conv(conv_threshold) #build_spike_conv() #build_cnn()

    def build_spike_conv(self,conv_threshold):
        
        class SpikeConv(nn.Module):
            def __init__(self,conv_th=0.05):
                super(SpikeConv, self).__init__()
                beta = 0.8  # neuron decay rate  #GROUPS : A: [0.7], B: [0.8], C: [0.85], D: [0.9 - 1]
                grad = surrogate.straight_through_estimator()
                
                self.neuron1=snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True,threshold=conv_th, learn_beta=True)
                self.neuron2=snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True,threshold=conv_th,learn_beta=True)
                self.neuron3=snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True,threshold=conv_th,learn_beta=True)

                self.conv1=nn.Conv1d(in_channels = 1, out_channels = 4, kernel_size = 5, stride= 1, padding=5//2, bias=True)
                self.conv2=nn.Conv1d(in_channels = 4, out_channels = 16, kernel_size = 5, stride= 1, padding=5//2, bias=True)
                self.conv3=nn.Conv1d(in_channels = 16, out_channels = 384, kernel_size = 19, stride= 5, padding=19//2, bias=True)
                """
                self.lin1=nn.Linear(1,4)
                self.lin2=nn.Linear(4,16)
                self.lin3=nn.Linear(16,384)
                """
                #self.lin4=nn.Linear(2000,400)
                #self.net= nn.Sequential(nn.Linear(1,4),self.neuron1,nn.Linear(4,16),self.neuron2,nn.Linear(16,384),self.neuron3)
                
                self.cnet=nn.Sequential(
                                nn.Conv1d(in_channels = 1, out_channels = 4, kernel_size = 5, stride= 1, padding=5//2, bias=True),
                                self.neuron1,
                                nn.Conv1d(in_channels = 4, out_channels = 16, kernel_size = 5, stride= 1, padding=5//2, bias=True),
                                self.neuron2,
                                nn.Conv1d(in_channels = 16, out_channels = 384, kernel_size = 19, stride= 5, padding=19//2, bias=True),
                                self.neuron3 #nn.SiLU()
                            )

            def forward(self, x):
                """ #uncomment to estimate sops and energy consumption
                spike_probe=[]

                spike_recording = []
                mem_recording=[]
                for x_step in x: #.permute(2,0,1)
                    cur1=self.conv1(x_step)
                    spk1=self.neuron1(cur1) #,mem1) ,mem1
                    cur2=self.conv2(spk1)
                    spk2=self.neuron2(cur2)#,mem2) ,mem2
                    cur3=self.conv3(spk2)
                    spk3=self.neuron3(cur3)#,mem3) ,mem3

                    spike_probe.append(spk1.flatten())
                    spike_probe.append(spk2.flatten())
                    spike_probe.append(spk3.flatten())

                    spike_recording.append(spk3)
                    #mem_recording.append(mem3)
                
                tot_spikes=torch.sum(torch.cat(spike_probe))/4
                energy = tot_spikes*5.07e-10
                print("spikeconv sops: ",tot_spikes)
                print("spikeconv energy: ",energy*1e6)
                
                return torch.stack(spike_recording) #,torch.stack(mem_recording)
                """
                #utils.reset(self.net)
                utils.reset(self.cnet)
                self.cnet.train()
                """
                spike_recording = []
                for x_step in x.permute(2,0,1):  #layer lineari lavorano sull'ultima dimensione
                    spike_recording.append(self.net(x_step))

                #spike_recording = []
                #for x_step in x:
                #    spike_recording.append(self.cnet(x_step))
                #x=torch.stack(spike_recording)

                #x=self.net(x.permute(2,0,1))
                #x=self.conv1(x.permute(1,2,0))
                #x=self.lin4(x.permute(1,2,0))  #self.neuron4(
                """
                return self.cnet(x)
                
        return SpikeConv(conv_threshold)
    
class BonitoSpikeLin(BonitoSNNModel):
    def __init__(self, convolution=None, encoder=None, decoder=None, reverse=True, load_default=False,slstm_threshold=0.05,conv_threshold=0.05, nlstm=0, *args, **kwargs):
        super().__init__(convolution, encoder, decoder, reverse, load_default,nlstm,slstm_threshold, *args, **kwargs)
        self.convolution=self.build_spike_lin(conv_threshold) #build_spike_conv() #build_cnn()

    def build_spike_lin(self,conv_threshold):
        
        class SpikeLin(nn.Module):
            def __init__(self,conv_th=0.05):
                super(SpikeLin, self).__init__()
                beta = 0.8  # neuron decay rate  #GROUPS : A: [0.7], B: [0.8], C: [0.85], D: [0.9 - 1]
                grad = surrogate.straight_through_estimator()
                
                self.neuron1=snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True,threshold=conv_th)
                self.neuron2=snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True,threshold=conv_th)
                self.neuron3=snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True,threshold=conv_th)

                self.neuron4=snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True,threshold=conv_th)

                
                self.lin1=nn.Linear(1,4)
                self.lin2=nn.Linear(4,16)
                self.lin3=nn.Linear(16,384)
                
                self.lin4=nn.Linear(2000,400)
                #self.net= nn.Sequential(nn.Linear(1,4),self.neuron1,nn.Linear(4,16),self.neuron2,nn.Linear(16,384),self.neuron3)
                
                self.linet=nn.Sequential(
                                #nn.Conv1d(in_channels = 1, out_channels = 4, kernel_size = 5, stride= 1, padding=5//2, bias=True),
                                self.lin1,
                                self.neuron1,
                                #nn.Conv1d(in_channels = 4, out_channels = 16, kernel_size = 5, stride= 1, padding=5//2, bias=True),
                                self.lin2,
                                self.neuron2,
                                #nn.Conv1d(in_channels = 16, out_channels = 384, kernel_size = 19, stride= 5, padding=19//2, bias=True),
                                self.lin3,
                                self.neuron3,

                            )

            def forward(self, x):
                #mem1 = self.neuron1.init_leaky() #non necessario se init_hidden True
                
                utils.reset(self.linet)

                self.linet.train()
                """ #uncomment for sops and energy estimation
                lin1=self.lin1(x.permute(2,0,1))
                spk1=self.neuron1(lin1)
                lin2=self.lin2(spk1)
                spk2=self.neuron2(lin2)
                lin3=self.lin3(spk2)
                spk3=self.neuron3(lin3)
                lin4=self.lin4(spk3.permute(1,2,0))
                spk4=self.neuron4(lin4)

                tot_spikes=torch.sum(torch.cat([spk.flatten() for spk in [spk1,spk2,spk3,spk4]]))/4
                energy = tot_spikes*5.07e-10
                print("spikelin sops: ",tot_spikes)
                print("spikelin energy: ",energy*1e6)
                """
                x1=self.linet(x.permute(2,0,1))
                y=self.neuron4(self.lin4(x1.permute(1,2,0)))

                """
                spike_recording = []
                for x_step in x.permute(2,0,1):  #layer lineari lavorano sull'ultima dimensione
                    spike_recording.append(self.net(x_step))

                #spike_recording = []
                #for x_step in x:
                #    spike_recording.append(self.cnet(x_step))
                #x=torch.stack(spike_recording)

                #x=self.net(x.permute(2,0,1))
                #x=self.conv1(x.permute(1,2,0))
                #x=self.lin4(x.permute(1,2,0))  #self.neuron4(
                """
                return y
                
        return SpikeLin(conv_threshold)
    
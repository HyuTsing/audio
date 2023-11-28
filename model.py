import torch
from torchvision.models.resnet import BasicBlock
import conv
import torch.nn as nn
from torch.nn.utils import weight_norm
import LSSC
import dconv
import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, base, neuron, surrogate
from spikingjelly.activation_based.model import spiking_resnet
from sklearn import preprocessing
import numpy as np


class incoder_decoder(nn.Module):
    def __init__(self):
        super(incoder_decoder, self).__init__()
        #self.con1 = nn.Sequential(
            #tcn.TemporalConvNet())
        #self.con2=torch.nn.Conv2d(1,16,5,1,2)
        self.conv=nn.Sequential(
            LSSC.TemporalConvNet(kernel_size=5)
        )
        """"""
        self.flat=nn.Flatten()
        # self.lin1=nn.Linear(36*12*41,12*41)
        self.lin1 = nn.Linear(5120, 800)
        self.sn1 = neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.lin2 = nn.Linear(800, 12 * 41)
        self.sn2 = neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.lin3 = nn.Linear(12 * 41, 35)
        self.sn3 = neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        # self.vote=VotingLayer(10)
        self.fc = nn.Sequential(self.flat, self.lin1, self.sn1, self.lin2, self.sn2, self.lin3, self.sn3)

    #tcn.TemporalConvNet(num_inputs, num_channels))

    def forward(self, x: torch.Tensor):
        a=1.1

        # x = x.permute(2, 0, 1 ,3,4)# [N, T, 2, H, W] -> [T, N, 2, H, W]
        # input = x[2]

        out1=self.conv(x)

        #out2=self.resnet(out1)
        # print(out1.size())
        out3=self.fc(out1)
        #print(out.shape)
        #print(out3.shape)
        out_spikes = out3
        input=x
        for t in range(1, 8):
            out1=self.conv(input)

            out_spikes += self.fc(out1)
        return out_spikes / 8
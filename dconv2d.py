import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, base, neuron, surrogate
class ConvBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_inputs, n_outputs, kernel_size,
                          stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(n_outputs)
        self.net = nn.Sequential(self.conv1, self.bn1)
        self.sn1 = neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.maxpool = nn.MaxPool2d(stride=stride, kernel_size=kernel_size, padding=padding,
                                    dilation=dilation)
        self.conv2=nn.Conv2d(n_inputs, n_outputs, kernel_size=1)
        self.net2=nn.Sequential(self.maxpool,self.conv2)
    def forward(self, x):
        out = self.net(x)
        res = self.net2(x)
        out = out + res
        out = self.sn1(out)
        return out


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        layers = []

        layers += [ConvBlock(1, 32, kernel_size=5, stride=3, dilation=2,
                                 padding=0)]
        layers += [ConvBlock(32, 64, kernel_size=3, stride=2, dilation=1,
                                 padding=0)]
        layers += [ConvBlock(64, 32, kernel_size=2, stride=1, dilation=1,
                             padding=0)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class BackEndNet(nn.Module):
    def __init__(self):
        super(BackEndNet, self).__init__()
        self.convnet=ConvNet()
        self.flat = nn.Flatten()
        # self.lin1=nn.Linear(36*12*41,12*41)
        self.lin1 = nn.Linear(3840, 1024)
        self.sn1 = neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.lin2 = nn.Linear(1024, 12 * 41)
        self.sn2 = neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.lin3 = nn.Linear(12 * 41, 35)
        self.sn3 = neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.fc = nn.Sequential(self.flat, self.lin1, self.sn1, self.lin2, self.sn2, self.lin3, self.sn3)


    def forward(self, x: torch.Tensor):
        a=1.1
        # x = x.permute(2, 0, 1 ,3,4)# [N, T, 2, H, W] -> [T, N, 2, H, W]
        # input = x[2]
        out1=self.convnet(x)
        out2=self.fc(out1)
        #print(out.shape)
        #print(out3.shape)
        out_spikes = out2
        input=x
        for t in range(1, 8):
            out1=self.convnet(input)

            out_spikes += self.fc(out1)
        return out_spikes / 8

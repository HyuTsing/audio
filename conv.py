import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch
import torch.nn as nn
from copy import deepcopy
import data_deal
from spikingjelly.activation_based import layer, base, neuron, surrogate


class BatchNorm1d(nn.BatchNorm1d, base.StepModule):
    def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            step_mode='s'
    ):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f', step_mode={self.step_mode}'

    def forward(self, x):
        if self.step_mode == 's':
            return super().forward(x)


def conv(in_planes: int, out_planes: int, kernel_size: int, stride: int, padding: int, dilation: int):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, groups=1, bias=False, dilation=dilation)



class DifferConv(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(DifferConv, self).__init__()
        norm_layer = nn.BatchNorm1d
        dropout=nn.Dropout(0.25)
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                      stride=stride, padding=padding, dilation=dilation)
        self.bn1 = norm_layer(n_outputs)
        self.sn1 = neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.net = nn.Sequential(self.conv1, self.bn1)
        self.net2 = nn.Sequential(nn.Conv1d(n_inputs, n_outputs, int(kernel_size+2),
                                             stride=stride, padding=int((2*kernel_size+3)/ 2),
                                             dilation=2),
                                        norm_layer(n_outputs))
        self.net3 = nn.Sequential(nn.Conv1d(n_inputs, n_outputs, int(kernel_size + 4),
                                       stride=stride, padding=int((5*kernel_size+16 ) / 2),
                                       dilation=5),
                                  norm_layer(n_outputs))
        # neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True),
        # nn.Dropout(dropout))
    def forward(self, x):
        out = self.net(x)

        # print(out.shape)
        # print(out)
        # out1 = self.net2(x)
        # out2 = self.net3(x)
        #
        # out=torch.cat((out,out1,out2),1)
        #print(out.shape)
        out = self.sn1(out)

        return out


class TemporalConvNet(nn.Module):
    def __init__(self, kernel_size, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        layers += [DifferConv(1, 4, kernel_size=7, stride=2, dilation=8,
                                 padding=3, dropout=dropout)]
        layers += [DifferConv(12, 8, kernel_size=5, stride=2, dilation=4,
                              padding=2, dropout=dropout)]
        layers += [DifferConv(24, 12, kernel_size=5, stride=1, dilation=2,
                              padding=2, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
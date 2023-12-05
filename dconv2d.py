import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, base, neuron, surrogate


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x.size() 30,40,50,30
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 30,1,50,30
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 30,1,50,30
        return self.sigmoid(x)  # 30,1,50,30


class FeedBackNet(nn.Module):
    def __init__(self):
        super(FeedBackNet, self).__init__()
        self.bn = nn.BatchNorm2d(16)
        self.preconv = nn.Conv2d(1, 16, 1, 1, 0)
        self.conv1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv1a = nn.Conv2d(16, 16, 5, 1, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv2a = nn.Conv2d(16, 32, 5, 1, 2)
        self.conv3 = nn.Conv2d(32, 16, 5, 1, 2)
        self.conv3a = nn.Conv2d(32, 16, 7, 1, 3)
        self.sa = SpatialAttention()
        self.conv4 = nn.Conv2d(16, 8, 3, 2, 0)
        self.conv5 = nn.Conv2d(8, 4, 3, 2, 0)
        self.back = nn.Sequential(self.conv4, self.conv5)

    def forward(self, x):
        x = self.preconv(x)
        input = x
        for i in range(4):
            a1 = self.conv1(input)
            a2 = self.conv1a(input)
            a3 = a1 + a2
            b1 = self.conv2(a3)
            b2 = self.conv2a(a3)
            b3 = b1 + b2
            c1 = self.conv3(b3)
            c2 = self.conv3a(b3)
            c3 = c1 + c2
            c3 = self.bn(c3)
            input = x + c3
        sa = self.sa(c3)
        output = sa * c3
        output=self.back(output)
        return output


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
        self.conv2 = nn.Conv2d(n_inputs, n_outputs, kernel_size=1)
        self.net2 = nn.Sequential(self.maxpool, self.conv2)

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
        self.fbn = FeedBackNet()
        self.convnet = ConvNet()
        self.flat = nn.Flatten()
        # self.lin1=nn.Linear(36*12*41,12*41)
        self.lin1 = nn.Linear(1560, 1024)
        self.sn1 = neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.lin2 = nn.Linear(1024, 12 * 41)
        self.sn2 = neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.lin3 = nn.Linear(12 * 41, 35)
        self.sn3 = neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.fc = nn.Sequential(self.flat, self.lin1, self.sn1, self.lin2, self.sn2, self.lin3, self.sn3)

    def forward(self, x: torch.Tensor):
        a = 1.1
        # x = x.permute(2, 0, 1 ,3,4)# [N, T, 2, H, W] -> [T, N, 2, H, W]
        # input = x[2]
        out1 = self.fbn(x)
        out2 = self.fc(out1)
        # print(out.shape)
        # print(out3.shape)
        out_spikes = out2
        input = x
        for t in range(1, 8):
            out1 = self.fbn(input)

            out_spikes += self.fc(out1)
        return out_spikes / 8

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch
import torch.nn as nn
from copy import deepcopy
import data_deal
from spikingjelly.activation_based import layer, base, neuron, surrogate


class ResBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, avg_pan=0, dropout=0.2):
        super(ResBlock, self).__init__()
        norm_layer = BatchNorm2d
        self.conv1 = conv(n_inputs, n_outputs, kernel_size,
                          stride=stride, padding=(kernel_size - 1) // 2, dilation=1)
        self.bn1 = norm_layer(n_outputs)
        self.se = SE_Block(inchannel=n_outputs)
        self.spatial_attention = SpatialAttention()
        self.net = nn.Sequential(self.conv1, self.bn1)
        self.sn1 = neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.downsample = nn.Sequential(conv(n_inputs, n_outputs, kernel_size=1,
                                             stride=stride, padding=0, dilation=1),
                                        norm_layer(n_outputs))

    def forward(self, x):
        out = self.net(x)
        sa_out = self.spatial_attention(out)
        res = self.downsample(x)
        out = out * sa_out
        out = out + res
        out = self.sn1(out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x.size() 30,40,50,30
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 30,1,50,30
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 30,1,50,30
        return self.sigmoid(x)  # 30,1,50,30


class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool1d(1)
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        # 读取批数据图片数量及通道数
        b, c, w = x.size()
        # Fsq操作：经池化后输出b*c的矩阵
        y = self.gap(x).view(b, c)
        # Fex操作：经全连接层输出（b，c，1，1）矩阵
        y = self.fc(y).view(b, c, 1)
        # Fscale操作：将得到的权重乘以原来的特征图x
        return x * y.expand_as(x)


class BatchNorm2d(nn.BatchNorm2d, base.StepModule):
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
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, groups=1, bias=False, dilation=dilation)


def conv1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, avg_pan, dropout=0.2):
        super(TemporalBlock, self).__init__()
        norm_layer = BatchNorm2d
        self.conv1 = conv(n_inputs, n_outputs, kernel_size,
                          stride=stride, padding=padding, dilation=dilation)
        self.bn1 = norm_layer(n_outputs)
        self.se = SE_Block(inchannel=n_outputs)
        self.spatial_attention = SpatialAttention()
        self.net = nn.Sequential(self.conv1, self.bn1)
        self.sn1 = neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.maxpool = nn.MaxPool2d(n_inputs, n_outputs, kernel_size, padding=padding,
                                    dilation=dilation)
        self.downsample = nn.Sequential(conv(n_inputs, n_outputs, int(kernel_size * (dilation / 4)),
                                             stride=stride, padding=int(((kernel_size * (dilation / 4)) - 1) / 2),
                                             dilation=1),
                                        norm_layer(n_outputs))
        # neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True),
        # nn.Dropout(dropout))

    def forward(self, x):
        out = self.net(x)
        # print(out.shape)
        # print(out)
        # res = self.downsample(x)
        res = self.maxpool(x)
        # res=self.ds(res)
        # print(res)
        # print(res.shape)
        # se_out = self.se(out)
        # sa_out = self.spatial_attention(out)
        # out = out * sa_out
        out = out + res
        out = self.sn1(out)
        # print(out)
        return out


class TemporalBlock1(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, avg_pan, dropout=0.2):
        super(TemporalBlock1, self).__init__()
        norm_layer = BatchNorm2d
        self.conv1 = conv(n_inputs, n_outputs, kernel_size,
                          stride=stride, padding=padding, dilation=dilation)
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.bn1 = norm_layer(n_outputs)
        self.sn1 = neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.se = SE_Block(inchannel=n_outputs)
        self.spatial_attention = SpatialAttention()
        self.net = nn.Sequential(self.conv1, self.bn1)
        self.downsample = nn.Sequential(conv(n_inputs, n_outputs, 41,
                                             stride=stride, padding=20, dilation=1),
                                        norm_layer(n_outputs)
                                        )
        self.maxpool = nn.MaxPool2d(n_inputs, n_outputs,kernel_size=kernel_size,  padding=padding, dilation=dilation)
        self.ds = nn.Conv1d(in_channels=n_inputs, out_channels=n_outputs, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.net(x)
        # print(out.shape)
        # print(out)
        res = self.maxpool(x)
        # print(res)
        # print(res.shape)
        # se_out = self.se(out)
        # sa_out = self.spatial_attention(out)
        # out = out * sa_out
        out = out + res
        out = self.sn1(out)
        # print(out)
        return out


class TemporalBlock2(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, avg_pan, dropout=0.2):
        super(TemporalBlock2, self).__init__()
        norm_layer = BatchNorm2d
        self.conv1 = conv(n_inputs, n_outputs, kernel_size,
                          stride=stride, padding=padding, dilation=dilation)
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.bn1 = norm_layer(n_outputs)
        self.sn1 = neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.se = SE_Block(inchannel=n_outputs)
        self.spatial_attention = SpatialAttention()
        self.net = nn.Sequential(self.conv1, self.bn1)
        self.downsample = nn.Sequential(conv(n_inputs, n_outputs, kernel_size=kernel_size,
                                             stride=stride, padding=1, dilation=1),
                                        norm_layer(n_outputs)
                                        )
        self.maxpool = nn.MaxPool2d(n_inputs, n_outputs, kernel_size=kernel_size,  padding=padding,
                                    dilation=dilation)
        self.ds = nn.Conv1d(in_channels=n_inputs, out_channels=n_outputs, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.net(x)
        # print(out.shape)
        # print(out)
        res = self.maxpool(x)
        # print(res)
        # print(res.shape)
        se_out = self.se(out)
        sa_out = self.spatial_attention(out)
        # out = out * sa_out
        out = out + res
        out = self.sn1(out)
        # print(out)
        return out


class TemporalBlock3(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, avg_pan, dropout=0.2):
        super(TemporalBlock3, self).__init__()
        norm_layer = BatchNorm2d
        self.conv1 = conv(n_inputs, n_outputs, kernel_size,
                          stride=stride, padding=padding, dilation=dilation)
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.bn1 = norm_layer(n_outputs)
        self.sn1 = neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.se = SE_Block(inchannel=n_outputs)
        self.spatial_attention = SpatialAttention()
        self.net = nn.Sequential(self.conv1, self.bn1)
        self.downsample = nn.Sequential(conv(n_inputs, n_outputs, kernel_size=kernel_size,
                                             stride=stride, padding=1, dilation=1),
                                        norm_layer(n_outputs)
                                        )
        self.pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.ds = nn.Conv1d(in_channels=n_inputs, out_channels=n_outputs, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.net(x)
        # print(out.shape)
        # print(out)
        res = self.downsample(x)
        # print(res)
        # print(res.shape)
        # se_out = self.se(out)
        # sa_out = self.spatial_attention(out)
        # out = out * sa_out
        out = out + res
        out = self.sn1(out)
        # print(out)
        return out


class TemporalBlock4(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, avg_pan, dropout=0.2):
        super(TemporalBlock4, self).__init__()
        norm_layer = BatchNorm2d
        self.conv1 = conv(n_inputs, n_outputs, kernel_size,
                          stride=stride, padding=padding, dilation=dilation)
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.bn1 = norm_layer(n_outputs)
        self.sn1 = neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.se = SE_Block(inchannel=n_outputs)
        self.spatial_attention = SpatialAttention()
        self.net = nn.Sequential(self.conv1, self.bn1)
        self.downsample = nn.Sequential(conv(n_inputs, n_outputs, kernel_size=kernel_size,
                                             stride=stride, padding=1, dilation=1),
                                        norm_layer(n_outputs)
                                        )
        self.pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.ds = nn.Conv1d(in_channels=n_inputs, out_channels=n_outputs, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.net(x)
        # print(out.shape)
        # print(out)
        res = self.downsample(x)
        # print(res)
        # print(res.shape)
        # se_out = self.se(out)
        # sa_out = self.spatial_attention(out)
        # out = out * sa_out
        out = out + res
        out = self.sn1(out)
        # print(out)
        return out


class TemporalConvNet(nn.Module):
    def __init__(self, kernel_size, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        # layers+=[ResBlock(n_inputs=1,n_outputs=16,kernel_size=9,stride=1)]
        # layers+=[ResBlock(n_inputs=16,n_outputs=24,kernel_size=7,stride=1)]
        # layers+=[ResBlock(n_inputs=24,n_outputs=32,kernel_size=5,stride=1)]
        # layers+=[ResBlock(n_inputs=32,n_outputs=32,kernel_size=3,stride=1)]
        layers += [TemporalBlock(1, 32, kernel_size=80, stride=10, dilation=2,
                                 padding=int((80 - 1) / 2 * 2), avg_pan=int((40 - 1) / 2 * 1), dropout=dropout)]
        layers += [TemporalBlock(32, 64, kernel_size=20, stride=5, dilation=4,
                                 padding=int((20 - 1) / 2 * 4), avg_pan=9, dropout=dropout)]
        layers += [TemporalBlock(64, 128, kernel_size=11, stride=2, dilation=8,
                                 padding=int((11 - 1) / 2 * 8), avg_pan=5, dropout=dropout)]
        layers += [TemporalBlock1(128, 128, kernel_size=5, stride=1, dilation=16,
                                  padding=int((5 - 1) / 2 * 16), avg_pan=2, dropout=dropout)]

        # layers += [TemporalBlock1(20, 24, kernel_size=5, stride=1, dilation=16,
        #                           padding=int((5 - 1) / 2 * 16), avg_pan=2, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

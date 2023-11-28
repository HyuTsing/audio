class MultiDimensionalAttention(nn.Module, base.MultiStepModule):
    def __init__(self, T: int, C: int, reduction_t: int = 16, reduction_c: int = 16, kernel_size=3):
        """
        * :ref:`API in English <MultiStepMultiDimensionalAttention.__init__-en>`

        .. _MultiStepMultiDimensionalAttention.__init__-cn:

        :param T: 输入数据的时间步长

        :param C: 输入数据的通道数

        :param reduction_t: 时间压缩比

        :param reduction_c: 通道压缩比

        :param kernel_size: 空间注意力机制的卷积核大小

        `Attention Spiking Neural Networks <https://ieeexplore.ieee.org/document/10032591>`_ 中提出
        的MA-SNN模型以及MultiStepMultiDimensionalAttention层。

        您可以从以下链接中找到MA-SNN的示例项目:
        - https://github.com/MA-SNN/MA-SNN
        - https://github.com/ridgerchu/SNN_Attention_VGG

        输入的尺寸是 ``[T, N, C, H, W]`` ，经过MultiStepMultiDimensionalAttention层，输出为 ``[T, N, C, H, W]`` 。

        * :ref:`中文API <MultiStepMultiDimensionalAttention.__init__-cn>`

        .. _MultiStepMultiDimensionalAttention.__init__-en:

        :param T: timewindows of input

        :param C: channel number of input

        :param reduction_t: temporal reduction ratio

        :param reduction_c: channel reduction ratio

        :param kernel_size: convolution kernel size of SpatialAttention

        The MA-SNN model and MultiStepMultiDimensionalAttention layer are proposed in ``Attention Spiking Neural Networks <https://ieeexplore.ieee.org/document/10032591>`_.

        You can find the example projects of MA-SNN in the following links:
        - https://github.com/MA-SNN/MA-SNN
        - https://github.com/ridgerchu/SNN_Attention_VGG

        The dimension of the input is ``[T, N, C, H, W]`` , after the MultiStepMultiDimensionalAttention layer, the output dimension is ``[T, N, C, H, W]`` .


        """
        super().__init__()

        assert T >= reduction_t, 'reduction_t cannot be greater than T'
        assert C >= reduction_c, 'reduction_c cannot be greater than C'

        from einops import rearrange

        # Attention
        class TimeAttention(nn.Module):
            def __init__(self, in_planes, ratio=16):
                super(TimeAttention, self).__init__()
                self.avg_pool = nn.AdaptiveAvgPool3d(1)
                self.max_pool = nn.AdaptiveMaxPool3d(1)
                self.sharedMLP = nn.Sequential(
                    nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
                    nn.ReLU(),
                    nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False),
                )
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                avgout = self.sharedMLP(self.avg_pool(x))
                maxout = self.sharedMLP(self.max_pool(x))
                return self.sigmoid(avgout + maxout)

        class ChannelAttention(nn.Module):
            def __init__(self, in_planes, ratio=16):
                super(ChannelAttention, self).__init__()
                self.avg_pool = nn.AdaptiveAvgPool3d(1)
                self.max_pool = nn.AdaptiveMaxPool3d(1)
                self.sharedMLP = nn.Sequential(
                    nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
                    nn.ReLU(),
                    nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False),
                )
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = rearrange(x, "b f c h w -> b c f h w")
                avgout = self.sharedMLP(self.avg_pool(x))
                maxout = self.sharedMLP(self.max_pool(x))
                out = self.sigmoid(avgout + maxout)
                out = rearrange(out, "b c f h w -> b f c h w")
                return out

        class SpatialAttention(nn.Module):
            def __init__(self, kernel_size=3):
                super(SpatialAttention, self).__init__()
                assert kernel_size in (3, 7), "kernel size must be 3 or 7"
                padding = 3 if kernel_size == 7 else 1
                self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = rearrange(x, "b f c h w -> b (f c) h w")
                avgout = torch.mean(x, dim=1, keepdim=True)
                maxout, _ = torch.max(x, dim=1, keepdim=True)
                x = torch.cat([avgout, maxout], dim=1)
                x = self.conv(x)
                x = x.unsqueeze(1)
                return self.sigmoid(x)

        self.ta = TimeAttention(T, reduction_t)
        self.ca = ChannelAttention(C, reduction_c)
        self.sa = SpatialAttention(kernel_size)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        assert x.dim() == 5, ValueError(
            f'expected 5D input with shape [T, N, C, H, W], but got input with shape {x.shape}')
        x = x.transpose(0, 1)
        out = self.ta(x) * x
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = self.relu(out)
        out = out.transpose(0, 1)
        return out

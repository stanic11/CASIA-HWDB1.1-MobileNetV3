from typing import Callable, List, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from functools import partial

def _make_divisible(ch, divisor=8, min_ch=None):
    """
    对通道数进行向上取整，使其可被 divisor 整除。
    源自 TensorFlow 官方实现，确保卷积层的通道数为 divisor 的整数倍，
    并保证舍入后不会低于原值的 90%。
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

class ConvBNActivation(nn.Sequential):
    """
    组合 Conv2d + BatchNorm2d + 激活函数 为常用的卷积模块。
    支持可选的归一化层和激活层。
    """
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        # 计算 padding 使得输入输出尺寸保持一致（same padding）
        padding = (kernel_size - 1) // 2
        # 默认归一化层为 BatchNorm2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # 默认激活层为 ReLU6
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNActivation, self).__init__(
            nn.Conv2d(
                in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False  # 与 BatchNorm 一起使用时通常关闭 bias
            ),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation 模块，用于通道注意力机制。
    """
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super(SEModule, self).__init__()
        # 计算 squeeze 后的通道数，并保证能被 8 整除
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        # 用 1x1 卷积模拟全连接层
        self.fc1 = nn.Conv2d(input_c, squeeze_c, kernel_size=1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        # 全局平均池化到 1x1
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        # 使用 Hardsigmoid 激活获得缩放系数
        scale = F.hardsigmoid(scale, inplace=True)
        # 通道加权
        return scale * x


class InvertedResidualConfig:
    """
    保存 InvertedResidual 模块的配置，包括扩展通道数、卷积核大小、
    是否使用 SE，激活类型等。
    """
    def __init__(self,
                 input_c: int,
                 kernel: int,
                 expanded_c: int,
                 out_c: int,
                 use_se: bool,
                 activation: str,
                 stride: int,
                 width_multi: float):
        # 应用宽度乘子调整通道数并保证可被 8 整除
        self.input_c = self.adjust_channels(input_c, width_multi)
        self.kernel = kernel
        self.expanded_c = self.adjust_channels(expanded_c, width_multi)
        self.out_c = self.adjust_channels(out_c, width_multi)
        self.use_se = use_se
        # 根据 activation 字符串判断使用 H-Swish 还是 ReLU
        self.use_hs = activation == "HS"
        self.stride = stride

    @staticmethod
    def adjust_channels(channels: int, width_multi: float) -> int:
        """
        根据 width_multi 对通道数进行缩放并使用 _make_divisible 保证可整除。
        """
        return _make_divisible(int(channels * width_multi), 8)


class InvertedResidual(nn.Module):
    """
    MobileNetV3 的基本单元:Inverted Residual 块，
    包含 expand、depthwise、SE、project 步骤，以及残差连接（可选）。
    """
    def __init__(self,
                 config: InvertedResidualConfig,
                 norm_layer: Callable[..., nn.Module]):
        super(InvertedResidual, self).__init__()

        # 当 stride=1 且输入输出通道相等时，使用残差连接
        self.use_res_connect = (config.stride == 1 and config.input_c == config.out_c)

        layers: List[nn.Module] = []
        # 选择激活函数，ReLU 或 H-Swish
        activation_layer = nn.Hardswish if config.use_hs else nn.ReLU

        # 1) 扩展层：1x1 卷积增加通道
        if config.expanded_c != config.input_c:
            layers.append(
                ConvBNActivation(
                    in_planes=config.input_c,
                    out_planes=config.expanded_c,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer
                )
            )

        # 2) Depthwise 卷积
        layers.append(
            ConvBNActivation(
                in_planes=config.expanded_c,
                out_planes=config.expanded_c,
                kernel_size=config.kernel,
                stride=config.stride,
                groups=config.expanded_c,  # groups=输入通道，实现 depthwise
                norm_layer=norm_layer,
                activation_layer=activation_layer
            )
        )

        # 3) 可选 SE 注意力模块
        if config.use_se:
            layers.append(SEModule(config.expanded_c))

        # 4) 投影层：1x1 卷积降低通道
        layers.append(
            ConvBNActivation(
                in_planes=config.expanded_c,
                out_planes=config.out_c,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Identity  # 投影层不使用激活函数
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = config.out_c
        self.is_strided = config.stride > 1

    def forward(self, x: Tensor) -> Tensor:
        """
        前向计算。计算块输出，如果满足条件则加上残差连接。
        """
        result = self.block(x)
        if self.use_res_connect:
            result += x
        return result


class MobileNetV3(nn.Module):
    """
    MobileNetV3 主干网络，实现从输入到分类器的完整结构。
    支持灰度图(in_channels=1)和 RGB 图(in_channels=3)。
    """
    def __init__(self,
                 num_classes: int = 1000,
                 in_channels: int = 1,  # 支持灰度或彩色输入
                 reduced_tail: bool = False,
                 width_multi: float = 1.0,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(MobileNetV3, self).__init__()

        # 默认基本块为 InvertedResidual
        if block is None:
            block = InvertedResidual
        # 默认归一化层为适配 MobileNetV3 的 BatchNorm
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        # 便捷方法：配置生成和通道缩放
        bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
        adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)
        reduce_divider = 2 if reduced_tail else 1

        layers: List[nn.Module] = []

        # 1) 第一层卷积，将输入通道(in_channels)映射到首个 bottleneck 的输入通道
        first_conf = bneck_conf(16, 3, 16, 16, False, "RE", 1)
        layers.append(
            ConvBNActivation(
                in_planes=in_channels,
                out_planes=first_conf.input_c,
                kernel_size=3,
                stride=1,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish
            )
        )

        # 2) 一系列 Inverted Residual 块，配置内联定义
        layers.extend([
            block(bneck_conf(16, 3, 16, 16, False, "RE", 1), norm_layer),
            block(bneck_conf(16, 3, 64, 24, False, "RE", 2), norm_layer),
            block(bneck_conf(24, 3, 72, 24, False, "RE", 1), norm_layer),
            block(bneck_conf(24, 5, 72, 40, True, "RE", 2), norm_layer),
            block(bneck_conf(40, 5, 120, 40, True, "RE", 1), norm_layer),
            block(bneck_conf(40, 5, 120, 40, True, "RE", 1), norm_layer),
            block(bneck_conf(40, 3, 240, 80, False, "HS", 2), norm_layer),
            block(bneck_conf(80, 3, 200, 80, False, "HS", 1), norm_layer),
            block(bneck_conf(80, 3, 184, 80, False, "HS", 1), norm_layer),
            block(bneck_conf(80, 3, 184, 80, False, "HS", 1), norm_layer),
            block(bneck_conf(80, 3, 480, 112, True, "HS", 1), norm_layer),
            block(bneck_conf(112, 3, 672, 112, True, "HS", 1), norm_layer),
            block(bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2), norm_layer),
            block(bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1), norm_layer)
        ])
        last_conf = bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1)

        # 3) 最后一层 1x1 卷积，扩展通道数到 6 倍
        layers.append(
            ConvBNActivation(
                in_planes=last_conf.out_c,
                out_planes=last_conf.out_c * 6,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish
            )
        )

        # 4) 将所有模块组合成特征提取器
        self.features = nn.Sequential(*layers)
        # 5) 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 6) 分类头：两层全连接 + H-Swish + Dropout
        last_channel = adjust_channels(1280 // reduce_divider)
        lastconv_output_c = last_conf.out_c * 6
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_c, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(last_channel, num_classes)
        )

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        """
        内部前向实现，分别经过特征提取、池化、平坦化和分类头。
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        标准前向方法，调用内部实现。
        """
        return self._forward_impl(x)
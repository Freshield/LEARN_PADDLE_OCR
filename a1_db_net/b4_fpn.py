# coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: b4_fpn.py
@Time: 2022-12-04 22:07
@Last_update: 2022-12-04 22:07
@Desc: None
@==============================================@
@      _____             _   _     _   _       @
@     |   __|___ ___ ___| |_|_|___| |_| |      @
@     |   __|  _| -_|_ -|   | | -_| | . |      @
@     |__|  |_| |___|___|_|_|_|___|_|___|      @
@                                    Freshield @
@==============================================@
"""
import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle import ParamAttr


class DBFPN(nn.Layer):
    def __init__(self, in_channels, out_channels, use_asf=False, **kwargs):
        """fpn层初始化"""
        super(DBFPN, self).__init__()
        self.out_channels = out_channels
        self.use_asf = use_asf
        weight_attr = paddle.nn.initializer.KaimingUniform()

        self.in2_conv = nn.Conv2D(
            in_channels=in_channels[0],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.in3_conv = nn.Conv2D(
            in_channels=in_channels[1],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.in4_conv = nn.Conv2D(
            in_channels=in_channels[2],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.in5_conv = nn.Conv2D(
            in_channels=in_channels[3],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.p5_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.p4_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.p3_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.p2_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)

    def forward(self, x):
        """前向过程，这里x为backbone的输出，要分别进行上采样"""
        # 获取backbone的输出
        c2, c3, c4, c5 = x
        # 对所有的输入进行一次卷积，用的1x1的卷积，大小一样，
        # 主要是增加channel为out channel数
        # 这里为256
        in5 = self.in5_conv(c5)
        in4 = self.in4_conv(c4)
        in3 = self.in3_conv(c3)
        in2 = self.in2_conv(c2)
        # 分别进行特征上采样
        # 1/16大小，256，40，40
        out4 = in4 + F.upsample(
            in5, scale_factor=2, mode='nearest', align_mode=1)
        # 1/8大小，256，80，80
        out3 = in3 + F.upsample(
            out4, scale_factor=2, mode='nearest', align_mode=1)
        # 1/4大小，256，160，160
        out2 = in2 + F.upsample(
            out3, scale_factor=2, mode='nearest', align_mode=1)
        # 这里在通过一次3x3的卷积，把channel降为out channel的1/4，都是64
        # 大小顺序为20，40，80，160
        p5 = self.p5_conv(in5)
        p4 = self.p4_conv(out4)
        p3 = self.p3_conv(out3)
        p2 = self.p2_conv(out2)
        # 然后上采样为原图1/4大小现在都是64，160，160
        p5 = F.upsample(p5, scale_factor=8, mode='nearest', align_mode=1)
        p4 = F.upsample(p4, scale_factor=4, mode='nearest', align_mode=1)
        p3 = F.upsample(p3, scale_factor=2, mode='nearest', align_mode=1)
        # 然后把所有的concat起来
        fuse = paddle.concat([p5, p4, p3, p2], axis=1)

        return fuse


if __name__ == '__main__':
    from PaddleOCR.ppocr.modeling.backbones.det_mobilenet_v3 import MobileNetV3

    fake_inputs = paddle.randn([1, 3, 640, 640], dtype='float32')
    model_backbone = MobileNetV3()
    # 输出为所需要的中间输出分别尺度为：
    # 输入为1，3，640，640的情况下
    # 1，16，160，160
    # 1，24，80，80
    # 1，56，40，40
    # 1，480，20，20
    outs = model_backbone(fake_inputs)
    model_fpn = DBFPN(model_backbone.out_channels, out_channels=256)
    fpn_outs = model_fpn(outs)
    print(fpn_outs.shape)



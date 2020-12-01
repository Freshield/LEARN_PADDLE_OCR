# coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: b3_backbone.py
@Time: 2022-11-16 21:26
@Last_update: 2022-11-16 21:26
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
from PaddleOCR.ppocr.modeling.backbones.det_mobilenet_v3 import MobileNetV3

fake_inputs = paddle.randn([1, 3, 640, 640], dtype='float32')

model_backbone = MobileNetV3()
model_backbone.eval()

# 输出为所需要的中间输出分别尺度为：
# 输入为1，3，640，640的情况下
# 1，16，160，160
# 1，24，80，80
# 1，56，40，40
# 1，480，20，20
outs = model_backbone(fake_inputs)

print(model_backbone)

for idx, out in enumerate(outs):
    print(f'The index is {idx} and the shape of output is {out.shape}')
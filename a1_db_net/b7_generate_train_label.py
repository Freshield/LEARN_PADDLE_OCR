# coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: b7_generate_train_label.py
@Time: 2022-12-16 16:48
@Last_update: 2022-12-16 16:48
@Desc: None
@==============================================@
@      _____             _   _     _   _       @
@     |   __|___ ___ ___| |_|_|___| |_| |      @
@     |   __|  _| -_|_ -|   | | -_| | . |      @
@     |__|  |_| |___|___|_|_|_|___|_|___|      @
@                                    Freshield @
@==============================================@
"""

import cv2
import json
import matplotlib.pyplot as plt
from paddleocr.ppocr.data.imaug.label_ops import DetLabelEncode
from paddleocr.ppocr.data.imaug.make_border_map import MakeBorderMap
from paddleocr.ppocr.data.imaug.make_shrink_map import MakeShrinkMap


def generate_train_label(image_data, res_label_list):
    """
    生成训练所有的label数据
    1. 对数据进行整合
    2. 解析points的数据
    3. 生成border map图像
    4. 生成shrink map图像
    """
    # 1. 对数据进行整合
    if type(res_label_list) is not str:
        res_label_list = json.dumps(res_label_list)
    data_dict = {'image': image_data, 'label': res_label_list}
    # 2. 解析points的数据
    decode_label = DetLabelEncode()
    data_dict = decode_label(data_dict)
    # 3. 生成border map图像
    generate_text_border = MakeBorderMap()
    data_dict = generate_text_border(data_dict)
    # 4. 生成shrink map图像
    generate_shrink_map = MakeShrinkMap()
    data_dict = generate_shrink_map(data_dict)

    return data_dict


if __name__ == '__main__':
    image = cv2.imread('data/aug_image.jpg')
    with open('data/aug_label_list.json', 'r') as f:
        gt_label = json.loads(f.read())
    for sub_dict in gt_label:
        new_points = []
        for point in sub_dict['points']:
            x, y = point
            new_points.append([int(x), int(y)])
        sub_dict['points'] = new_points
    print(gt_label)
    data = generate_train_label(image, gt_label)

    cv2.imshow('test', data['image'])
    cv2.imshow('test1', data['threshold_map'])
    cv2.imshow('test2', data['shrink_map'])
    cv2.waitKey()
    exit()

    plt.figure(figsize=(10, 10))
    plt.imshow(data['image'])
    plt.figure(figsize=(10, 10))
    plt.imshow(data['threshold_map'])
    plt.figure(figsize=(10, 10))
    plt.imshow(data['shrink_map'])
    plt.show()

# coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: b6_simple_run_data_preprocess.py
@Time: 2022-12-05 16:34
@Last_update: 2022-12-05 16:34
@Desc: None
@==============================================@
@      _____             _   _     _   _       @
@     |   __|___ ___ ___| |_|_|___| |_| |      @
@     |   __|  _| -_|_ -|   | | -_| | . |      @
@     |__|  |_| |___|___|_|_|_|___|_|___|      @
@                                    Freshield @
@==============================================@
"""
import json
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PaddleOCR.ppocr.data.imaug.operators import DecodeImage
from PaddleOCR.ppocr.data.imaug.label_ops import DetLabelEncode
from PaddleOCR.ppocr.data.imaug.make_border_map import MakeBorderMap
from PaddleOCR.ppocr.data.imaug.make_shrink_map import MakeShrinkMap


if __name__ == '__main__':
    data_dir = '/home/freshield/Projects/codes/person/LEARN_PADDLE_OCR/a1_db_net/train_data/icdar2015/text_localization'
    label_path = os.path.join(data_dir, 'train_icdar2015_label.txt')
    img_dir = data_dir

    with open(label_path, 'r') as f:
        lines = f.readlines()

    line = lines[0]
    print(line)
    img_name, gt_label = line.strip().split('\t')

    img_dir = '/home/freshield/Projects/codes/zkme/id_ocr/a4_create_chinese_id/data/merge_data'
    img_name = '0001.jpg'

    label_path = '/home/freshield/Projects/codes/zkme/id_ocr/a4_create_chinese_id/data/merge_data/0001.json'
    with open(label_path, 'r') as f:
        label_data = json.loads(f.read())
    gt_label = []
    for key, sub_dict in label_data.items():
        tmp_dict = {'transcription': key}
        bbox = sub_dict['bbox']
        y_min, y_max, x_min, xmax  = bbox
        points = [
            [x_min, y_max], [xmax, y_max], [xmax, y_min], [x_min, y_min]
        ]
        tmp_dict['points'] = points
        gt_label.append(tmp_dict)
    print(json.dumps(gt_label, indent=4, ensure_ascii=False))
    gt_label = json.dumps(gt_label)

    with open(os.path.join(img_dir, img_name), 'rb') as f:
        image = f.read()
    data = {'image': image, 'label': gt_label}
    decode_image = DecodeImage(img_mode='RGB', channel_first=False)
    data = decode_image(data)
    src_img = data['image']

    decode_label = DetLabelEncode()
    data = decode_label(data)

    generate_text_border = MakeBorderMap()
    data = generate_text_border(data)

    generate_shrink_map = MakeShrinkMap()
    data = generate_shrink_map(data)

    plt.figure(figsize=(10, 10))
    plt.imshow(data['image'])
    plt.figure(figsize=(10, 10))
    plt.imshow(data['threshold_map'])
    plt.figure(figsize=(10, 10))
    plt.imshow(data['shrink_map'])
    plt.show()


# coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: b2_run_ocr_by_code.py
@Time: 2022-11-13 21:49
@Last_update: 2022-11-13 21:49
@Desc: None
@==============================================@
@      _____             _   _     _   _       @
@     |   __|___ ___ ___| |_|_|___| |_| |      @
@     |   __|  _| -_|_ -|   | | -_| | . |      @
@     |__|  |_| |___|___|_|_|_|___|_|___|      @
@                                    Freshield @
@==============================================@
"""
from paddleocr import PaddleOCR
import numpy as np
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ocr = PaddleOCR()
    img_path = 'data/12.jpg'

    # rec为false表示只进行目标检测
    result = ocr.ocr(img_path, rec=False)
    print(f'The predicted text box of {img_path} are follows.')
    print(result)

    image = cv2.imread(img_path)
    boxes = [line[0] for line in result]
    for box in result[0]:
        box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.show()


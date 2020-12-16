# coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: b1_run_ocr.py
@Time: 2022-11-13 21:47
@Last_update: 2022-11-13 21:47
@Desc: None
@==============================================@
@      _____             _   _     _   _       @
@     |   __|___ ___ ___| |_|_|___| |_| |      @
@     |   __|  _| -_|_ -|   | | -_| | . |      @
@     |__|  |_| |___|___|_|_|_|___|_|___|      @
@                                    Freshield @
@==============================================@
"""
import os


if __name__ == '__main__':
    # rec为false表示值进行文本检测
    cmd = 'paddleocr --image_dir ./data/12.jpg --rec false'
    os.system(cmd)

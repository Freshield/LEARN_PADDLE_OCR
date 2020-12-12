# coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: b5_db_head.py
@Time: 2022-12-04 22:41
@Last_update: 2022-12-04 22:41
@Desc: None
@==============================================@
@      _____             _   _     _   _       @
@     |   __|___ ___ ___| |_|_|___| |_| |      @
@     |   __|  _| -_|_ -|   | | -_| | . |      @
@     |__|  |_| |___|___|_|_|_|___|_|___|      @
@                                    Freshield @
@==============================================@
"""
import math
import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle import ParamAttr

paddle.reciprocal
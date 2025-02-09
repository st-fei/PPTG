'''
2024.9.30 by stf
一些可复用的工具函数
'''

import os
import re
import json
import torch
import logging
import random
import numpy as np
import matplotlib.pyplot as plt


def create_logger(log_path):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=log_path, mode='w')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger

# 用于加载json数据
def load_json(data_path):
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
            return data
    except:
        raise FileNotFoundError(f'Fail to load the data file {data_path}, please check if the data file exists')


# 移除对话文本中的表情
def remove_emoji(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  
        u"\U0001F300-\U0001F5FF"  
        u"\U0001F680-\U0001F6FF"  
        u"\U0001F1E0-\U0001F1FF"  
        u"\U00002702-\U000027B0"
        u"\U00010000-\U0010ffff"
        u"\U0001f926-\U0001f937"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  
        u"\u3030"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# 用于清洗文本
def text_process(text):
    # clean the text data
    text = text.strip().lower()
    text = text.replace(' ', '')
    text = remove_emoji(text)
    return text.strip()

# 获取ground-truth标题文本
def get_labels(data_dir, mode='test'):
    assert mode in ['dev', 'test'], 'mode需设定为dev或test'
    data_file = os.path.join(data_dir, f'{mode}.json')
    data = load_json(data_file)
    # 获得所有匿名用户的列表
    anonymous_list = list(data.keys())
    # 创建存储ground-truth的一维列表
    gth_response = [[]]
    # 遍历匿名用户信息
    for anonymous_name in anonymous_list:
        title = text_process(data[anonymous_name]['target']['title'])
        gth_response[0].append(title)
    return gth_response

# 查询当前图像index所对应的图像文件夹
def query(img_dir, navigation_path, index):
    navigation_data = load_json(navigation_path)
    for idx, detail in navigation_data.items():
        st = detail["start"]
        ed = detail["end"]
        if index >= st and index <= ed:
            return os.path.join(img_dir, f"image_base{idx}")

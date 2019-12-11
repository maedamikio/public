#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""共通設定."""


import os


CX = 'YOUR-SEARCH-ENGINE-ID'
DEVELOPER_KEY = 'YOUR-API-KEY'

IMG_SIZES = [
    'huge',
    # 'icon',
    'large',
    'medium',
    'small',
    'xlarge',
    'xxlarge'
]

CLASSES = [
    '安倍乙',
    '石原さとみ',
    '大原優乃',
    '小芝風花',
    '川口春奈',
    '森七菜',
    '浜辺美波',
    '清原果耶',
    '福原遥',
    '黒島結菜'
]

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_PATH, 'data')
LINK_PATH = os.path.join(DATA_PATH, 'link')
DOWNLOAD_PATH = os.path.join(DATA_PATH, 'download')
FACE_PATH = os.path.join(DATA_PATH, 'face')

HAARCASCADE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'haarcascade_frontalface_default.xml')

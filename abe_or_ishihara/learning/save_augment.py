#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""学習画像の水増し."""


import argparse
import glob
import math
import os
import shutil

import numpy as np
from PIL import Image, ImageFile

from config import CLASSES, TRAIN_PATH, AUGMENT_PATH, AUGMENT_NUM


ImageFile.LOAD_TRUNCATED_IMAGES = True


def augment(query):
    """学習画像の読み込み、水増し、保存."""

    train_path = os.path.join(TRAIN_PATH, query)
    if not os.path.isdir(train_path):
        print('no train path: {}'.format(train_path))
        return

    augment_path = os.path.join(AUGMENT_PATH, query)
    if os.path.isdir(augment_path):
        shutil.rmtree(augment_path)
    os.makedirs(augment_path, exist_ok=True)

    train_list = glob.glob(os.path.join(train_path, '*.jpeg'))
    train_list.sort()

    loop_num = math.ceil(AUGMENT_NUM / len(train_list))

    augment_num = 0
    for num in range(1, loop_num + 1):
        for train_file in train_list:
            if augment_num == AUGMENT_NUM:
                break

            image = Image.open(train_file)

            image = horizontal_flip(image)
            image = random_crop(image)

            augment_file = os.path.join(AUGMENT_PATH, query, os.path.basename(train_file).split('.')[0] + '-{:04d}.jpeg'.format(num))
            image.save(augment_file, optimize=True, quality=95)
            print('query: {}, train_file: {}, augment_file: {}'.format(
                query, os.path.basename(train_file), os.path.basename(augment_file)))

            augment_num += 1

    print('query: {}, augment num: {}'.format(query, augment_num))


def horizontal_flip(image, rate=0.5):
    """水平方向に反転."""

    image = np.array(image, dtype=np.float32)

    if np.random.rand() < rate:
        image = np.fliplr(image)

    return Image.fromarray(np.uint8(image))


def random_crop(image, size=0.8):
    """ランダムなサイズでクロップ."""

    image = np.array(image, dtype=np.float32)

    height, width, _ = image.shape
    crop_size = int(min(height, width) * size)

    top = np.random.randint(0, height - crop_size)
    left = np.random.randint(0, width - crop_size)
    bottom = top + crop_size
    right = left + crop_size
    image = image[top:bottom, left:right, :]

    return Image.fromarray(np.uint8(image))


def main(_):
    """config.py の CLASSES、もしくは引数で処理を実施."""

    os.makedirs(AUGMENT_PATH, exist_ok=True)

    if _.query:
        query_list = [_.query]
    else:
        query_list = CLASSES

    for query in query_list:
        augment(query)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='学習画像の水増し')
    parser.add_argument('--query', help='例: 安倍乙')
    args = parser.parse_args()
    main(args)

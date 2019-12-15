#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""データセットの作成."""


import argparse
import glob
import os
import pickle
import random
import tqdm

import numpy as np

from PIL import Image, ImageFile

from config import CLASSES, TRAIN_PATH, TEST_PATH, AUGMENT_PATH, DATASETS_PATH, AUGMENT_NUM, IMG_ROWS, IMG_COLS


ImageFile.LOAD_TRUNCATED_IMAGES = True


def make_filesets(augment):
    """ファイルセットの作成."""

    filesets = {'train': dict(), 'test': dict(), 'augment': dict()}

    for query in CLASSES:

        train_path = os.path.join(TRAIN_PATH, query)
        test_path = os.path.join(TEST_PATH, query)
        augment_path = os.path.join(AUGMENT_PATH, query)

        if not os.path.isdir(train_path):
            print('no train path: {}'.format(train_path))
            return None
        if not os.path.isdir(test_path):
            print('no test path: {}'.format(test_path))
            return None
        if not os.path.isdir(augment_path):
            print('no augment path: {}'.format(augment_path))
            return None

        train_files = glob.glob(os.path.join(train_path, '*.jpeg'))
        train_files.sort()
        filesets['train'][query] = train_files

        test_files = glob.glob(os.path.join(test_path, '*.jpeg'))
        test_files.sort()
        filesets['test'][query] = test_files

        augment_files = glob.glob(os.path.join(augment_path, '*.jpeg'))
        random.shuffle(augment_files)
        filesets['augment'][query] = augment_files

        if augment and len(augment_files) < AUGMENT_NUM:
            print('less augment num: {}, path: {}'.format(len(augment_files), augment_path))
            return None

    return filesets


def make_datasets(augment, filesets):
    """データセットの作成."""

    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    for num, query in enumerate(CLASSES):
        print('create dataset: {}'.format(query))

        if augment:
            train_files = filesets['augment'][query][:AUGMENT_NUM]
        else:
            train_files = filesets['train'][query]
        test_files = filesets['test'][query]

        for train_file in tqdm.tqdm(train_files, desc='create train', leave=False):
            train_images.append(read_image(train_file))
            train_labels.append(num)
        for test_file in tqdm.tqdm(test_files, desc='create test', leave=False):
            train_images.append(read_image(test_file))
            test_labels.append(num)

    datasets = ((np.array(train_images), (np.array(train_labels))), (np.array(test_images), (np.array(test_labels))))

    datasets_path = os.path.join(DATASETS_PATH, ','.join(CLASSES))
    os.makedirs(datasets_path, exist_ok=True)
    train_num = AUGMENT_NUM if augment else 0
    datasets_file = os.path.join(datasets_path, '{}x{}-{}.pickle'.format(IMG_ROWS, IMG_COLS, train_num))
    with open(datasets_file, 'wb') as fout:
        pickle.dump(datasets, fout)
    print('save datasets: {}'.format(datasets_file))


def read_image(filename):
    """画像の読み込み、リサイズ、グレー変換."""

    image = Image.open(filename)
    image = image.resize((IMG_ROWS, IMG_COLS), Image.LANCZOS)
    image = image.convert('L')
    image = np.array(image, dtype=np.uint8)

    return image


def main(_):
    """config.py の CLASSES、引数で処理を実施."""

    os.makedirs(DATASETS_PATH, exist_ok=True)
    filesets = make_filesets(_.augment)
    if filesets:
        make_datasets(_.augment, filesets)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='データセットの作成')
    parser.add_argument('--augment', action='store_true', help='水増し画像の利用')
    args = parser.parse_args()
    main(args)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""顔画像を学習とテストに分割."""


import argparse
import glob
import os
import random
import shutil

from config import CLASSES, FACE_PATH, TRAIN_PATH, TEST_PATH, TEST_NUM


def split(query):
    """顔画像の一覧の取得、学習とテストに分割しコピー."""

    face_path = os.path.join(FACE_PATH, query)
    if not os.path.isdir(face_path):
        print('no face path: {}'.format(face_path))
        return

    train_path = os.path.join(TRAIN_PATH, query)
    test_path = os.path.join(TEST_PATH, query)
    if os.path.isdir(train_path):
        shutil.rmtree(train_path)
    if os.path.isdir(test_path):
        shutil.rmtree(test_path)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    face_file_list = glob.glob(os.path.join(face_path, '*.jpeg'))
    face_file_list.sort()

    random.shuffle(face_file_list)

    train_file_list = face_file_list[:-TEST_NUM]
    test_file_list = face_file_list[len(train_file_list):]

    for face_file in train_file_list:
        train_file = os.path.join(train_path, os.path.basename(face_file))
        shutil.copy(face_file, train_file)

    for face_file in test_file_list:
        test_file = os.path.join(test_path, os.path.basename(face_file))
        shutil.copy(face_file, test_file)

    print('query: {}, face: {}, train: {}, test: {}'.format(
        query, len(face_file_list), len(train_file_list), len(test_file_list)))


def main(_):
    """config.py の CLASSES、もしくは引数で処理を実施."""

    os.makedirs(TRAIN_PATH, exist_ok=True)
    os.makedirs(TEST_PATH, exist_ok=True)

    if _.query:
        query_list = [_.query]
    else:
        query_list = CLASSES

    for query in query_list:
        split(query)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='顔画像を学習とテストに分割')
    parser.add_argument('--query', help='例: 安倍乙')
    args = parser.parse_args()
    main(args)

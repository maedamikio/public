#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""顔画像の検出と保存."""


import argparse
import glob
import os

import cv2

from config import CLASSES, DOWNLOAD_PATH, FACE_PATH, HAARCASCADE_PATH


def detect(query):
    """画像の読み込み、顔画像の検出、顔画像の保存."""

    download_path = os.path.join(DOWNLOAD_PATH, query)
    if not os.path.isdir(download_path):
        print('no download path: {}'.format(download_path))
        return

    os.makedirs(os.path.join(FACE_PATH, query), exist_ok=True)

    face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)

    download_list = glob.glob(os.path.join(download_path, '*.jpeg'))
    download_list.sort()

    for download in download_list:

        img = cv2.imread(download)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)

        if len(faces) < 1:
            print('query: {}, download: {}, face num: 0'.format(query, os.path.basename(download)))
            continue

        for num, (x, y, w, h) in enumerate(faces, start=1):
            face = img[y:y+h, x:x+w]

            filename = os.path.join(FACE_PATH, query, os.path.basename(download).split('.')[0] + '-{:04d}.jpeg'.format(num))
            cv2.imwrite(filename, face)
            print('query: {}, download: {}, filename: {}'.format(query, os.path.basename(download), os.path.basename(filename)))


def main(_):
    """config.py の CLASSES、もしくは引数で検索を実施."""

    os.makedirs(FACE_PATH, exist_ok=True)

    if _.query:
        query_list = [_.query]
    else:
        query_list = CLASSES

    for query in query_list:
        detect(query)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='顔画像の検出と保存')
    parser.add_argument('--query', help='例: 安倍乙')
    args = parser.parse_args()
    main(args)

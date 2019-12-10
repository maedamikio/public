#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""画像のダウンロードと保存."""


import argparse
import io
import os

import requests
from PIL import Image, ImageFile

from config import CLASSES, LINK_PATH, DOWNLOAD_PATH


ImageFile.LOAD_TRUNCATED_IMAGES = True


def download(query):
    """データのダウンロード、データのチェック、画像の保存."""

    linkfile = os.path.join(LINK_PATH, '{}.txt'.format(query))
    if not os.path.isfile(linkfile):
        print('no linkfile: {}'.format(linkfile))
        return

    with open(linkfile, 'r') as fin:
        link_list = fin.read().split('\n')[:-1]

    os.makedirs(os.path.join(DOWNLOAD_PATH, query), exist_ok=True)

    for num, link in enumerate(link_list, start=1):

        try:
            result = requests.get(link)
            content = result.content
            content_type = result.headers['Content-Type']
        except Exception as err:
            print('err: {}, link: {}'.format(err, link))
            continue

        if not content_type.startswith('image/'):
            print('err: {}, link: {}'.format(content_type, link))
            continue

        try:
            image = Image.open(io.BytesIO(content))
        except Exception as err:
            print('err: {}, link: {}'.format(err, link))
            continue

        if image.mode != 'RGB':
            image = image.convert('RGB')
        data = io.BytesIO()
        image.save(data, 'jpeg', optimize=True, quality=95)
        content = data.getvalue()

        filename = os.path.join(DOWNLOAD_PATH, query, '{:04d}.jpeg'.format(num))
        with open(filename, 'wb') as fout:
            fout.write(content)
        print('query: {}, filename: {}, link: {}'.format(query, os.path.basename(filename), link))


def main(_):
    """config.py の CLASSES、もしくは引数で検索を実施."""

    os.makedirs(DOWNLOAD_PATH, exist_ok=True)

    if _.query:
        query_list = [_.query]
    else:
        query_list = CLASSES

    for query in query_list:
        download(query)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='画像のダウンロードと保存')
    parser.add_argument('--query', help='例: 安倍乙')
    args = parser.parse_args()
    main(args)

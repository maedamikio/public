#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""画像の検索とリンクの保存."""


import argparse
import os
import pprint

from googleapiclient.discovery import build

from config import CX, DEVELOPER_KEY, IMG_SIZES, CLASSES, LINK_PATH


def customsearch(query):
    """Googleカスタムサーチ 画像用."""

    service = build('customsearch', 'v1', developerKey=DEVELOPER_KEY)

    link_list = []

    for img_size in IMG_SIZES:

        start_index = 1

        while True:
            try:
                result = service.cse().list(
                    q=query,
                    cx=CX,
                    imgSize=img_size,
                    # imgType='face',
                    lr='lang_ja',
                    num=10,
                    searchType='image',
                    start=start_index
                ).execute()
                link_list.extend([item['link'] for item in result['items']])
                start_index = result['queries']['nextPage'][0]['startIndex']
            except Exception as err:
                pprint.pprint(err)
                break
            if start_index > 100:
                break

    link_list = [link for link in link_list if link.startswith('http')]
    link_list = list(set(link_list))

    filename = os.path.join(LINK_PATH, '{}.txt'.format(query))
    with open(filename, 'w') as fout:
        fout.write('\n'.join(link_list)+'\n')
    print('query: {}, link num: {}, filename: {}'.format(query, len(link_list), filename))

    return link_list


def main(_):
    """config.py の CLASSES、もしくは引数で検索を実施."""

    os.makedirs(LINK_PATH, exist_ok=True)

    if _.query:
        query_list = [_.query]
    else:
        query_list = CLASSES

    for query in query_list:
        customsearch(query)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='画像の検索とリンクの保存')
    parser.add_argument('--query', help='例: 安倍乙')
    args = parser.parse_args()
    main(args)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""画像閲覧ツール."""


import glob
import io
import os
import pprint

import numpy as np

from flask import Flask, abort, make_response, render_template, request
from PIL import Image, ImageFile

from config import CLASSES, DATA_PATH, DOWNLOAD_PATH, FACE_PATH, IMG_ROWS, IMG_COLS

import face_deep

app = Flask(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True


@app.route('/')
def index():
    """Top Page."""
    return render_template('index.html', items=CLASSES)


@app.route('/download_and_face/<item>', methods=['GET', 'POST'])
def download_and_face(item):
    """ダウンロード画像、顔画像."""

    if request.method == 'POST' and request.form.get('action') == 'delete':
        for filename in request.form.getlist('filename'):
            filename = os.path.join(FACE_PATH, item, filename)
            if os.path.isfile(filename):
                os.remove(filename)
                print('delete face image: {}'.format(filename))

    download_list = glob.glob(os.path.join(DOWNLOAD_PATH, item, '*.jpeg'))
    download_list = sorted([os.path.basename(filename) for filename in download_list])
    face_list = glob.glob(os.path.join(FACE_PATH, item, '*.jpeg'))
    face_list = sorted([os.path.basename(filename) for filename in face_list])

    rows = []
    for download in download_list:
        row = [download]
        key = download.split('.')[0] + '-'
        for face in face_list:
            if face.startswith(key):
                row.append(face)
        rows.append(row)

    return render_template('download_and_face.html', item=item, rows=rows)


@app.route('/data/<folder>/<item>/<filename>')
def get_image(folder, item, filename):
    """画像のレスポンス size で拡大縮小."""

    if folder not in ['download', 'face', 'train', 'test']:
        abort(404)

    filename = os.path.join(DATA_PATH, folder, item, filename)

    try:
        image = Image.open(filename)
    except Exception as err:
        pprint.pprint(err)
        abort(404)

    if 'size' in request.args:
        height = int(request.args.get('size'))
        width = int(image.size[0] * height / image.size[1])
        image = image.resize((width, height), Image.LANCZOS)

    data = io.BytesIO()
    image.save(data, 'jpeg', optimize=True, quality=95)
    response = make_response()
    response.data = data.getvalue()
    response.mimetype = 'image/jpeg'

    return response


@app.route('/predict/<folder>/<item>')
def predict(folder, item):
    """画像の推論."""

    if folder not in ['train', 'test']:
        abort(404)

    filename_list = sorted(glob.glob(os.path.join(DATA_PATH, folder, item, '*.jpeg')))

    image_list = []
    for filename in filename_list:

        face = Image.open(filename)
        face = face.resize((IMG_ROWS, IMG_COLS), Image.LANCZOS)
        face = face.convert('L')
        face = np.array(face, dtype=np.float32) / 255.0
        face = np.ravel(face)
        image_list.append(face)

    percent_list = face_deep.predict(image_list, dtype='int')

    rows = []
    for filename, percent in zip(filename_list, percent_list):
        color = CLASSES.index(item) in [index for index, value in enumerate(percent) if value == max(percent)]
        row = {'filename': os.path.basename(filename), 'percent': percent, 'color': color}
        rows.append(row)

    return render_template('predict.html', folder=folder, item=item, headers=CLASSES, rows=rows)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

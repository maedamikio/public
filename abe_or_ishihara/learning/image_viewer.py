#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""画像閲覧ツール."""


import glob
import io
import os
import pprint

from flask import Flask, abort, make_response, render_template, request
from PIL import Image, ImageFile

from config import DATA_PATH, DOWNLOAD_PATH, FACE_PATH, CLASSES

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

    if folder not in ['download', 'face']:
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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

import base64
import datetime
import imghdr
import io
import os

import boto3
import cv2
import numpy as np
import tensorflow as tf

from PIL import Image, ImageFont, ImageDraw

from django.http import HttpResponse, Http404
from django.shortcuts import render
from django.views.defaults import bad_request, page_not_found, permission_denied, server_error

from app.models import Log


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
IMG_ROWS, IMG_COLS = 28, 28
MODEL_FILE = os.path.join(os.path.dirname(__file__), 'data/model')


def index(request):
    return render(request, 'index.html')


def api(request):

    log = Log()

    if request.method != 'POST':
        log.status = 'post error'
        log.save()
        raise Http404

    try:
        formdata = request.FILES['file']
        filename = formdata.name
        filesize = formdata.size
    except Exception as err:
        log.message = err
        log.status = 'formdata error'
        log.save()
        return server_error(request)
    log.filename = filename
    log.filesize = filesize
    log.save()

    if filesize > 10000000:
        log.status = 'filesize error'
        log.save()
        return server_error(request)

    try:
        filedata = formdata.open().read()
    except Exception as err:
        log.message = err
        log.status = 'filedata error'
        log.save()
        return server_error(request)

    ext = imghdr.what(None, h=filedata)
    if ext not in ['jpeg', 'png']:
        log.message = ext
        log.status = 'filetype error'
        log.save()
        return server_error(request)

    try:
        s3_key = save_image(filedata, filename)
    except Exception as err:
        log.message = err
        log.status = 's3 error'
        log.save()
        return server_error(request)
    log.s3_key = s3_key
    log.save()

    image = np.fromstring(filedata, np.uint8)
    image = cv2.imdecode(image, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml'))
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if type(faces) != np.ndarray:
        log.status = 'faces error'
        log.save()
        return server_error(request)

    face_list = []
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (IMG_ROWS, IMG_COLS))
        face = np.array(face, dtype=np.float32) / 255.0
        face = np.ravel(face)
        face_list.append(face)

    try:
        percent_list = predict(face_list, dtype='int')
    except Exception as err:
        log.message = err
        log.status = 'predict error'
        log.save()
        return server_error(request)

    predict_list = []
    for (x, y, w, h), percent in zip(faces, percent_list):
        max_index = np.argmax(percent)
        max_value = np.amax(percent)
        if max_index == 0:
            color = (177, 107, 1)
        elif max_index == 1:
            color = (15, 30, 236)
        else:
            color = (0, 0, 0)
        text = '{} {}%'.format(CLASSES[max_index], max_value)
        image = write_text(image, text, (x, y+h+10), color, int(h/10))
        cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness=2)
        predict_list.append(text)
    log.message = ','.join(predict_list)

    image = cv2.imencode('.jpeg', image)[1].tostring()
    response = 'data:image/jpeg;base64,' + base64.b64encode(image).decode('utf-8')

    log.status = 'success'
    log.save()

    return HttpResponse(response, status=200)


def write_text(image, text, xy, color, size):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    fill = (color[2], color[1], color[0])
    size = size if size > 16 else 16
    font = ImageFont.truetype(os.path.join(os.path.dirname(__file__), 'font.ttf'), size)
    draw = ImageDraw.Draw(image)
    draw.text(xy, text, font=font, fill=fill)

    image = np.array(image, dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image


def save_image(filedata, filename):

    now = (datetime.datetime.utcnow() + datetime.timedelta(hours=9)).strftime('%Y-%m-%dT%H:%M:%S+09:00')
    key = '{}_{}'.format(now, filename)
    resource = boto3.resource('s3', aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'], aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])
    resource.Object(os.environ['BUCKET'], key).put(Body=filedata)

    return key


def model():
    """MNIST 参考モデル."""

    num_classes = len(CLASSES)
    img_rows, img_cols = IMG_ROWS, IMG_COLS

    x = tf.compat.v1.placeholder(tf.float32, [None, img_rows*img_cols])

    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, img_rows, img_cols, 1])

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([int(h_pool2.shape[1]) * int(h_pool2.shape[2]) * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, int(h_pool2.shape[1]) * int(h_pool2.shape[2]) * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.name_scope('dropout'):
        keep_prob = tf.compat.v1.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, rate=1-keep_prob)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, num_classes])
        b_fc2 = bias_variable([num_classes])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return x, y_conv, keep_prob


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def predict(images, dtype=None):
    """推論 結果は numpy, int, argmax を dtype で切り替え."""

    tf.compat.v1.reset_default_graph()

    x, y_conv, keep_prob = model()

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, MODEL_FILE)

        results = sess.run(tf.nn.softmax(y_conv), feed_dict={x: images, keep_prob: 1.0})
        results = np.array(results * 100, dtype=np.uint8)
        if dtype == 'int':
            results = [[int(y) for y in result] for result in results]
        if dtype == 'argmax':
            results = [np.argmax(y) for y in results]

    return results

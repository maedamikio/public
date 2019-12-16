#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""学習と推論 deep."""


import argparse
import os

import numpy as  np
import tensorflow as tf

from datasets import load_data

from config import CLASSES, MODEL_FILE, IMG_ROWS, IMG_COLS


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


def train(datasets, batch_size=128, epochs=12):
    """学習."""

    x, y_conv, keep_prob = model()

    y_ = tf.compat.v1.placeholder(tf.float32, [None, 10])

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    saver = tf.compat.v1.train.Saver()
    os.makedirs(os.path.dirname(os.path.abspath(MODEL_FILE)), exist_ok=True)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        next_epoch = 1
        print('epoch, train accuracy, test accuracy')
        while datasets.train.epochs_completed < epochs:
            train_images, train_labels = datasets.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: train_images, y_: train_labels, keep_prob: 0.5})

            if datasets.train.epochs_completed == next_epoch:

                train_accuracy = accuracy.eval(feed_dict={x: datasets.train.images, y_: datasets.train.labels, keep_prob: 1.0})
                test_accuracy = accuracy.eval(feed_dict={x: datasets.test.images, y_: datasets.test.labels, keep_prob: 1.0})
                print('{:d}, {:.4f}, {:.4f}'.format(datasets.train.epochs_completed, train_accuracy, test_accuracy))

                saver.save(sess, MODEL_FILE)

                next_epoch = datasets.train.epochs_completed + 1


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


def main(_):
    """config.py の CLASSES、引数で処理を実施."""

    datasets = load_data(one_hot=True)

    if _.train:
        train(datasets, epochs=120)
    else:
        print(datasets.test.labels[:10])
        print(predict(datasets.test.images[:10]))
        print(predict([datasets.test.images[0]]))
        print(predict([datasets.test.images[0]], dtype='int'))
        print(predict([datasets.test.images[0]], dtype='argmax'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='学習と推論 deep.')
    parser.add_argument('--train', action='store_true', help='例: 学習は --train を指定')
    args = parser.parse_args()
    main(args)

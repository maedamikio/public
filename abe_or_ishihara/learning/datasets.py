#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""データセットの読み出し."""


import argparse
import collections
import os
import pickle

import numpy

from config import CLASSES, DATASETS_PATH, AUGMENT_NUM, USE_AUGMENT, IMG_ROWS, IMG_COLS


Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


class DataSet():
    """データセットの管理."""

    def __init__(self, images, labels):
        self._num_examples = images.shape[0]
        images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return numpy.concatenate((images_rest_part, images_new_part), axis=0), numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def load_data(one_hot=False, validation_size=0):
    """データセットを config.py に従い読み込み."""

    train_num = AUGMENT_NUM if USE_AUGMENT else 0
    datasets_file = os.path.join(DATASETS_PATH, ','.join(CLASSES), '{}x{}-{}.pickle'.format(IMG_ROWS, IMG_COLS, train_num))

    if not os.path.isfile(datasets_file):
        print('no datasets file: {}'.format(datasets_file))
        return

    with open(datasets_file, 'rb') as fin:
        (train_images, train_labels), (test_images, test_labels) = pickle.load(fin)

    if one_hot:
        num_classes = len(numpy.unique(train_labels))
        train_labels = dense_to_one_hot(train_labels, num_classes)
        test_labels = dense_to_one_hot(test_labels, num_classes)

    perm = numpy.arange(train_images.shape[0])
    numpy.random.shuffle(perm)
    train_images = train_images[perm]
    train_labels = train_labels[perm]

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    train = DataSet(train_images, train_labels)
    validation = DataSet(validation_images, validation_labels)
    test = DataSet(test_images, test_labels)

    return Datasets(train=train, validation=validation, test=test)


def main(_):
    """config.py の CLASSES、引数で処理を実施."""

    face = load_data()

    for num in range(20000):
        train_images, train_labels = face.train.next_batch(50)
        print(num, train_images.shape, train_labels.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='データセットの読み出し.')
    args = parser.parse_args()
    main(args)

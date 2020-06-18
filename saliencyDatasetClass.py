
import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np


def load_train(trainSamples, image_size, labels):
    images = []

    print('Going to read training images')
    for sample in trainSamples:
        image = sample
        # image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
        # image = image.astype(np.float32)
        # image = np.multiply(image, 1.0 / 255.0)
        images.append(image)
    return images, labels


class DataSet(object):
    def __init__(self, images, labels):
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._epochs_done = 0
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
    def epochs_done(self):
        return self._epochs_done

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # After each epoch we update this
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end]


def read_train_sets(trainSamples, image_size, labels, validation_size):
    class DataSets(object):
        pass

    data_sets = DataSets()

    images, labels = load_train(trainSamples, image_size, labels)
    images, labels = shuffle(trainSamples, labels)

    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])

    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]

    train_images = images[validation_size:]
    train_labels = labels[validation_size:]

    data_sets.train = DataSet(train_images, train_labels)
    data_sets.valid = DataSet(validation_images, validation_labels)

    return data_sets
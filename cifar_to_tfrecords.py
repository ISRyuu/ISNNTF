import numpy as np
import _pickle as cPickle
import os, sys
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import python_io as tpio

def load_pickle(f):
    return cPickle.load(f, encoding="latin1")

def load_CIFAR10_batch(path):
    with open(path, "rb") as f:
        data = load_pickle(f)
        X = data['data']
        Y = data['labels']
        # see CIFAR-10 format
        X = np.reshape(X, (10000, 3, 32, 32)).transpose(0, 2, 3, 1).astype('float32')
        mean = np.mean(X)
        max = np.max(np.max(X))
        X = (X - mean)/max
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(dir_path):
    xs = []
    ys = []
    for i in range(1, 6):
        X, Y = load_CIFAR10_batch(os.path.join(dir_path, "data_batch_%d" % i))
        xs.append(X)
        ys.append(Y)
    X = np.concatenate(xs, axis=0)
    Y = np.concatenate(ys, axis=0)
    Xte, Yte = load_CIFAR10_batch(os.path.join(dir_path, "test_batch"))
    return X, Y, Xte, Yte

def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_features(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_tfrecords(data, out_file, GZ=True):
    image = data[0]
    labels = data[1]
    count = np.size(labels)

    filename = out_file + '.tfrecords'
    options = None
    if GZ:
        filename += '.gz'
        options = tpio.TFRecordOptions(tpio.TFRecordCompressionType.GZIP)
    if os.path.exists(filename):
        print("File %s exists" % filename)
        return
    print("writing to %s..." % filename)
    writer = tpio.TFRecordWriter(filename, options=options)

    for i in range(count):
        sys.stdout.write("%d\r" % (i+1))
        image_raw = image[i].tostring()
        example = tf.train.Example(features=tf.train.Features(
            feature={
            'label': _int64_features(labels[i]),
            'image_raw': _bytes_features(image_raw)
            }))
        writer.write(example.SerializeToString())
    writer.close()
    print("\nfinished")
    
if __name__ == '__main__':
    X, Y, Xte, Yte = load_CIFAR10("datasets/cifar-10-batches-py")
    
    convert_to_tfrecords([X, Y], "cifar10")
    convert_to_tfrecords([Xte, Yte], "cifar10_test")

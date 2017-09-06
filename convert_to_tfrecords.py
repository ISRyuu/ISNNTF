import tensorflow as tf
import numpy as np
from tensorflow import python_io as tpio

def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_features(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_tfrecords(file, data, height, width, depth, GZ=True):
    '''
    :param data: must be a numpy array with two dimension, data[0] contains image data, data[1] for labels.
    '''
    images = data[0]
    labels = data[1]
    num_ex = np.size(data[1])

    filename = file + '.tfrecords'
    options = None
    if GZ:
        filename += '.gz'
        options = tpio.TFRecordOptions(tpio.TFRecordCompressionType.GZIP)
    print('Writing ', filename)
    writer = tf.python_io.TFRecordWriter(filename, options=options)

    for i in range(num_ex):
        print(i)
        image_raw = images[i].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_features(height),
            'width': _int64_features(width),
            'depth': _int64_features(depth),
            'label': _int64_features(int(labels[i])),
            'image_raw': _bytes_features(image_raw)
        }))
        writer.write(example.SerializeToString())
    writer.close()
    print('finished')

def read_minst_from_tfrecords(filequeue, shape, one_hot=0, GZ=True):
    # Note: if num_epochs is not None, this function creates local counter epochs.
    # Use local_variables_initializer() to initialize local variables.
    # refer to document of string_input_producer().
    options = None
    if GZ:
        options = tpio.TFRecordOptions(tpio.TFRecordCompressionType.GZIP)
    reader = tf.TFRecordReader(options=options)
    _, serialized_example = reader.read(filequeue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64)
        })

    image = tf.decode_raw(features['image_raw'], tf.float32)
    image.set_shape([shape])
    label = tf.cast(features['label'], tf.int32)
    if one_hot > 0:
        label = tf.one_hot(label, one_hot)
    return image, label

def parse_function_maker(shape, one_hot=0):
    '''
    return: a function for TFRecordDataset.map()'s parameter.
    '''
    def _parse_function(example_proto):
        features = tf.parse_single_example(
            example_proto,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'depth': tf.FixedLenFeature([], tf.int64)
            })

        image = tf.decode_raw(features['image_raw'], tf.float32)
        image.set_shape([shape])
        label = tf.cast(features['label'], tf.int32)
        if one_hot > 0:
            label = tf.one_hot(label, one_hot)
        return image, label
    return _parse_function

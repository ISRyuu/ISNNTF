import tensorflow as tf
import numpy as np
from tensorflow import python_io as tpio
import os


def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_features(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class VOC_TFRecords(object):
    def __init__(self, output_file):
        self.output_file = output_file

    def __enter__(self):
        if os.path.exists(self.output_file):
            raise IOError("file %s exists"
                          % self.output_file)
        else:
            options = tpio.TFRecordOptions(tpio.TFRecordCompressionType.GZIP)
            self.writer = tpio.TFRecordWriter(self.output_file, options=options)
        return self

    def __exit__(self, type, value, trace):
        self.close()

    def add_example(self, image, annotations):
        data = VOC_TFRecords.example(image, annotations)
        self.writer.write(data.SerializeToString())

    def close(self):
        if self.writer:
            self.writer.close()

    @classmethod
    def example(cls, image, annotations):
        image_raw = image.tostring()
        annotations_raw = annotations.tostring()
        _example = tf.train.Example(features=tf.train.Features(
            feature={
                'image_raw': _bytes_features(image_raw),
                'annotations_raw': _bytes_features(annotations_raw)
            }
        ))
        return _example

    @classmethod
    def parse_function_maker(cls, shape_img, shape_anno):
        def _parse_function(example_proto):
            features = tf.parse_single_example(
                example_proto,
                features={
                    'image_raw': tf.FixedLenFeature([], tf.string),
                    'annotations_raw': tf.FixedLenFeature([], tf.string),
                })

            image = tf.decode_raw(features['image_raw'], tf.uint8)
            image = tf.cast(image, tf.float32)
            max = tf.reduce_max(image)
            min = tf.reduce_min(image)
            image = tf.div(tf.subtract(image, min), tf.subtract(max, min))
#            image = image / 255 * 2.0 - 1.0
            image.set_shape(np.prod(shape_img))
            image = tf.reshape(image, shape_img)
            
            annotations = tf.decode_raw(features['annotations_raw'], tf.float32)
            annotations.set_shape(np.prod(shape_anno))

            return image, annotations

        return _parse_function

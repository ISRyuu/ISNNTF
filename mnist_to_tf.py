1;95;0c# Spirit of Ryuu
# 2017/8/23
# Kevin
#
import numpy as np
import tensorflow as tf
import gzip
import _pickle as cPickle
from convert_to_tfrecords import convert_to_tfrecords, read_minst_from_tfrecords

def load_from_minst(path, one_hot=0):
    #  encoding='latin1' :
    # ![https://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3]
    with gzip.open(path, 'rb') as f:
        training_data, validation_data, test_data = cPickle.load(f, encoding='latin1')
        # training_data = (training_data[0][0:50], training_data[1][0:50])
        # validation_data = (validation_data[0][0:10], validation_data[1][0:10])
        # test_data = (test_data[0][0:10], test_data[1][0:10])
        def _one_hot(data):
            def __one_hot(i):
                z = [0]*one_hot
                z[i] = 1
                return z
            return [data[0], [__one_hot(e) for e in data[1]]]
        if one_hot > 0:
            return _one_hot(training_data), _one_hot(validation_data), _one_hot(test_data)
        else:
            return training_data, validation_data, test_data

import tensorflow.contrib.data as tdata
from convert_to_tfrecords import parse_function_maker

if __name__ == '__main__':
    tr, v, te = load_from_minst("mnist.pkl.gz")
    convert_to_tfrecords('training', tr, 28, 28, 1)
    convert_to_tfrecords('validation', v, 28, 28, 1)
    convert_to_tfrecords('test', te, 28, 28, 1)
    exit()
    mbs = 10
    file = 'MNIST_GZ/training.tfrecords.gz'
    vfile = 'MNIST_GZ/validation.tfrecords.gz'
    file_placeholder = tf.placeholder(dtype=tf.string)
    dataset = tdata.TFRecordDataset(file_placeholder, compression_type='GZIP')
    dataset = dataset.map(parse_function_maker(784))
    dataset = dataset.batch(mbs)
    # currently there is no option like "allow_smaller_final_batch" in tf.train.batch
    # using the filter is an alternative way.
    dataset = dataset.filter(lambda x, y: tf.equal(tf.shape(y)[0], mbs))
    # iterate the whole dataset once an initiation.
    dataset = dataset.repeat(1)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    sess = tf.Session()
    sess.run(iterator.initializer, feed_dict={file_placeholder: vfile})

    while True:
        try:
            print(sess.run(tf.shape(next_element[0])))
        except tf.errors.OutOfRangeError:
            print('complete an epoch')
            break

    # filequeue = tf.train.string_input_producer([file], num_epochs=10)
    # img, lab = read_minst_from_tfrecords(filequeue, 784)
    #
    # i, l = tf.train.shuffle_batch([img, lab], batch_size=10, capacity=1030,
    #                               min_after_dequeue=1000, num_threads=2)
    #
    # sess = tf.Session()
    # init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    # sess.run(init)
    #
    # coord = tf.train.Coordinator()
    # thread = tf.train.start_queue_runners(sess=sess, coord=coord)
    # l = tf.Print(l, data=[tf.size(l)], message="size")
    #
    #
    # try:
    #     sess.run([l])
    # except tf.errors.OutOfRangeError:
    #     print("finishied")
    #
    # try:
    #     coord.request_stop()
    #     coord.join(thread)
    # except (tf.errors.NotFoundError) as e:
    #     print("file not found:", e)
    #
    # sess.close()


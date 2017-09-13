# Spirit of Ryuu
# 2017/8/15
# Kevin

import time
import tensorflow as tf
import numpy as np
import gzip
import _pickle as cPickle
from mnist_to_tf import load_from_minst
from abc import ABCMeta, abstractmethod, abstractproperty
from tensorflow.python.client import timeline
from merge_tracing import *

class ISNNLayer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def input(self, inpt, mini_batch_size):
        pass

    @abstractmethod
    def cost(self, net):
        pass

    @abstractmethod
    def accuracy(self, y):
        pass

    @property
    @abstractmethod
    def y_out(self):
        pass

    @property
    @abstractmethod
    def output(self):
        # output in one-hot.
        pass

    @property
    @abstractmethod
    def parameters(self):
        # structure: [weights, biases]
        pass


class ISTFNN(object):

    def __init__(self, layers, mini_batch_size):
        self.layers = layers
        self.mbs = mini_batch_size
        self.x = tf.placeholder(dtype=tf.float32)
        self.y = tf.placeholder(dtype=tf.float32)

        self.layers[0].input(self.x, self.mbs)
        for p, n in zip(self.layers[1:], self.layers[:-1]):
            p.input(n.output, self.mbs)

        self.output = self.layers[-1].output
        self.steps = tf.Variable(0, tf.int32)

    def train(self, *, training_data, validation_data=None, test_data=None,
              eta, lambd=0.0, epochs=10, optimizer=tf.train.GradientDescentOptimizer):
        training_batches = np.size(training_data[0], 0) // self.mbs
        validation_batches = np.size(validation_data[0], 0) // self.mbs
        test_batches = np.size(test_data[0], 0) // self.mbs

        with tf.Session() as sess:
            # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # run_metadata = tf.RunMetadata()
            # many_runs_timeline = TimeLiner()
            cost = self.layers[-1].cost(self)
            # regularization
            L2 = tf.reduce_sum([tf.reduce_sum(layer.parameters[0]**2) for layer in self.layers])
            cost += 0.5*lambd/training_batches*L2
            opt = optimizer(eta)
            trainer = opt.minimize(cost, global_step=self.steps)
            accuracy = self.layers[-1].accuracy(self.y)
            tf.global_variables_initializer().run()

            for e in range(epochs):
                t = time.time()
                for b in range(training_batches):
                    data_x = training_data[0][b*self.mbs: (b+1)*self.mbs]
                    data_y = training_data[1][b*self.mbs: (b+1)*self.mbs]
                    cost_v, _ = sess.run([cost, trainer], feed_dict={self.x:data_x, self.y:data_y})
                    # tl = timeline.Timeline(run_metadata.step_stats)
                    # ctf = tl.generate_chrome_trace_format()
                    # many_runs_timeline.update_timeline(ctf)
                # print(sess.run([accuracy], feed_dict={self.x:training_data[0], self.y:training_data[1]}))
                # print(sess.run([accuracy], feed_dict={self.x:validation_data[0], self.y:validation_data[1]}))
                # print(sess.run([accuracy], feed_dict={self.x:test_data[0], self.y:test_data[1]}))
                print("epoch %s complete. cost %.2fs" % (e, time.time() - t))
                # many_runs_timeline.save('fd_timeline_03_merged_%d_runs.json' % e)
                res = []
                for b in range(training_batches):
                    res.append(sess.run([accuracy], feed_dict={
                        self.x:training_data[0][b*self.mbs: (b+1)*self.mbs],
                        self.y:training_data[1][b*self.mbs: (b+1)*self.mbs]}))
                print(np.mean(res))

                res = []
                for b in range(validation_batches):
                    res.append(sess.run([accuracy], feed_dict={
                        self.x:validation_data[0][b*self.mbs: (b+1)*self.mbs],
                        self.y:validation_data[1][b*self.mbs: (b+1)*self.mbs]}))
                print(np.mean(res))

                res = []
                for b in range(test_batches):
                    res.append(sess.run([accuracy], feed_dict={
                        self.x:test_data[0][b*self.mbs: (b+1)*self.mbs],
                        self.y:test_data[1][b*self.mbs: (b+1)*self.mbs]}))
                print(np.mean(res))

                print("epoch total cost %.2fs" % (time.time() - t))

class FullyConnectedLayer(ISNNLayer):

    def __init__(self, size_in, size_out, activation_fn=tf.nn.relu):
        self._size_in = size_in
        self._size_out = size_out
        self._activation_fn = activation_fn
        self._weights = tf.Variable(tf.truncated_normal(shape=(self._size_in, self._size_out),
                                                       stddev=1.0/self._size_in), dtype=tf.float32)
        self._biases = tf.Variable(0.1, self._size_out, dtype=tf.float32)

    def input(self, inpt, mini_batch_size):
        inpt = tf.reshape(inpt, [mini_batch_size, self._size_in])
        z = tf.matmul(inpt, self._weights) + self._biases
        self._output = self._activation_fn(z)
        self._y_out = tf.argmax(self._output, axis=1)

    def cost(self, net):
        ''' This cost function should only use with sigmoid function.'''
        v_y = tf.reshape(net.y, [1, net.mbs * self._size_out])
        # shape (n, 1)
        v_a = tf.transpose(tf.reshape(self._output, v_y.shape))
        return -( tf.matmul(v_y, tf.log(v_a)) + tf.matmul(1 - v_y, tf.log(1 - v_a)) ) / net.mbs

    def accuracy(self, y):
        return tf.reduce_mean(tf.cast(
                tf.equal(self._y_out, tf.argmax(y, axis=1)), dtype=tf.float32))

    @property
    def parameters(self):
        return [self._weights, self._biases]

    @property
    def output(self):
        return self._output

    @property
    def y_out(self):
        return self._y_out


class ConvolutionalLayer(ISNNLayer):
    def __init__(self, input_shape, filter_shape,
                 strides=(1, 1, 1, 1), pool_size=(1, 2, 2, 1), activation_fn=tf.nn.relu):
        '''
        :param input_shape: [batch_size, height, width, channels]
        :param filter_shape: [height, width, in_channels, out_channels]
        :param strides: same as input_shape, normally strides[0] = strides[3] = 1
        :param pool_size: same as input_shape normally pool_size[0] = pool_size[3] = 1
        '''
        self._input_shape = input_shape
        self._strides = strides
        self._filter_shape = filter_shape
        self._pool_size = pool_size
        self._activation_fn = activation_fn
        self._weights = tf.Variable(tf.truncated_normal(shape=self._filter_shape,
                                                        stddev=1.0/np.prod(filter_shape[0:2]),
                                                        dtype=tf.float32))
        self._biases = tf.Variable(0.1, self._filter_shape[-1], dtype=tf.float32)

    def input(self, inpt, mini_batch_size):
        inpt = tf.reshape(inpt, self._input_shape)
        conv = tf.nn.conv2d(inpt, self._weights, self._strides, 'VALID') + self._biases
        # strides is the same as kernel size.
        pool = tf.nn.max_pool(conv, ksize=self._pool_size, strides=self._pool_size, padding='VALID')
        self._output = self._activation_fn(pool)


    def cost(self, net):
        raise ArithmeticError

    def accuracy(self, y):
        raise ArithmeticError

    @property
    def y_out(self):
        raise ArithmeticError

    @property
    def output(self):
        return self._output

    @property
    def parameters(self):
        return [self._weights, self._biases]


class SoftmaxLayer(ISNNLayer):
    def __init__(self, size_in, size_out):
        self._size_in = size_in
        self._size_out = size_out
        self._weights = tf.Variable(tf.zeros(shape=(size_in, size_out), dtype=tf.float32))
        self._biases = tf.Variable(tf.zeros(shape=(size_out,), dtype=tf.float32))

    def input(self, inpt, mini_batch_size):
        inpt = tf.reshape(inpt, [mini_batch_size, self._size_in])
        z = tf.matmul(inpt, self._weights) + self._biases
        self._output = tf.nn.softmax(z)
        self._y_out = tf.argmax(self._output, axis=1)

    def cost(self, net):
        indices = tf.stack([tf.range(net.mbs, dtype=tf.int64), tf.argmax(net.y, axis=1)], -1)
        return -tf.reduce_mean(tf.log(
            tf.gather_nd(self._output, indices)))

    def accuracy(self, y):
        return tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(y, axis=1), self.y_out), dtype=tf.float32))

    @property
    def y_out(self):
        return self._y_out

    @property
    def output(self):
        return self._output

    @property
    def parameters(self):
        return [self._weights, self._biases]


tr, v, te = load_from_minst('mnist.pkl.gz', one_hot=10)

mbs = 1000
network = ISTFNN([ConvolutionalLayer([mbs, 28, 28, 1], [5, 5, 1, 10]),
                  FullyConnectedLayer(12*12*10, 100),
                  SoftmaxLayer(100, 10)], mbs)

network.train(training_data=tr, validation_data=v, test_data=te, eta=0.1, lambd=0.5, epochs=30)

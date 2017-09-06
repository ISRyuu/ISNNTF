# Spirit of Ryuu
# 2017/8/31
# Kevin

# Spirit of Ryuu
# 2017/8/15
# Kevin

import os, time, select, socket, signal
import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty
from convert_to_tfrecords import read_minst_from_tfrecords

_IS_MSG_INFORM = b'0'
_IS_MSG_SHUTDOWN = b'1'

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

    @abstractproperty
    def y_out(self):
        pass

    @abstractproperty
    def output(self):
        # output in one-hot.
        pass

    @abstractproperty
    def parameters(self):
        # structure: [weights, biases]
        pass


class ISTFNN(object):

    def __init__(self, layers, mini_batch_size, x, y, scopes):
        '''
        :param x: input queue
        :param y: input queue
        '''
        self.layers = layers
        self.mbs = mini_batch_size
        self.x = x
        self.y = y
        with tf.variable_scope(scopes[0]):
            self.layers[0].input(self.x, self.mbs)
        i = 1
        for p, n in zip(self.layers[1:], self.layers[:-1]):
            with tf.variable_scope(scopes[i]):
                p.input(n.output, self.mbs)
            i += 1

        self.output = self.layers[-1].output
        self.steps = tf.Variable(0, tf.int32)

    def train(self, *, eta, lambd=0.0, optimizer=tf.train.GradientDescentOptimizer,
              validation_process=False, period=100, validation_data=None):

        with tf.Session() as sess:
            pipe = []
            if validation_process:
                # TODO: validation process is not implemented.
                pipe[:] = os.pipe()
                signal.signal(signal.SIGCHLD, self.sigchld_handler)
                pid = os.fork()
                if pid < 0:
                    print("cannot fork")
                    exit()

                if pid == 0:

                    print(os.getpid())
                    while True:
                        r, *_ = select.select([pipe[0]], [], [])
                        msg = os.read(r[0], 1)

                        if len(msg) == 0:
                            break
                        if msg == _IS_MSG_INFORM:
                            print('finished')
                        if msg == _IS_MSG_SHUTDOWN:
                            break

                    os.close(pipe[0])
                    os.close(pipe[1])
                    exit()

                else: self._cpid = pid

            with tf.variable_scope('cost'):
                cost = self.layers[-1].cost(self)
                # L2 regularization
                L2 = tf.reduce_sum([tf.reduce_sum(layer.parameters[0]**2) for layer in self.layers])
                cost += 0.5*lambd/self.mbs*L2
                accuracy = self.layers[-1].accuracy(self.y)

            opt = optimizer(eta)
            trainer = opt.minimize(cost, global_step=self.steps)

            tf.summary.scalar('cost', cost)
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('train', sess.graph)

            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)

            coord = tf.train.Coordinator()
            thread = tf.train.start_queue_runners(sess=sess, coord=coord)
            t = 0
            step = 0
            try:
                step = 0
                t = time.time()

                while not coord.should_stop():
                    summary, _ = sess.run([merged, trainer])
                    train_writer.add_summary(summary, step)
                    if step % period == 0:
                        if validation_process:
                            os.write(pipe[1], _IS_MSG_INFORM)
                        print("cost %.2fs, steps: %s" % ((time.time() - t), step))
                        t = time.time()
                    step += 1
            except tf.errors.OutOfRangeError:
                print("cost %.2fs, steps: %s" % ((time.time() - t), step))
                print("complete")

            try:
                coord.request_stop()
                coord.join(thread)
            except (tf.errors.NotFoundError,) as e:
                print("file not found", e)


            sess.close()
            if pipe:
                os.close(pipe[0])
                os.close(pipe[1])

    def sigchld_handler(self, p1, p2):
        print('validation process is dead.')
        os.waitpid(self._cpid, 0)
        exit()


class FullyConnectedLayer(ISNNLayer):

    def __init__(self, size_in, size_out, activation_fn=tf.nn.relu):
        self._size_in = size_in
        self._size_out = size_out
        self._activation_fn = activation_fn
        self._weights = tf.get_variable('weights', initializer=tf.truncated_normal(shape=(self._size_in, self._size_out),
                                                       stddev=1.0/self._size_in), dtype=tf.float32)
        self._biases = tf.get_variable('biases', initializer=tf.constant_initializer(0.1),
                                       shape=self._size_out, dtype=tf.float32)

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
        self._weights = tf.get_variable('weights', initializer=tf.truncated_normal(shape=self._filter_shape,
                                                        stddev=1.0/np.prod(filter_shape[0:2]),
                                                        dtype=tf.float32))
        self._biases = tf.get_variable('biases', initializer=tf.constant_initializer(0.1),
                                       shape=self._filter_shape[-1], dtype=tf.float32)

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
        self._weights = tf.get_variable('weights', initializer=tf.zeros(shape=(size_in, size_out), dtype=tf.float32))
        self._biases = tf.get_variable('biases', initializer=tf.zeros(shape=(size_out,), dtype=tf.float32))

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


mbs = 100
epochs = 10

with tf.variable_scope('input'):
    filequeue = tf.train.string_input_producer(['MNIST_GZ/training.tfrecords.gz'], num_epochs=epochs)
    img, label = read_minst_from_tfrecords(filequeue, 784, one_hot=10)
    x, y = tf.train.shuffle_batch([img, label], batch_size=100, capacity=1000+3*mbs,
                              min_after_dequeue=1000, num_threads=2)

layers = []
with tf.variable_scope('conv'):
    layers.append(ConvolutionalLayer([mbs, 28, 28, 1], [5, 5, 1, 10]))

with tf.variable_scope('fully'):
    layers.append(FullyConnectedLayer(12*12*10, 100))

with tf.variable_scope('softmax'):
    layers.append(SoftmaxLayer(100, 10))

# add a slash to re enter scope. that's not a reliable but the only way.
network = ISTFNN(layers ,mbs, x, y, ['conv/', 'fully/', 'softmax/'])

network.train(eta=0.1, lambd=0.05, period=50000/100)

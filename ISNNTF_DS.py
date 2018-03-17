# Spirit of Ryuu
# 2017/9/3
# Kevin

# Spirit of Ryuu
# 2017/8/31
# Kevin

# Spirit of Ryuu
# 2017/8/15
# Kevin

import os, time, select, signal
import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod
from convert_to_tfrecords import parse_function_maker

from tensorflow.python.client import timeline
from merge_tracing import *

_IS_MSG_INFORM = b'0'
_IS_MSG_SHUTDOWN = b'1'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


class ISNNLayer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def input(self, inpt, mini_batch_size, keep_prob):
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

    def __init__(self, layers, mini_batch_size, parse_func, buffer_mbs=100):
        '''
        :param parse_func: use as the parameter of dataset.map(), the dataset
         structure should be like
         (img, label), and labels should be one-hot.

        :param scopes: scope name for each layer.
        '''
        self.scoped_layers = layers
        self.layers, self.scopes = zip(*self.scoped_layers)
        self.parse_func = parse_func
        self.mbs = mini_batch_size

        # dropout
        self.keep_prob = tf.placeholder_with_default(1.0, shape=(), name='dropout_keep_prob')

        with tf.variable_scope('input'):
            self.input_file_placeholder = tf.placeholder(dtype=tf.string, name='input_file')
            dataset = tf.data.TFRecordDataset(self.input_file_placeholder, compression_type='GZIP')
            # i.e.: Don't repeat.
            dataset = dataset.repeat(1)
            dataset = dataset.map(parse_func, num_parallel_calls=20)
            dataset = dataset.prefetch(buffer_mbs * self.mbs)
            # must set batch size before setting filter.
            dataset = dataset.shuffle(buffer_size=self.mbs*buffer_mbs)
            dataset = dataset.batch(self.mbs)
            # currently there is no option like "allow_smaller_final_batch" in tf.train.batch
            # using the filter is an alternative way.
            dataset = dataset.filter(lambda x, y: tf.equal(tf.shape(y)[0], self.mbs))
            self.iterator = dataset.make_initializable_iterator()
            data = self.iterator.get_next()
            self.x = data[0]
            self.y = data[1]
                
        with tf.variable_scope(self.scopes[0]):
            self.layers[0].input(self.x, self.mbs, self.keep_prob)
        i = 1
        for p, n in zip(self.layers[1:], self.layers[:-1]):
            with tf.variable_scope(self.scopes[i]):
                p.input(n.output, self.mbs, self.keep_prob)
            i += 1

        self.output = self.layers[-1].output
        self.y_out = self.layers[-1].y_out

    def train(self, *, eta, lambd=0.0, epochs=10, optimizer=tf.train.GradientDescentOptimizer,
              validation_file=None, training_file, test_file=None, period=100,
              saved_model_dir='saved_models', dropout=None):

        self.steps = tf.Variable(0, tf.int32, name='steps')

        pipe = []

        pipe[:] = os.pipe()
        signal.signal(signal.SIGCHLD, self.sigchld_handler)
        pid = os.fork()
        if pid < 0:
            print("cannot fork")
            exit()

        if pid == 0:
            # validation process
            print('validation process ', os.getpid())
            net = ISTFNN(self.scoped_layers, self.mbs, self.parse_func)
            vaccuracy = net.layers[-1].accuracy(net.y)

            writer_tra = tf.summary.FileWriter('accuracy_log/training_set')
            writer_val = tf.summary.FileWriter('accuracy_log/validation_set')
            writer_tes = tf.summary.FileWriter('accuracy_log/test_set')
            writers = [writer_tra, writer_val, writer_tes]

            saver = tf.train.Saver()

            with tf.Session(config=config) as vsess:
                epoch = 0
                while True:
                    # waiting for hyper-parameter updating.
                    rc = [pipe[0]]
                    r, *_ = select.select(rc, [], [])
                    msg = os.read(r[0], 1)

                    if len(msg) == 0 or msg == _IS_MSG_SHUTDOWN:
                        # training process closed the pipe. quit.
                        break

                    if msg == _IS_MSG_INFORM:
                        # hyper-parameter updated.
                        saver.restore(vsess, tf.train.latest_checkpoint(saved_model_dir))
                        files = [training_file, validation_file, test_file]
                        set = ['training', 'validation', 'test']
                        t = time.time()

                        for f, s, w in zip(files, set, writers):
                            if f:
                                vsess.run(net.iterator.initializer,
                                          feed_dict={net.input_file_placeholder: f})
                                ac = []
                                try:
                                    while True:
                                        ac.append(vsess.run(vaccuracy))
                                except tf.errors.OutOfRangeError:
                                    if ac:
                                        ac_mean = np.mean(ac)
                                        print("%s set accuracy: %.2f%%" % (s, ac_mean * 100))
                                        # ![https://stackoverflow.com/questions/43322131/tensorflow-summary-adding-a-variable-which-does-not-belong-to-computational-gra]
                                        manual_summary = tf.Summary()
                                        # numpy.item() to get <class 'float'>
                                        manual_summary.value.add(tag='accuracy', simple_value=ac_mean.item())
                                        w.add_summary(manual_summary, epoch)
                                        w.flush()
                                    else: print('size error, empty accuracy records.')
                        epoch += 1
                        print('validation cost %.2fs' % (time.time() - t))

            os.close(pipe[0])
            os.close(pipe[1])
            exit()

        # training process
        else: self._cpid = pid


        #training process
        with tf.variable_scope('IS_cost_scope'):
            cost = self.layers[-1].cost(self)
            # L2 regularization
            L2 = tf.reduce_sum([tf.reduce_sum(layer.parameters[0]**2) for layer in self.layers])
            cost += 0.5*lambd/self.mbs*L2
            accuracy = self.layers[-1].accuracy(self.y)

        opt = optimizer(eta)
        trainer = opt.minimize(cost, global_step=self.steps)

        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:

            # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # run_metadata = tf.RunMetadata()
            # many_runs_timeline = TimeLiner()

            # visualization data
            tf.summary.scalar('cost', cost)
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('training_log', sess.graph)

            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

            sess.run(init)

            steps = 0
            # training loop
            for epoch in range(epochs):
                t = time.time()
                try:
                    sess.run(self.iterator.initializer,
                             feed_dict={self.input_file_placeholder: training_file})

                    while True:
                        if dropout and dropout > 0 and dropout < 1.0:
                            feed_dict = {self.keep_prob: dropout}
                        else:
                            feed_dict = None
                        summary, _ = sess.run([merged, trainer], feed_dict=feed_dict)

                        train_writer.add_summary(summary, steps)
                        # tl = timeline.Timeline(run_metadata.step_stats)
                        # ctf = tl.generate_chrome_trace_format()
                        # many_runs_timeline.update_timeline(ctf)
                        if steps % period == 0 and steps != 0:
                            if validation_file:
                                # save the model
                                saver.save(sess, global_step=self.steps, 
                                          save_path=os.path.join(saved_model_dir, 'checkpoint'))
                                os.write(pipe[1], _IS_MSG_INFORM)
                                # many_runs_timeline.save('ds_timeline_03_merged_%d_runs.json' % steps)
                            print("steps: %s" % (steps,))
                        steps += 1

                except tf.errors.OutOfRangeError:
                    print("epoch %s complete, cost %.2fs, steps: %s" % (epoch, (time.time() - t), steps))

            print('training complete.')
            os.write(pipe[1], _IS_MSG_SHUTDOWN)

        if pipe:
            os.close(pipe[0])
            os.close(pipe[1])

    def sigchld_handler(self, p1, p2):
        print('validation process is dead, killing training process.')
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

    def input(self, inpt, mini_batch_size, keep_prob):
        inpt = tf.reshape(inpt, [mini_batch_size, self._size_in])
        z = tf.matmul(inpt, self._weights) + self._biases
        self._output = tf.nn.dropout(self._activation_fn(z), keep_prob)
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
        :param strides: same as input_shape, normally strides[0] = strides[4] = 1
        :param pool_size: same as input_shape normally pool_size[0] = pool_size[4] = 1
        '''
        self._input_shape = input_shape
        self._strides = strides
        self._filter_shape = filter_shape
        self._pool_size = pool_size
        self._activation_fn = activation_fn
        self._weights = tf.get_variable('weights', initializer=tf.truncated_normal(shape=self._filter_shape,
                                                        stddev=1.0/np.prod(filter_shape[:-1]),
                                                        dtype=tf.float32))
        self._biases = tf.get_variable('biases', initializer=tf.constant_initializer(0.1),
                                       shape=self._filter_shape[-1], dtype=tf.float32)

    def input(self, inpt, mini_batch_size, keep_prob):
        # convolutional layer doesn't need dropout.
        inpt = tf.reshape(inpt, self._input_shape)
        conv = tf.nn.conv2d(inpt, self._weights, self._strides, 'SAME') + self._biases
        # strides is the same as kernel size.
        if self._pool_size:
            pool = tf.nn.max_pool(conv, ksize=self._pool_size, strides=self._pool_size, padding='SAME')
            self._output = self._activation_fn(pool)
        else:
            self._output = self._activation_fn(conv)


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

    def input(self, inpt, mini_batch_size, keep_prob):
        inpt = tf.reshape(inpt, [mini_batch_size, self._size_in])
        z = tf.matmul(inpt, self._weights) + self._biases
        self._output = tf.nn.dropout(tf.nn.softmax(z), keep_prob)
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


if __name__ == '__main__':
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    mbs = 1000
    epochs = 30
    training_file = 'MNIST_GZ/cifar10.tfrecords.gz'
    validation_file = 'MNIST_GZ/cifar10_test.tfrecords.gz'
    #test_file = 'MNIST_GZ/test.tfrecords.gz'
    test_file = None
    img_shape = 32*32*3
    one_hot = 10

    layers = []
    with tf.variable_scope('conv'):
        layers.append([ConvolutionalLayer([mbs, 32, 32, 3], [5, 5, 3, 50]), "conv/"])

    # with tf.variable_scope('conv2'):
    #     layers.append(ConvolutionalLayer([mbs, 12, 12, 100], [3, 3, 100, 100]))

    with tf.variable_scope('fully'):
        layers.append([FullyConnectedLayer(16*16*50, 100), "fully/"])

    with tf.variable_scope('softmax'):
        layers.append([SoftmaxLayer(100, 10), "softmax/"])

    # add a slash to re enter scope. that's not a reliable but the only way.
    network = ISTFNN(layers, mbs, parse_function_maker(img_shape, one_hot))
    #                 ['conv/', 'conv2', 'fully/', 'softmax/'])

    network.train(eta=0.1, lambd=0, epochs=epochs, period=50000/mbs,
                  training_file=training_file, validation_file=validation_file, test_file=test_file)

import tensorflow as tf
import numpy as np
import os
import time
from ISNNTF_DS import ConvolutionalLayer
from ISNNTF_DS import FullyConnectedLayer
from ISNNTF_DS import ISTFNN
from TFRConverter import VOC_TFRecords
import tensorflow.contrib.slim as slim


def leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.1)


def YOLO_layers(mbs, inp):
    keep_prob = tf.placeholder_with_default(1.0, shape=(), name='dropout_keep_prob')
    with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                activation_fn=leaky_relu,
                weights_regularizer=slim.l2_regularizer(0.001),
                weights_initializer=tf.keras.initializers.he_normal()
            ):
        net = slim.conv2d(inp, 16, 3, scope='conv1')
        net = slim.max_pool2d(net, 2, padding='SAME', scope='pool1')

        net = slim.conv2d(net, 32, 3, scope='conv2')
        net = slim.max_pool2d(net, 2, padding='SAME', scope='pool2')

        net = slim.conv2d(net, 64, 3, scope='conv3')
        net = slim.max_pool2d(net, 2, padding='SAME', scope='pool3')

        net = slim.conv2d(net, 128, 3, scope='conv4')
        net = slim.max_pool2d(net, 2, padding='SAME', scope='pool4')

        net = slim.conv2d(net, 256, 3, scope='conv5')
        net = slim.max_pool2d(net, 2, padding='SAME', scope='pool5')

        net = slim.conv2d(net, 512, 3, scope='conv6')
        net = slim.max_pool2d(net, 2, padding='SAME', scope='pool6')

        net = slim.conv2d(net, 1024, 3, scope='conv7')
#        net = slim.conv2d(net, 256, 3, scope='conv_8')        
        net = slim.conv2d(net, 1024, 3, scope='conv_8')
        net = slim.conv2d(net, 1024, 3, scope='conv_9')
        net = slim.flatten(net, scope='flatten0')
        net = slim.fully_connected(net, 256, scope='fc1')
        net = slim.fully_connected(net, 4096, scope='fc2', activation_fn=None)

        net = slim.dropout(
             net, keep_prob=keep_prob,
             scope='dropout0')
        net = slim.fully_connected(
             net, 7*7*30, activation_fn=None, scope='fc3')
    return net, keep_prob

    layers = []

    with tf.variable_scope("conv1"):
        layers.append([
            ConvolutionalLayer(
                [mbs, 448, 448, 3], [3, 3, 3, 16], activation_fn=leaky_relu
            ),
            "conv1/"
        ])
    with tf.variable_scope("conv2"):
        layers.append([
            ConvolutionalLayer(
                [mbs, 224, 224, 16], [3, 3, 16, 32], activation_fn=leaky_relu
            ),
            "conv2/"
        ])
    with tf.variable_scope("conv3"):
        layers.append([
            ConvolutionalLayer(
                [mbs, 112, 112, 32], [3, 3, 32, 64], activation_fn=leaky_relu
            ),
            "conv3/"
        ])
    with tf.variable_scope("conv4"):
        layers.append([
            ConvolutionalLayer(
                [mbs, 56, 56, 64], [3, 3, 64, 128], activation_fn=leaky_relu
            ),
            "conv4/"
        ])
    with tf.variable_scope("conv5"):
        layers.append([
            ConvolutionalLayer(
                [mbs, 28, 28, 128], [3, 3, 128, 256], activation_fn=leaky_relu
            ),
            "conv5/"
        ])
    with tf.variable_scope("conv6"):
        layers.append([
            ConvolutionalLayer(
                [mbs, 14, 14, 256], [3, 3, 256, 512], activation_fn=leaky_relu
            ),
            "conv6/"
        ])
    with tf.variable_scope("conv7"):
        layers.append([
            ConvolutionalLayer(
                [mbs, 7, 7, 512], [3, 3, 512, 1024], pool_size=None,
                activation_fn=leaky_relu
            ),
            "conv7/"
        ])
    with tf.variable_scope("conv8"):
        layers.append([
            ConvolutionalLayer(
                [mbs, 7, 7, 1024], [3, 3, 1024, 1024], pool_size=None,
                activation_fn=leaky_relu
            ),
            "conv8/"
        ])
    with tf.variable_scope("conv9"):
        layers.append([
            ConvolutionalLayer(
                [mbs, 7, 7, 1024], [3, 3, 1024, 1024], pool_size=None,
                activation_fn=leaky_relu
            ),
            "conv9/"
        ])
    with tf.variable_scope("conn10"):
        layers.append([
            FullyConnectedLayer(
                7*7*1024, 256, activation_fn=None,
            ),
            "conn10/"
        ])
    with tf.variable_scope("conn11"):
        layers.append([
            FullyConnectedLayer(
                256, 4096, activation_fn=leaky_relu, keep_prob=keep_prob
            ),
            "conn11/"
        ])
    with tf.variable_scope("conn12"):
        layers.append([
            FullyConnectedLayer(
                4096, 7*7*30, activation_fn=None
            ),
            "conn12/"
        ])

    return layers, keep_prob


class MYYOLO(object):
    def __init__(self, img_size, mbs, classes, cell_size, pred_boxes):
        self.img_size = img_size
        self.classes = classes
        self.cell_size = cell_size
        self.predict_boxes = pred_boxes
        self.mbs = mbs
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.offset_y = np.reshape(
            np.asarray([np.arange(self.cell_size)]*self.cell_size*self.predict_boxes).T,
            (self.cell_size, self.cell_size, self.predict_boxes))

    def loss_layer(self, predictions, gbox):
        with tf.variable_scope("loss"):
            gbox = tf.reshape(gbox, [self.mbs, self.cell_size, self.cell_size, -1])

            label = gbox[..., :self.classes]

            # contain object or not
            confidence = tf.reshape(
                gbox[..., self.classes],
                [self.mbs, self.cell_size, self.cell_size, 1])

            # groud true boxes
            gtb = tf.reshape(
                gbox[..., self.classes+1:],
                [self.mbs, self.cell_size, self.cell_size, 1, 4]) / self.img_size

            p_labels = tf.reshape(
                predictions[:, :self.classes*self.cell_size*self.cell_size],
                [self.mbs, self.cell_size, self.cell_size, -1]
            )
            c_begin = self.cell_size * self.cell_size * self.classes
            c_end = self.cell_size * self.cell_size * (self.classes + self.predict_boxes)
            p_confidences = tf.reshape(
                predictions[:, c_begin:c_end],
                [self.mbs, self.cell_size, self.cell_size, self.predict_boxes])

            p_boxes = tf.reshape(
                predictions[..., c_end:],
                [self.mbs, self.cell_size, self.cell_size, self.predict_boxes, 4])

            # repeat gtb to fit predictions
            size_fitted_gtb = tf.tile(gtb, [1, 1, 1, self.predict_boxes, 1])

            offset_y = tf.expand_dims(
                tf.constant(self.offset_y, dtype=tf.float32), 0)

            offset_y = tf.tile(offset_y, [self.mbs, 1, 1, 1])
            offset_x = tf.transpose(offset_y, (0, 2, 1, 3))

            # convert x, y to values relative to the whole image
            # and square back w, h, predict sqrted w, h according
            # to original darknet implementation, for convenience.
            p_boxes_squared_offset = tf.stack(
                [(p_boxes[..., 0] + offset_x) / self.cell_size,
                 (p_boxes[..., 1] + offset_y) / self.cell_size,
                 tf.square(p_boxes[..., 2]),
                 tf.square(p_boxes[..., 3])],
                axis=-1)

            iou = self.calculate_IOU(p_boxes_squared_offset, size_fitted_gtb)
            responsible_iou = tf.reduce_max(iou, axis=-1, keepdims=True)
            responsible_mask = tf.cast(iou >= responsible_iou, tf.float32)

            object_responsible_mask = responsible_mask * confidence
            noobj_mask = tf.ones_like(object_responsible_mask) - \
                         object_responsible_mask

            # convert x, y to values relative to bounds of the grid cell
            boxes_offset_sqrted = tf.stack(
                [size_fitted_gtb[..., 0] * self.cell_size - offset_x,
                 size_fitted_gtb[..., 1] * self.cell_size - offset_y,
                 tf.sqrt(size_fitted_gtb[..., 2]),
                 tf.sqrt(size_fitted_gtb[..., 3])],
                axis=-1)

            loss_boxes = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(
                        tf.multiply(
                            p_boxes - boxes_offset_sqrted,
                            tf.expand_dims(object_responsible_mask, axis=-1)
                        )
                    ),
                    axis=[1, 2, 3, 4]
                )
            ) * self.lambda_coord

            loss_classes = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square((p_labels - label) * confidence),
                    axis=[1, 2, 3]
                )
            )

            # https://github.com/pjreddie/darknet/blob/master/src/detection_layer.c
            # line 166
            # It seems this is inconsistent with the loss function in paper.
            loss_obj_confidence = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(
                        tf.multiply(
                            iou - p_confidences,
                            object_responsible_mask
                        )
                    ),
                    axis=[1, 2, 3]
                )
            )

            loss_noobj_confidence = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(p_confidences * noobj_mask),
                    axis=[1, 2, 3]
                    )
                ) * self.lambda_noobj

            loss = loss_boxes + loss_classes + \
                   loss_obj_confidence + loss_noobj_confidence

            tf.summary.scalar('loss', loss)
            tf.summary.scalar('loss_boxes', loss_boxes)
            tf.summary.scalar('loss_classes', loss_classes)
            tf.summary.scalar('loss_obj_confidence', loss_obj_confidence)
            tf.summary.scalar('loss_noobj_confidence', loss_noobj_confidence)

            return loss

    def calculate_IOU(self, predictions, gtb):
        # convert boxes from [centerx, centery, w, h] to
        # [x_upper_left, y_upper_left, x_lower_right, y_lower_right]
        gtb_boxes_x_ul = gtb[..., 0] - gtb[..., 2] / 2
        gtb_boxes_y_ul = gtb[..., 1] - gtb[..., 3] / 2
        gtb_boxes_x_lr = gtb[..., 0] + gtb[..., 2] / 2
        gtb_boxes_y_lr = gtb[..., 1] + gtb[..., 3] / 2

        pred_boxes_x_ul = predictions[..., 0] - predictions[..., 2] / 2
        pred_boxes_y_ul = predictions[..., 1] - predictions[..., 3] / 2
        pred_boxes_x_lr = predictions[..., 0] + predictions[..., 2] / 2
        pred_boxes_y_lr = predictions[..., 1] + predictions[..., 3] / 2

        # stack points back to shape [mbs, cell, cell, boxes, *4]
        # *4 == [x_ul, y_ul, x_lr, y_lr]
        gtb_boxes = tf.stack([gtb_boxes_x_ul, gtb_boxes_y_ul,
                              gtb_boxes_x_lr, gtb_boxes_y_lr],
                             axis=-1)

        pred_boxes = tf.stack([pred_boxes_x_ul, pred_boxes_y_ul,
                               pred_boxes_x_lr, pred_boxes_y_lr],
                              axis=-1)

        # find upper left and lower right points of overlap
        # shape overlap_ul/lr == [mbs, cell, cell, boxes, 2]
        overlap_ul = tf.maximum(gtb_boxes[..., :2], pred_boxes[..., :2])
        overlap_lr = tf.minimum(gtb_boxes[..., 2:], pred_boxes[..., 2:])

        # area of overlap
        overlap_area = tf.reduce_prod(
            tf.maximum(0.0, tf.subtract(overlap_lr, overlap_ul)),
            axis=-1)

        # area of union
        union_area = tf.subtract(
            tf.add(
                tf.multiply(predictions[..., 2], predictions[..., 3]),
                tf.multiply(gtb[..., 2], gtb[..., 3])
            ),
            overlap_area)

        # avoid zero division error
        union_area = tf.maximum(union_area, 1e-10)

        # iou
        iou = tf.div(overlap_area, union_area)
        iou = tf.minimum(tf.maximum(iou, 0), 1)

        return iou


if __name__ == '__main__':
    mbs = 64
#    layers, keep_prob = YOLO_layers(mbs)
    parser = VOC_TFRecords.parse_function_maker([448, 448, 3], [7, 7, 25])
    net = ISTFNN([], mbs, parser, buffer_mbs=10)
    out, keep_prob = YOLO_layers(1, net.x)
    training_file = "voc2012.gz"
    test_file = "voc2007test.tfrecords.gz"

    global_steps = tf.Variable(0, tf.int32, name='steps')

    yolo = MYYOLO(448, mbs, 20, 7, 2)
    optimizer = tf.train.AdamOptimizer(0.0001)
    cost_test = yolo.loss_layer(out, net.y)
    cost = cost_test + tf.add_n(tf.losses.get_regularization_losses())
    tf.summary.scalar('loss_with_l2_reg', cost)
    trainer = optimizer.minimize(tf.reduce_sum(cost), global_step=global_steps)

    saver = tf.train.Saver()

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('training_log', graph=sess.graph)
        test_writer = tf.summary.FileWriter('test_log')

        sess.run(init)
        steps = 0
        test_steps = 0
        with open("output", "w") as outputfile:
            for epoch in range(100):
                sess.run(net.iterator.initializer,
                         feed_dict={net.input_file_placeholder: training_file})
                try:
                    while True:
                        last = time.time()
                        loss, summary, _ = sess.run([cost, merged, trainer], feed_dict={keep_prob: 0.5})
                        train_writer.add_summary(summary, steps)
                        steps += 1
                        print("cost: %f time: %f" % (loss, time.time() - last), file=outputfile)
                        outputfile.flush()
                except tf.errors.OutOfRangeError:
                    print(epoch, file=outputfile)
                    # if epoch == 99:
                    #     sess.run(net.iterator.initializer,
                    #              feed_dict={net.input_file_placeholder: training_file})
                    #     cost_t, outp, img, tbox = sess.run([cost, out, net.x, net.y])
                    #     print("cost t", cost_t)
                    #     np.save("save", outp)
                    #     np.save("img", img)
                    #     np.save("true", tbox)
                    #     exit(0)
                    # continue
                    if (epoch+1) % 30 == 0 or epoch == 99:
                        saver.save(sess, global_step=global_steps,
                                   save_path=os.path.join('model', 'checkpoint'))
                    losses = []
                    sess.run(net.iterator.initializer,
                             feed_dict={net.input_file_placeholder: test_file})
                    try:
                        start_time = time.time()
                        while True:
                            loss, summary = sess.run([cost_test, merged])
                            test_writer.add_summary(summary, test_steps)
                            test_steps += 1
                            print("test batch loss: %f" % loss, file=outputfile)
                            outputfile.flush()
                            losses += [loss]
                    except tf.errors.OutOfRangeError:
                            print("test loss: %f" % np.mean(losses), file=outputfile)
                            print("test evaluation time: %f" % (time.time() - start_time))


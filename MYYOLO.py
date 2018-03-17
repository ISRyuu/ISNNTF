import tensorflow as tf
import numpy as np
import time
from ISNNTF_DS import ConvolutionalLayer
from ISNNTF_DS import FullyConnectedLayer
from ISNNTF_DS import ISTFNN


def leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.1)


if __name__ == '__main__':
    mbs = 10
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
    with tf.variable_scope("conn8"):
        layers.append([
            FullyConnectedLayer(
                7*7*1024, 4096, activation_fn=leaky_relu
            ),
            "conn8/"
        ])
    with tf.variable_scope("conn9"):
        layers.append([
            FullyConnectedLayer(
                4096, 10
            ),
            "conn9/"
        ])

    # net = ISTFNN(layers, mbs, None)
    # init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    # with tf.Session() as sess:
    #     sess.run(init)
    #     last = time.time()
    #     res = sess.run([net.output], feed_dict={
    #         net.x: np.random.randn(mbs, 448, 448, 3)
    #     })


class MYYOLO(object):
    def __init__(self):
        self.classes
        self.cell_size
        self.predict_boxes
        self.mbs

    def loss_layer(self, predictions, gbox):
        label = gbox[..., :self.classes]
        confidence = tf.reshape(
            gbox[..., self.classes],
            [self.mbs, self.cell_size, self.cell_size, 1])

        gtb = tf.reshape(
            gbox[..., self.classes:],
            [self.mbs, self.cell_size, self.cell_size, 1, 4])

        p_label = predictions[..., :self.classes]
        p_confidences = tf.reshape(
            predictions[..., self.classes:self.classes+self.predict_boxes],
            [self.mbs, self.cell_size, self.cell_size, self.predict_boxes, 1])

        p_boxes = tf.reshape(
            predictions[..., self.classes+self.predict_boxes:],
            [self.mbs, self.cell_size, self.cell_size, self.predict_boxes, 4])

        # repeat gtb to fit predictions
        gtb = tf.tile(gtb, [1, 1, 1, self.predict_boxes, 1])
        iou = self.calculate_IOU(p_boxes, gtb)

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
            tf.maximum(0, tf.subtract(overlap_lr, overlap_ul)),
            axis=-1)

        # area of union
        union_area = tf.subtract(
            tf.add(
                tf.multiply(predictions[..., 2], predictions[..., 3]),
                tf.multiply(gtb[..., 2], gtb[..., 2])
            ),
            overlap_area)

        # avoid zero division error
        union_area = tf.maximum(union_area, 1e-10)

        # iou
        iou = tf.div(overlap_area, union_area)

        return iou

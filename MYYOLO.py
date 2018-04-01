import tensorflow as tf
import numpy as np
import time
from ISNNTF_DS import ConvolutionalLayer
from ISNNTF_DS import FullyConnectedLayer
from ISNNTF_DS import ISTFNN
from TFRConverter import VOC_TFRecords


def leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.1)


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

        predictions = tf.reshape(predictions, [self.mbs, self.cell_size, self.cell_size, -1])
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

        p_labels = predictions[..., :self.classes]
        p_confidences = tf.reshape(
            predictions[..., self.classes:self.classes+self.predict_boxes],
            [self.mbs, self.cell_size, self.cell_size, self.predict_boxes])

        p_boxes = tf.reshape(
            predictions[..., self.classes+self.predict_boxes:],
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
    mbs = 1
    layers = []
    layer_no = 1

    # layer 1
    with tf.variable_scope("conv%d" % layer_no):
        layers.append([
            ConvolutionalLayer(
                [mbs, 448, 448, 3],
                [7, 7, 3, 64],
                strides=[1, 2, 2, 1],
                activation_fn=leaky_relu
            ),
            "conv%d/" % layer_no
        ])
    layer_no += 1

    # layer 2
    with tf.variable_scope("conv%d" % layer_no):
        layers.append([
            ConvolutionalLayer(
                [mbs, 112, 112, 64],
                [3, 3, 64, 192],
                activation_fn=leaky_relu
            ),
            "conv%d/" % layer_no
        ])
    layer_no += 1

    # layer 3
    with tf.variable_scope("conv%d" % layer_no):
        layers.append([
            ConvolutionalLayer(
                [mbs, 56, 56, 192],
                [1, 1, 192, 128],
                pool_size=None,
                activation_fn=leaky_relu
            ),
            "conv%d/" % layer_no
        ])
    layer_no += 1

    # layer 4
    with tf.variable_scope("conv%d" % layer_no):
        layers.append([
            ConvolutionalLayer(
                [mbs, 56, 56, 128],
                [3, 3, 128, 256],
                pool_size=None,
                activation_fn=leaky_relu
            ),
            "conv%d/" % layer_no
        ])
    layer_no += 1

    # layer 5
    with tf.variable_scope("conv%d" % layer_no):
        layers.append([
            ConvolutionalLayer(
                [mbs, 56, 56, 256],
                [1, 1, 256, 256],
                pool_size=None,
                activation_fn=leaky_relu
            ),
            "conv%d/" % layer_no
        ])
    layer_no += 1

    # layer 6
    with tf.variable_scope("conv%d" % layer_no):
        layers.append([
            ConvolutionalLayer(
                [mbs, 56, 56, 256],
                [3, 3, 256, 512],
                activation_fn=leaky_relu
            ),
            "conv%d/" % layer_no
        ])
    layer_no += 1

    # layer 7..14
    for _ in range(4):
        with tf.variable_scope("conv%d" % layer_no):
            layers.append([
                ConvolutionalLayer(
                    [mbs, 28, 28, 512],
                    [1, 1, 512, 256],
                    pool_size=None,
                    activation_fn=leaky_relu
                ),
                "conv%d/" % layer_no
            ])
        layer_no += 1

        with tf.variable_scope("conv%d" % layer_no):
            layers.append([
                ConvolutionalLayer(
                    [mbs, 28, 28, 256],
                    [3, 3, 256, 512],
                    pool_size=None,
                    activation_fn=leaky_relu
                ),
                "conv%d/" % layer_no
            ])
        layer_no += 1

    # layer 15
    with tf.variable_scope("conv%d" % layer_no):
        layers.append([
            ConvolutionalLayer(
                [mbs, 28, 28, 512],
                [1, 1, 512, 512],
                pool_size=None,
                activation_fn=leaky_relu
            ),
            "conv%d/" % layer_no
        ])
    layer_no += 1

    # layer 16
    with tf.variable_scope("conv%d" % layer_no):
        layers.append([
            ConvolutionalLayer(
                [mbs, 28, 28, 512],
                [3, 3, 512, 1024],
                activation_fn=leaky_relu
            ),
            "conv%d/" % layer_no
        ])
    layer_no += 1

    # layer 17..20
    for _ in range(2):
        with tf.variable_scope("conv%d" % layer_no):
            layers.append([
                ConvolutionalLayer(
                    [mbs, 14, 14, 1024],
                    [1, 1, 1024, 512],
                    pool_size=None,
                    activation_fn=leaky_relu
                ),
                "conv%d/" % layer_no
            ])
        layer_no += 1

        with tf.variable_scope("conv%d" % layer_no):
            layers.append([
                ConvolutionalLayer(
                    [mbs, 14, 14, 512],
                    [3, 3, 512, 1024],
                    pool_size=None,
                    activation_fn=leaky_relu
                ),
                "conv%d/" % layer_no
            ])
        layer_no += 1

    # layer 21
    with tf.variable_scope("conv%d" % layer_no):
        layers.append([
            ConvolutionalLayer(
                [mbs, 14, 14, 1024],
                [3, 3, 1024, 1024],
                pool_size=None,
                activation_fn=leaky_relu
            ),
            "conv%d/" % layer_no
        ])
    layer_no += 1

    # layer 22
    with tf.variable_scope("conv%d" % layer_no):
        layers.append([
            ConvolutionalLayer(
                [mbs, 14, 14, 1024],
                [3, 3, 1024, 1024],
                strides=[1, 2, 2, 1],
                pool_size=None,
                activation_fn=leaky_relu,
            ),
            "conv%d/" % layer_no
        ])
    layer_no += 1

    # layer 23..24
    for _ in range(2):
        with tf.variable_scope("conv%d" % layer_no):
            layers.append([
                ConvolutionalLayer(
                    [mbs, 7, 7, 1024],
                    [3, 3, 1024, 1024],
                    pool_size=None,
                    activation_fn=leaky_relu
                ),
                "conv%d/" % layer_no
            ])
        layer_no += 1

    # layer 25
    with tf.variable_scope("conn%d" % layer_no):
        layers.append([
            FullyConnectedLayer(
                7*7*1024, 512, activation_fn=leaky_relu
            ),
            "conn%d/" % layer_no
        ])
    layer_no += 1

    # layer 26
    with tf.variable_scope("conn%d" % layer_no):
        layers.append([
            FullyConnectedLayer(
                512, 4096, activation_fn=leaky_relu
            ),
            "conn%d/" % layer_no
        ])
    layer_no += 1
    
    # layer 27
    with tf.variable_scope("conn%d" % layer_no):
        layers.append([
            FullyConnectedLayer(
                4096, 7*7*30, activation_fn=None, keep_prob=0.5
            ),
            "conn%d/" % layer_no
        ])

    parser = VOC_TFRecords.parse_function_maker([448, 448, 3], [7, 7, 25])
    net = ISTFNN(layers, mbs, parser, buffer_mbs=10)
    training_file = "voc2007.tfrecords.gz"
    test_file = "voc2007test.tfrecords.gz"

    saver = tf.train.Saver()
    yolo = MYYOLO(448, mbs, 20, 7, 2)
    optimizer = tf.train.AdamOptimizer(0.00001)
    cost = yolo.loss_layer(net.output, net.y)
    trainer = optimizer.minimize(cost)

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init)
        vs = tf.global_variables()
        w = np.load("weights.npy")
        for tensor, v in zip(vs, w):
            tensor.load(v, sess)
        
        for _ in range(100):
            sess.run(net.iterator.initializer,
                     feed_dict={net.input_file_placeholder: training_file})
            try:
                while True:
                    last = time.time()
                    loss, pred, out, _ = sess.run([cost, net.output, net.y, trainer])
                    print("cost: %f time: %f" % (loss, time.time() - last))

            except tf.errors.OutOfRangeError:
                loss = []
                sess.run(net.iterator.initializer,
                         feed_dict={net.input_file_placeholder: test_file})
                try:
                    start_time = time.time()
                    while True:
                        loss += sess.run([cost])
                except tf.errors.OutOfRangeError:
                        print("test loss: %f" % np.mean([loss]))
                        print("test evaluation time: %f" % (time.time() - start_time))

                        
if __name__ != '__main__':
    yolo = MYYOLO(448, 10, 20, 7, 2)
    pseudo_pred = tf.random_uniform([10, 7, 7, 30])
    pseudo_gtb = tf.random_uniform([10, 7, 7, 25])
    res = yolo.loss_layer(pseudo_pred, pseudo_gtb)
    sess = tf.Session()
    print(sess.run(res))

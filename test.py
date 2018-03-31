import numpy as np
import time


def iou(box_1, box_2):

    box_1_ulx = box_1[0] - box_1[2] * 0.5
    box_1_uly = box_1[1] - box_1[3] * 0.5
    box_1_lrx = box_1[0] + box_1[2] * 0.5
    box_1_lry = box_1[1] + box_1[3] * 0.5

    box_2_ulx = box_2[0] - box_2[2] * 0.5
    box_2_uly = box_2[1] - box_2[3] * 0.5
    box_2_lrx = box_2[0] + box_2[2] * 0.5
    box_2_lry = box_2[1] + box_2[3] * 0.5

    overlap_ulx = max(box_1_ulx, box_2_ulx)
    overlap_uly = max(box_1_uly, box_2_uly)
    overlap_lrx = min(box_1_lrx, box_2_lrx)
    overlap_lry = min(box_1_lry, box_2_lry)

    overlap = max(
        0,
        (overlap_lrx - overlap_ulx) * (overlap_lry - overlap_uly)
    )

    union = box_1[2] * box_1[3] + box_2[2] * box_2[3] - overlap

    return min(max(0, overlap / union), 1)


def non_max_suppression(output, cell_size, class_num, boxes_per_cell,
                        threshold=0.2, iou_threshold=0.5):
    '''output [cell_size, cell_size, boxes_per_cell, values]'''
    offset_y = np.reshape(
        np.asarray([np.arange(cell_size)]*cell_size*boxes_per_cell).T,
        (cell_size, cell_size, boxes_per_cell))

    offset_x = np.transpose(offset_y, [1, 0, 2])

    output = np.asarray(output)
    classes = np.reshape(output[..., :class_num],
                         [cell_size, cell_size, class_num])
    confidences = np.reshape(output[..., class_num:class_num+boxes_per_cell],
                             [cell_size, cell_size, boxes_per_cell])
    boxes = np.reshape(output[..., class_num+boxes_per_cell:],
                       [cell_size, cell_size, boxes_per_cell, -1])

    boxes[..., 0] = (boxes[..., 0] + offset_x) / cell_size
    boxes[..., 1] = (boxes[..., 1] + offset_y) / cell_size
    boxes[..., 2:] = np.square(boxes[..., 2:])

    class_confidences = []
    for i in range(boxes_per_cell):
        class_confidences += [np.expand_dims(confidences[..., i], axis=-1) * classes]
    class_confidences = np.stack(class_confidences, axis=-2)

    class_filter = class_confidences >= threshold
    class_filtered_indices = np.nonzero(class_filter)
    boxes_filtered = boxes[class_filtered_indices[0:3]]
    class_filtered = np.argmax(class_confidences, axis=-1)[class_filtered_indices[0:3]]
    probabilites_filtered = class_confidences[class_filter]

    sorted_probs_indices = np.flip(np.argsort(probabilites_filtered), axis=0)
    probabilites_filtered = probabilites_filtered[sorted_probs_indices]
    boxes_filtered = boxes_filtered[sorted_probs_indices]
    class_filtered = class_filtered[sorted_probs_indices]

    for i in range(len(sorted_probs_indices)):
        if probabilites_filtered[i] == 0:
            continue
        for j in range(i+1, len(sorted_probs_indices)):
            if iou(boxes_filtered[i], boxes_filtered[j]) >= iou_threshold:
                probabilites_filtered[j] = 0

    result_indices = probabilites_filtered > 0
    confidence_result = probabilites_filtered[result_indices]
    classes_result = class_filtered[result_indices]
    boxes_result = boxes_filtered[result_indices]

    return np.concatenate([np.expand_dims(confidence_result, axis=-1),
                           np.expand_dims(classes_result, axis=-1),
                           boxes_result],
                          axis=-1)


if __name__ == '__main__':
    test_data = np.abs(np.random.randn(7, 7, 30))
    non_max_suppression(test_data, 7, 20, 2)

    # confidences = np.random.randn(3,3,2)
    # classes = np.random.randn(3,3,20)
    # boxes_per_cell = 2

    # probs = np.zeros([3,3,2,20])
    # for i in range(boxes_per_cell):
    #         for j in range(20):
    #             probs[:, :, i, j] = np.multiply(
    #                 classes[:, :, j], confidences[:, :, i])

    # probabilites = []
    # for i in range(boxes_per_cell):
    #     probabilites += [np.expand_dims(confidences[..., i], axis=-1) * classes]

    # print(probs == np.stack(probabilites, axis=-2))

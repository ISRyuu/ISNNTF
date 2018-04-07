import os
import sys
import cv2
import numpy as np
import xml.etree.ElementTree as xmlET
import im_utilities
from TFRConverter import VOC_TFRecords


class DataUtility(object):
    def __init__(self, image_size, cell_size, class_number, label_to_index, data_dirs):
        self.image_size = image_size
        self.cell_size = cell_size
        self.class_number = class_number
        self.class_indices = label_to_index
        self.data_dirs = data_dirs

    def process(self, tf_output):
        with VOC_TFRecords(tf_output) as voc_tf:
#        if True:
            total = 0
            for data_dir in self.data_dirs:
                print("processing: %s" % data_dir)
                image_dir = os.path.join(data_dir, "JPEGImages")
                anno_dir = os.path.join(data_dir, "Annotations")
                count = 0
#                cv2.namedWindow("Test", cv2.WINDOW_AUTOSIZE)
#                cv2.namedWindow("Test2", cv2.WINDOW_AUTOSIZE)                
                for filename in os.listdir(image_dir):
                    if not filename.endswith(".jpg"):
                        continue
                    total += 1
                    index = os.path.splitext(filename)[0]
                    image_file = os.path.join(image_dir, filename)
                    anno_file = os.path.join(anno_dir, index + ".xml")
                    image = self.read_images(image_file)
                    annotations = self.load_annotations(anno_file)
                    count += 1
                    voc_tf.add_example(self.resize_RGB_img(image),
                                       self.process_boxes(annotations))
                    sys.stdout.write("%d %s\r" % (count, filename))

                    # data augmentation
                    image, annotations = self.augment(image, annotations)

                    count += 1                    
                    voc_tf.add_example(image, annotations)
                    sys.stdout.write("%d %s\r" % (count, filename))
                    if count >= 30000:
                        break
            print("%d images processed" % total)

    def process_boxes(self, boxes):
        # 5 : [size(contain obj or not) == 1, size(bbox) == 4]
        annotations = np.zeros((self.cell_size, self.cell_size, self.class_number + 5), dtype='float32')

        for box in boxes:
            label = box[-1]
            height = box[-2]
            width = box[-3]

            width_ratio = self.image_size / width
            height_ratio = self.image_size / height

            xmin = max(min((box[0] - 1) * width_ratio, self.image_size - 1), 0)
            xmax = max(min((box[1] - 1) * width_ratio, self.image_size - 1), 0)
            ymin = max(min((box[2] - 1) * height_ratio, self.image_size - 1), 0)
            ymax = max(min((box[3] - 1) * height_ratio, self.image_size - 1), 0)

            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            width_x = xmax - xmin
            width_y = ymax - ymin

            bbox = [center_x, center_y, width_x, width_y]

            x_cell_index = int(center_x / self.image_size * self.cell_size)
            y_cell_index = int(center_y / self.image_size * self.cell_size)
            class_index = self.class_indices[label]

            annotations[y_cell_index, x_cell_index][class_index] = 1
            annotations[y_cell_index, x_cell_index][self.class_number] = 1
            annotations[y_cell_index, x_cell_index][self.class_number+1:self.class_number+5] = bbox

        return annotations

    def resize_RGB_img(self, img):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size)).astype('uint8')
        return image

    def _fix(self, obj, dims, scale, offs):
        for i in range(4):
            obj[i] = obj[i]*scale - offs[i//2]
            obj[i] = max(0, min(obj[i], dims[i//2]))

    def augment(self, img, objs):
        im, dims, trans_param = im_utilities.imcv2_affine_trans(img)
        scale, offs, flip = trans_param
        for obj in objs:
            self._fix(obj, dims, scale, offs)
            if flip:
                width = dims[0]
                xmin = obj[0]
                obj[0] = width - obj[1]
                obj[1] = width - xmin

        im = im_utilities.imcv2_recolor(im)
        image = self.resize_RGB_img(im)
        annotations = self.process_boxes(objs)
        return image, annotations

    def read_images(self, img_path):
        img = cv2.imread(img_path, 1)
        try:
            if not img.any():
                raise IOError()
        except Exception as e:
            print("error reading image", img_path)
            raise e
        return img

    def load_annotations(self, xml_path):
        xml_obj = xmlET.parse(xml_path)
        size = xml_obj.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)

        annotations = []
        for obj in xml_obj.findall('object'):
            label = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            xmax = float(bndbox.find('xmax').text)
            ymin = float(bndbox.find('ymin').text)
            ymax = float(bndbox.find('ymax').text)

            annotations += [[xmin, xmax, ymin, ymax, width, height, label]]

        return annotations


if __name__ == "__main__":
    data_dir = sys.argv[1:-1]
    output = sys.argv[-1]
    CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']
    label_to_index = dict(zip(CLASSES, range(len(CLASSES))))
    utl = DataUtility(448, 7, 20, label_to_index, data_dir)
    utl.process(output)

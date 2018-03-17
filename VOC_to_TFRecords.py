import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as xmlET
from TFRConverter import VOC_TFRecords

class DataUtility(object):
    def __init__(self, image_size, cell_size, class_number, label_to_index, data_dir):
        self.image_size = image_size
        self.cell_size = cell_size
        self.class_number = class_number
        self.class_indices = label_to_index
        self.data_dir = data_dir

    def process(self, tf_output):
        with VOC_TFRecords(tf_output) as voc_tf:
            image_dir = os.path.join(self.data_dir, "JPEGImages")
            anno_dir = os.path.join(self.data_dir, "Annotations")
            count = 0
            cv2.namedWindow("Test", cv2.WINDOW_AUTOSIZE)
            for filename in os.listdir(image_dir):
                if not filename.endswith(".jpg"):
                    continue
                index = os.path.splitext(filename)[0]
                image_file = os.path.join(image_dir, filename)
                anno_file = os.path.join(anno_dir, index + ".xml")
                image = self.read_images(image_file)
                annotations = self.load_annotations(anno_file)
                count += 1
                voc_tf.add_example(image, annotations)
                sys.stdout.write("%d %s\r" % (count, filename))

    def read_images(self, img_path):
        img = cv2.imread(img_path, 1)
        try:
            if not img.any():
                raise IOError()
        except Exception as e:
            print("error reading image", img_path)
            raise e
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size)).astype('uint8')
        return img

    def load_annotations(self, xml_path):
        xml_obj = xmlET.parse(xml_path)
        size = xml_obj.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        width_ratio = self.image_size / width
        height_ratio = self.image_size / height

        # 5 : [size(contain obj or not) == 1, size(bbox) == 4]
        annotations = np.zeros((self.cell_size, self.cell_size, self.class_number + 5))

        for obj in xml_obj.findall('object'):
            label = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = max(min((float(bndbox.find('xmin').text) - 1) * width_ratio, self.image_size - 1), 0)
            xmax = max(min((float(bndbox.find('xmax').text) - 1) * width_ratio, self.image_size - 1), 0)
            ymin = max(min((float(bndbox.find('ymin').text) - 1) * height_ratio, self.image_size - 1), 0)
            ymax = max(min((float(bndbox.find('ymax').text) - 1) * height_ratio, self.image_size - 1), 0)

            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            width_x = xmax - xmin
            width_y = ymax - ymin

            bbox = [center_x, center_y, width_x, width_y]

            x_cell_index = int(center_x / self.image_size * self.cell_size)
            y_cell_index = int(center_y / self.image_size * self.cell_size)
            class_index = self.class_indices[label]
            
            annotations[x_cell_index, y_cell_index][0] = 1
            annotations[x_cell_index, y_cell_index][1:5] = bbox
            annotations[x_cell_index, y_cell_index][5 + class_index] = 1

        return annotations
    

if __name__ == "__main__":
    data_dir = sys.argv[1]
    output = sys.argv[2]
    CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']
    label_to_index = dict(zip(CLASSES, range(len(CLASSES))))
    utl = DataUtility(448, 7, 20, label_to_index, data_dir)
    utl.process(output)

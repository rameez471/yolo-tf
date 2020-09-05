import tensorflow as tf
import numpy as np

from model.yolo import anchors_wh

class Preprocessor(object):

    def __init__(self,is_train,num_classes,output_shape=(416,416)):
        self.is_train = is_train
        self.num_classes = num_classes
        self.output_shape = output_shape

    def __call__(self,example):

        features = self.parse_tfexample(example)

        encoded = features['image/encoded']
        image = tf.io.decode_jpeg(encoded)
        image = tf.cast(image, tf.float32)

        classes,bboxes = self.parse_y_features(features)

        image = tf.image.resize(image,self.output_shape)
        image = tf.cast(image, tf.float32) / 127.5 -1

        label = (
            self.preprocess_label_for_one_scale(classes,bboxes,52,np.array([0,1,2])),
            self.preprocess_label_for_one_scale(classes,bboxes,26,np.array([3,4,5])),
            self.preprocess_label_for_one_scale(classes,bboxes,13,np.array([6,7,8])),
        )

        return image, label
    
    def parse_y_features(self, features):

        classes = tf.sparse.to_dense(features['image/object/class/label'])
        classes = tf.one_hot(classes,self.num_classes)

        bboxes = tf.stack([
            tf.sparse.to_dense(features['images/object/bboxes/xmin']),
            tf.sparse.to_dense(features['images/object/bboxes/ymin']),
            tf.sparse.to_dense(features['images/object/bboxes/xmax']),
            tf.sparse.to_dense(features['images/object/bboxes/ymax'])
        ],axis=1)

        return classes,bboxes

    def preprocess_label_for_one_scale(self,classes,bboxes,grid_size=13,valid_anchors=None):

        y = tf.zeros((grid_size,grid_size,3,5+self.num_classes))
        anchors_indices = self.find_best_anchors(bboxes)

        tf.Assert(classes.shape[0] == bboxes.shape[0],[classes])
        tf.Assert(anchors_indices.shape[0] == bboxes.shape[0],[anchors_indices])

        num_boxes = tf.shape(classes)[0]

        indices = tf.TensorArray(tf.int32,1,dynamic_size=True)
        updates = tf.TensorArray(tf.int32,1,dynamic_size=True)

        valid_count = 0
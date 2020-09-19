"""
Train the Yolo for your dataset
"""

import os
import argparse
import sys
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import settings
from model.yolo import preprocess_true_boxes, yolo_body, yolo_loss
from model.utils import get_random_data

def _main(FLAGS):
    annotation_path = FLAGS.annotation
    log_dir = settings.LOGS_DIR
    classes_path = FLAGS.classes
    anchors_path = settings.DEFAULT_ANCHORS_PATH
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = settings.MODEL_IMAGE_SIZE

    model = create_model(input_shape, anchors, num_classes,
                        freeze_body=2,
                         weights_path=settings.PRETRAINED_WEIGHT)
    
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                monitor='val_loss',save_weights_only=True,save_best_only=True,period=3)
    reduce_lr  = ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=3,verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss',min_delta=0,patience=10,verbose=1)

    val_split = settings.VALID_SPLIT

    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(0)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    for i in range(len(model.layers)):
        model.layers[i].trainable = True
        
    model.compile(optimizer=Adam(lr=1e-4),loss={
        'yolo_loss':lambda y_true, y_pred: y_pred
    })

    batch_size = settings.UNFREEZE_TRAIN_BATCH_SIZE
    print('Train on {} samples,val on {} samples, with batch size {}'.format(num_train,num_val,batch_size))
    model.fit_generator(data_generator_wrapper(lines[:num_train],batch_size,input_shape,anchors,
                        num_classes),
                        steps_per_epoch=max(1,num_train // batch_size),
                        validation_data=data_generator_wrapper(lines[num_train:],batch_size,input_shape,anchors,num_classes),
                        validation_steps=max(1,num_val // batch_size),
                        epochs=100,
                        initial_epochs=50,
                        callbacks=[logging, checkpoint, reduce_lr, early_stopping])
    
    model.save_weights(os.path.join(log_dir, settings.UNFREEZE_TRAIN_OUTPUT_WEIGHTS))

def get_classes(classes_path):
    '''load the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''Load the anchors'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def create_model(input_shape, anchors, num_classes,load_pretrained=True,freeze_body=2,weights_path='model_data/yolo_weights.h5'):
    '''Create training model'''
    K.clear_session()
    image_input = Input(shape=(None,None,3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32,1:16,2:8}[l], w//{0:32 ,1:16, 2:8}[l],\
                num_anchors//3, num_classes + 5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Yolo modle with {} anchors and {} classes'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True,)
        print('Load weights {}'.format(weights_path))
        if freeze_body in [1,2]:
            num = (185, len(model_body.layers)-3)[freeze_body - 1]
            for i in range(num):
                model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {}'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,),name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_threshold':0.5})(
                            [*model_body.outputs, *y_true])

    model = Model([model_body.input, *y_true],model_loss)

    return model

def data_generator(annotation, batch_size, input_shape, anchors, num_classes):
    '''Data generator for fit generator'''
    n = len(annotation)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation)
            image, box = get_random_data(annotation[i],input_shape,random=settings.IMAGE_ARGMENTATION)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data,input_shape,anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation, batch_size, input_shape, anchors, num_classes):
    n = len(annotation)
    if n==0 or batch_size <= 0: return None
    return data_generator(annotation, batch_size, input_shape, anchors, num_classes)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument(
        '--annotation',type=str,
        help='Path to annotation file'
    )

    parser.add_argument(
        '--classes',type=str,
        help='Path to class name file'
    )

    FLAGS = parser.parse_args()

    _main(FLAGS)
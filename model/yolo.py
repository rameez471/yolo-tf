""" YOLO v3 Model defined in Keras"""
from functools import wraps

import tensorflow as tf 
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.layers import  (
    Input,
    Conv2D,
    Add,
    Concatenate,
    Lambda,
    MaxPool2D,
    BatchNormalization,
    LeakyReLU,
    UpSampling2D,
    Lambda,
    ZeroPadding2D
    )
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from model.utils import compose,xywh_to_x1x2y1y2, xywh_to_y1x1y2x2, broadcast_iou, binary_crossentropy

anchors_wh = np.array([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                       [59, 119], [116, 90], [156, 198], [373, 326]],
                      np.float32) / 416


@wraps(Conv2D)
def DarknetConv2D(*args,**kwargs):
    '''Wrapper to set Darknet parameters for Convolution Layer'''
    darknet_conv_kwargs = {'kernel_regularizer':l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args,**darknet_conv_kwargs)

def DarknetConv2D_BN_Leaky(*args,**kwargs):
    """Darknet Convolution followed by Batch Normalization and LeakyReLU"""
    no_bias_kwargs = {'use_bias':False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
    )

def resblock_body(x,filters,num_blocks):
    """Residual blocks for Darknet"""
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(filters,(3,3),strides=(2,2))(x)
    for _ in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Leaky(filters//2,(1,1)),
                DarknetConv2D_BN_Leaky(filters,(3,3)))(x)
        x = Add()([x,y])
    return x

def darknet_body(x):
    '''Darknet body having 52 Convolutional layers'''
    x = DarknetConv2D_BN_Leaky(32,(3,3))(x)
    x = resblock_body(x,64,1)
    x = resblock_body(x,128,2)
    x = resblock_body(x,256,8)
    x = resblock_body(x,512,8)
    x = resblock_body(x,1024,4)
    return x

def make_last_layers(x,filters,out_filters):
    """"6 Convolution Leaky_RELU Layers followed by a linear Convolution layer"""
    x = compose(
        DarknetConv2D_BN_Leaky(filters,(1,1)),
        DarknetConv2D_BN_Leaky(filters * 2,(3,3)),
        DarknetConv2D_BN_Leaky(filters,(1,1)),
        DarknetConv2D_BN_Leaky(filters * 2,(3,3)),
        DarknetConv2D_BN_Leaky(filters,(1,1)))(x)
    y = compose(
        DarknetConv2D_BN_Leaky(filter * 2,(3,3)),
        DarknetConv2D(out_filters,(1,1))(x)
    ) 
    return x,y

def yolo_body(inputs,num_anchors,num_classes):
    """Create YOLOv3 model"""
    darknet = Model(inputs,darknet_body(inputs))
    x,y1 = make_last_layers(darknet.output,512,num_anchors * (num_classes + 5))

    x = compose(
            DarknetConv2D_BN_Leaky(256,(1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[152].output])
    x,y2 = make_last_layers(x,256,num_anchors * (num_classes + 5))

    x = compose(
            DarknetConv2D_BN_Leaky(128,(1,1)),
            UpSampling2D(2))(x)
    x,y3 = make_last_layers(x,128,num_anchors * (num_classes + 5))

    return Model(inputs,[y1,y2,y2])


def yolo_head(feats,anchors,num_classes,input_shape,calc_loss=False):
    """Convert final predictions into bounding boxes"""
    num_anchors = len(anchors)
    # (batch, height, width, num_anchors, box_prams)
    anchor_tensor = K.reshape(K.constant(anchors),[1,1,1,num_anchors,2])

    grid_shape = K.shape(feats)[1:3] #(height,width)
    grid_y = K.tile(K.reshape(K.arrange(0,stop=grid_shape[0]),[-1,1,1,1]),
                    [1,grid_shape[1],1,1])
    grid_x = K.tile(K.reshape(K.arrange(0,stop=grid_shape[1]),[1,-1,1,1]),
                    [grid_shape[0],1,1,1])
    grid = K.concatenate([grid_x,grid_y])
    grid = K.cast(grid,K.dtype(feats))

    feats = K.reshape(
                feats,[-1,grid.shape[0],grid.shape[1],num_anchors,num_classes+5])

    box_xy = (K.sigmoid(feats[...,:2])+grid) / K.cast(grid_shape[::-1],K.dtype(feats))
    box_wh = K.exp(feats[...,2:4]) * anchor_tensor / K.cast(input_shape[::-1],K.dtype(feats))
    box_confidence = K.sigmoid(feats[...,4:5])
    box_class_probs = K.sigmoid(feats[...,5:])

    if calc_loss:
        return grid,feats,box_xy,box_wh

    return box_xy,box_wh,box_confidence,box_class_probs

def yolo_correct_boxes(box_xy,box_wh,input_shape,image_shape):
    """Get Correct boxes"""
    box_yx = box_xy[...,::-1]
    box_hw = box_wh[...,::-1]
    input_shape = K.cast(input_shape,K.dtype(box_yx))
    image_shape = K.cast(image_shape,K.dtype(box_hw))
    new_shape = K.round(input_shape * K.min(input_shape / image_shape))
    offset = (input_shape - new_shape ) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[...,0:1],  #y min
        box_mins[...,1:2],  #x min
        box_maxes[...,0:1], #y max
        box_maxes[...,1:2]  # x max
    ])

    boxes *= K.concatenate([image_shape,image_shape])
    return boxes

def yolo_boxes_and_scores(feats,anchors,num_classes,input_shape,image_shape):
    '''Process Convolutional output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
                        anchors,num_classes,input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, image_shape,image_shape)
    boxes = K.reshape(boxes,[-1,4])
    boxes_scores = box_confidence * box_class_probs
    boxes_scores = K.reshape(boxes_scores,[-1,num_classes])

    return boxes,boxes_scores

def yolo_eval(yolo_outputs,anchors,num_classes,image_shape,
             max_boxes=20,score_threshold=0.6,iou_threshold=0.5):
    """Evaluate Yolo model on inout and return filters boxes"""
    num_layers = len(yolo_outputs)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
                                                    anchors[anchor_mask[l]],num_classes,input_shape,image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    
    boxes = K.concatenate(boxes,axis=0)
    box_scores = K.concatenate(box_scores,axis=0)

    mask = box_scores >= score_threshold
    max_box_tensor = K.constant(max_boxes,dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ =[]
    for c in range(num_classes):
        class_boxes = tf.boolean_mask(boxes,mask[:,c])
        class_box_scores = tf.boolean_mask(box_scores[:,c],mask[:,c])
        nms_index = tf.image.non_max_suppression(
                    class_boxes,class_box_scores,max_box_tensor,iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes,nms_index)
        class_box_scores = K.gather(class_box_scores,nms_index)
        classes = K.ones_like(class_boxes,'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_,axis=0)
    scores_ = K.concatenate(scores_,axis=0)
    classes_ = K.concatenate(classes_,axis=0)

    return boxes_, scores_, classes_
        

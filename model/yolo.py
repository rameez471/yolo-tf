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
from model.utils import compose

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
        LeakyReLU(alpha=0.1))


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
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
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
        DarknetConv2D_BN_Leaky(filters * 2,(3,3)),
        DarknetConv2D(out_filters,(1,1)))(x) 
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
    x = Concatenate()([x,darknet.layers[92].output])
    x,y3 = make_last_layers(x,128,num_anchors * (num_classes + 5))

    return Model(inputs, [y1, y2, y3])


def yolo_head(feats,anchors,num_classes,input_shape,calc_loss=False):
    """Convert final predictions into bounding boxes"""
    num_anchors = len(anchors)
    # (batch, height, width, num_anchors, box_prams)
    anchor_tensor = K.reshape(K.constant(anchors),[1,1,1,num_anchors,2])

    grid_shape = K.shape(feats)[1:3] #(height,width)
    grid_y = K.tile(K.reshape(K.arange(0,stop=grid_shape[0]),[-1,1,1,1]),
                    [1,grid_shape[1],1,1])
    grid_x = K.tile(K.reshape(K.arange(0,stop=grid_shape[1]),[1,-1,1,1]),
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

    return box_xy, box_wh, box_confidence, box_class_probs

def yolo_correct_boxes(box_xy,box_wh,input_shape,image_shape):
    """Get Correct boxes"""
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes

def yolo_boxes_and_scores(feats,anchors,num_classes,input_shape,image_shape):
    '''Process Convolutional output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
                                                                anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores

def yolo_eval(yolo_outputs,anchors,num_classes,image_shape,
             max_boxes=20,score_threshold=0.6,iou_threshold=0.5):
    """Evaluate Yolo model on inout and return filters boxes"""
    num_layers = len(yolo_outputs)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]  # default setting
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
                                                    anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_
        
def preprocess_true_boxes(true_boxes,input_shape,anchors,num_classes):
    '''Preprocess true boxes to train input'''
    assert (true_boxes[...,4] < num_classes).all(), 'Class id must be less than num_classes'
    num_layers = len(anchors) // 3
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[...,0:2] + true_boxes[...,2:4]) // 2
    boxes_wh = true_boxes[...,2:4] - true_boxes[...,0:2]
    true_boxes[...,0:2] = boxes_xy / input_shape[::-1]
    true_boxes[...,2:4] = boxes_wh / input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape // {0:32, 1:16, 2:8}[l] for l in range(num_layers)]
    y_true = [np.zeros((m,grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
                                            dtype='float32') for l in range(num_layers)]

    #Expand dimension by broadcasting
    anchors = np.expand_dims(anchors, 0)
    anchors_maxes = anchors / 2.
    anchors_mins = -anchors_maxes
    valid_mask = boxes_wh[...,0] > 0

    for b in range(m):
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0:
            continue
        wh = np.expand_dims(wh, -2)
        box_maxes = wh/2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchors_mins)
        intersect_maxes = np.minimum(box_maxes, anchors_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[...,0] * intersect_wh[...,1]
        box_area = wh[...,0] * wh[...,1]
        anchor_area = anchors[...,0] * anchors[...,1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)
        
        best_anchor = np.argmax(iou, axis=-1)

        for t,n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, t,1] * grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b,t,4].astype('int32')
                    y_true[l][b,j,i,k,0:4] = true_boxes[b,t,0:4]
                    y_true[l][b,j,i,k,4] = 1
                    y_true[l][b,j,i,k, 5+c] = 1

    return y_true

def box_iou(b1, b2):
    '''Return iou tensor'''
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[...,:2]
    b1_wh = b1[...,2:4]
    b1_mins = b1_xy - b1_wh/2.
    b1_maxes = b1_xy + b1_wh/2.

    b2 = K.expand_dims(b2,0)
    b2_xy = b2[...,:2]
    b2_wh = b2[...,2:4]
    b2_mins = b2_xy - b2_wh/2.
    b2_maxes = b2_xy + b2_wh/2.

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0)
    intersect_area = intersect_wh[...,0] * intersect_wh[...,1]
    b1_area = b1_wh[...,0] * b1_wh[...,1]
    b2_area = b2_wh[...,0] * b2_wh[...,1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou

def yolo_loss(args, anchors, num_classes, ignore_threshold=0.5, print_loss=True):
    '''Return yolo loss tensor'''
    num_layers = len(anchors) // 3
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32,K.dtype(y_true[0]))
    grid_shape = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    m = K.shape(yolo_outputs[0])[0]
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        object_mask = y_true[l][...,4:5]
        true_class_probs = y_true[l][...,5:]

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
                                            anchors[anchor_mask[l]],num_classes,input_shape,calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        raw_true_xy = y_true[l][...,:2] * grid_shape[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][...,2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))
        box_loss_scale = 2 - y_true[l][...,2:3] * y_true[l][...,3:4]

        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask,'bool')

        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_threshold, K.dtype(true_box)))
            return b+1, ignore_mask

        _, ignore_mask = tf.while_loop(lambda b, *args: b<m, loop_body ,[0, ignore_mask]) 
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2],
                                                                        from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[...,2:4])
        confidence_loss = object_mask * K.binary_crossentropy(object_mask,raw_pred[...,4:5], from_logits=True) + \
                            (1-object_mask) * K.binary_crossentropy(object_mask,raw_pred[...,4:5],from_logits=True) * ignore_mask

        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:],from_logits=True)

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            print('Loss: ',loss)

    return loss


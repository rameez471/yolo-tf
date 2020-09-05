import tensorflow as tf 
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
    Upsampling2D,
    Lambda,
    )
from tensorflow.keras.models import Model

anchors_wh = np.array([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                       [59, 119], [116, 90], [156, 198], [373, 326]],
                      np.float32) / 416

def  YoloConv(inputs,filters,kernel_size,strides):
    x = Conv2D(filters=filters,kernel_size=kernel_size,
                strides=strides,padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    return x

def YoloResidual(inputs,filters):
    shortcut = inputs
    x = YoloConv(inputs,filters=filters,kernel_size=1,strides=1)
    x = YoloConv(x,filters=2*filters,kernel_size=3,strides=1)
    x = Add()([shortcut,x])

    return x


def Darknet(shape=(256,256,3)):
    inputs = Input(shape=shape)

    x = YoloConv(inputs,filters=32,kernel_size=3,strides=1)
    x = YoloConv(x,filters=64,kernel_size=3,strides=2)
    ## 1 Residual
    for _ in range(1):
        x = YoloResidual(x,filters=32)

    x = YoloConv(x,filters=128,kernel_size=3,strides=2)
    ## 2 Residual
    for _ in range(2):
        x = YoloResidual(x,filters=64)

    x = YoloConv(x,filters=256,kernel_size=3,strides=2)
    ## 8 Residual
    for _ in range(8):
        x = YoloResidual(x,filters=128)
    y0 = x

    x = YoloConv(x,filters=512,kernel_size=3,strides=2)
    ## 8 Residual
    for _ in range(8):
        x = YoloResidual(x,filters=256)
    y1 = x

    x = YoloConv(x,filters=1024,kernel_size=3,strides=2)
    ## 4 Residual
    for _ in range(4):
        x = YoloResidual(x,filters=512)
    y2 = x

    return Model(inputs,(y0,y1,y1),name='darknet_53')

def YoloV3(shape=(416,416,3),num_classes=2,training=False):

    final_filters = 3*(4+1+num_classes)

    inputs = Input(shape=shape)

    backbone = Darknet(shape)
    x_small,x_medium,x_large = backbone(inputs)
    ## Large Scale Detection
    x = YoloConv(x_large,filters=512,kernel_size=1,strides=1)
    x = YoloConv(x,filters=1024,kernel_size=3,strides=1)
    x = YoloConv(x,filters=512,kernel_size=1,strides=1)
    x = YoloConv(x,filters=1024,kernel_size=3,strides=1)
    x = YoloConv(x,filters=512,kernel_size=1,strides=1)

    y_large = YoloConv(x,filters=1024,kernel_size=3,strides=1)
    y_large = Conv2D(filters=final_filters,kernel_size=1,strides=1,padding='same')(y_large)

    ## Medium Scale Detection

    x = YoloConv(x,filters=256,kernel_size=3,strides=1)
    x = Upsampling2D(size=(2,2))(x)
    x = Concatenate()([x,x_medium])
    x = YoloConv(x,filters=256,kernel_size=1,strides=1)
    x = YoloConv(x,filters=512,kernel_size=3,strides=1)
    x = YoloConv(x,filters=256,kernel_size=1,strides=1)
    x = YoloConv(x,filters=512,kernel_size=3,strides=1)
    x = YoloConv(x,filters=256,kernel_size=1,strides=1)

    y_medium = YoloConv(x,filters=512,kernel_size=3,strides=1)
    y_medium = Conv2D(filters=final_filters,kernel_size=1,strides=1,padding='same')(y_medium)

    ## Small Scale Detection

    x = YoloConv(x,filters=128,kernel_size=1,strides=1)
    x = Upsampling2D(size=(2,2))(x)
    x = Concatenate()[x,x_small]
    x = YoloConv(x,filters=128,kernel_size=1,strides=1)
    x = YoloConv(x,filters=256,kernel_size=3,strides=1)
    x = YoloConv(x,filters=128,kernel_size=1,strides=1)
    x = YoloConv(x,filters=256,kernel_size=3,strides=1)
    x = YoloConv(x,filters=128,kernel_size=1,strides=1)

    y_small = YoloConv(x,filters=256,kernel_size=3,strides=1)
    y_small = Conv2D(filters=final_filters,kernel_size=1,strides=1,padding='same')(y_small)

    y_samll_shape = tf.shape(y_small)
    y_medium_shape = tf.shape(y_medium)
    y_large_shape = tf.shape(y_large)

    y_samll = tf.reshape(y_small,(y_samll_shape[0],y_samll_shape[1],y_samll_shape[2],3,-1))
    y_medium = tf.reshape(y_medium,(y_medium_shape[0],y_medium_shape[1],y_medium_shape[2],3,-1))
    y_large = tf.reshape(y_large,(y_large_shape[0],y_large_shape[1],y_large_shape[2],3,-1))

    if training:
        return Model(inputs,(y_samll,y_medium,y_large))

    box_small = Lambda(lambda x:get_absolute_yolo_box(x,anchors_wh[0:3],num_classes))(y_small)
    box_medium = Lambda(lambda x:get_absolute_yolo_box(x,anchors_wh[3:6],num_classes))(y_medium)
    box_large = Lambda(lambda x:get_absolute_yolo(x,anchors_wh[6:9],num_classes))(y_large)

    outputs = (box_small,box_medium,box_large)
    return Model(inputs,outputs)
    
def get_absolute_yolo_box(y_pred,valid_anchors_wh,num_classes):
    t_xy,t_wh,objectness,classes = tf.split(
        y_pred,(2,2,1,num_classes),axis=-1
    )

    objectness = tf.sigmoid(objectness)
    classes = tf.sigmoid(classes)

    grid_size = tf.shape(y_pred)[1]

    C_xy = tf.meshgrid(tf.range(grid_size),tf.range(grid_size))
    C_xy = tf.stack(C_xy,axis=-1)
    C_xy = tf.expand_dims(C_xy,axis=2)

    b_xy = tf.sigmoid(t_xy) + tf.cast(C_xy, tf.float32)
    b_xy = b_xy / tf.cast(grid_size,tf.float32)

    b_wh = tf.exp(t_wh) *valid_anchors_wh

    y_box = tf.concat([b_xy,b_wh],axis=1)


class YoloLoss(object):
    def __init__(self,num_classes,valid_anchors_wh):
        self.num_classes = num_classes
        self.ignore_threshold = 0.5
        self.valid_anchors_wh = valid_anchors_wh
        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5

    def __call__(self,y_true,y_pred):
        #Split y_pred into xy,wh,objectness and num_classes

        pred_xy_rel = tf.sigmoid(y_pred[...,0:2])
        pred_wh_rel = y_pred[...,2:4]

        pred_box_abs, pred_obj, pred_class = get_absolute_yolo_box(
            y_pred,self.valid_anchors_wh,self.num_classes
        )
        pred_box_abs = xywh_to_x1x2y1y1(pred_box_abs)

        true_xy_abs, true_wh_abs,true_obj,true_class = tf.split(
            y_true,(2,2,1,self.num_classes),axis=-1)
        true_box_abs = tf.concat([true_xy_abs,true_wh_abs],axis=-1)
        true_box_abs = xywh_to_x1x2y1y1(true_box_abs)

        true_box_rel = get_absolute_yolo_box(y_true,self.valid_anchors_wh)
        true_xy_rel = true_box_rel[...,0:2]
        true_wh_rel = true_box_abs[...,2:4]

        weight = 2 - true_wh_abs[...,0] * true_wh_abs[...,1]

        xy_loss = self.calc_xy_loss(true_obj,true_xy_rel,pred_xy_rel,weight)
        wh_loss = self.calc_wh_loss(true_obj,true_wh_rel,pred_wh_rel,weight)
        class_loss = self.calc_class_loss(true_obj,true_class,pred_class)

        ignore_mask = self.calc_ignore_mask(true_obj,true_box_abs,pred_box_abs)
        obj_loss = self.calc_obj_loss(true_obj,pred_obj,ignore_mask)

        return xy_loss + wh_loss + class_loss + obj_loss,(xy_loss,wh_loss,
                                                          class_loss,obj_loss)

    def calc_ignore_mask(self,true_obj,true_box,pred_box):

        true_box_shape = tf.shape(true_box)
        pred_box_shape = tf.shape(pred_box)

        true_box = tf.reshape(true_box,[true_box_shape[0],-1,4])
        true_box = tf.sort(true_box,axis=1,direction='DESCENDING')

        true_box = true_box[:,0:100,:]
        pred_box = tf.reshape(pred_box,[pred_box_shape[0],-1,4])

        iou = broadcast_iou(pred_box,true_box)
        
        best_iou = tf.reduce_max(iou,axis=-1)
        best_iou = tf.reshape(best_iou,[pred_box_shape[0],pred_box_shape[1],pred_box_shape[2],pred_box_shape[3]])

        ignore_mask = tf.cast(best_iou < self.ignore_threshold,tf.float32)
        ignore_mask = tf.expand_dims(ignore_mask,axis=-1)

        return ignore_mask

    def clac_obj_loss(self,true_obj,pred_obj,ignore_mask):

        obj_entropy = binary_crossentropy(pred_obj,true_obj)
        noobj_loss = (1-true_obj)*obj_entropy * ignore_mask

        obj_loss = tf.reduce_sum(noobj_loss,axis=(1,2,3,4))
        noobj_loss = tf.reduce_sum(noobj_loss,axis=(1,2,3,4)) * self.lambda_noobj

        return obj_loss + noobj_loss

    def calc_class_loss(self,true_obj,true_class,pred_class):

        class_loss = binary_crossentropy(pred_class,true_class)
        class_loss = true_obj * class_loss
        class_loss = tf.reduce_sum(class_loss,axis=(1,2,3,4))

        return class_loss

    def calc_xy_loss(self,true_obj,true_xy,pred_xy,weight):

        xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy),axis=-1)
        true_obj = tf.squeeze(true_obj,axis=-1)

        xy_loss = true_obj * xy_loss * weight
        xy_loss = tf.reduce_sum(xy_loss,axis=(1,2,3)) * self.lambda_coord

        return xy_loss

    def calc_wh_loss(self,true_obj,true_wh,pred_wh,weight):

        wh_loss = tf.reduce_sum(tf.square(true_wh-pred_wh),axis=-1)
        true_obj = tf.squeeze(true_obj,axis=-1)

        wh_loss = true_obj * wh_loss * weight
        wh_loss = tf.reduce_sum(wh_loss,axis=(1,2,3)) * self.lambda_coord
        return wh_loss


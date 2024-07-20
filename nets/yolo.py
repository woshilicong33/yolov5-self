from tensorflow.keras.layers import (Concatenate, Input, Lambda, UpSampling2D,
                                     ZeroPadding2D)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from nets.CSPdarknet import (C3,C3_dw, DarknetConv2D, DarknetConv2D_BN_SiLU,DarknetConv2D_BN_ReLU_dw,
                             darknet_body,DarknetConv2D_dw)
from nets.yolo_training import yolo_loss

import tensorflow as tf
#---------------------------------------------------#
#   Panet网络的构建，并且获得预测结果
#---------------------------------------------------#
def yolo_body(input_shape, anchors_mask, num_classes, phi, weight_decay=5e-4,convert_model =False,anchors=None):
    depth_dict          = {'s' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
    width_dict          = {'s' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
    dep_mul, wid_mul    = depth_dict[phi], width_dict[phi]

    base_channels       = int(wid_mul * 64)  # 64
    base_depth          = max(round(dep_mul * 3), 1)  # 3

    inputs              = Input(input_shape)
    #---------------------------------------------------#   
    #   生成主干模型，获得三个有效特征层，他们的shape分别是：
    #   80, 80, 256
    #   40, 40, 512
    #   20, 20, 1024
    #---------------------------------------------------#
    feat1, feat2, feat3 = darknet_body(inputs, base_channels, base_depth, weight_decay)

    P5          = DarknetConv2D_BN_ReLU_dw(int(base_channels * 2), (1, 1), weight_decay=weight_decay, name = 'conv_for_feat3')(feat3)  
    P5_upsample = UpSampling2D()(P5) 
    P5_upsample = Concatenate(axis = -1)([P5_upsample, feat2])
    P5_upsample = C3_dw(P5_upsample, int(base_channels * 2), base_depth, shortcut = False, weight_decay=weight_decay, name = 'conv3_for_upsample1')

    P4          = DarknetConv2D_BN_ReLU_dw(int(base_channels * 2), (1, 1), weight_decay=weight_decay, name = 'conv_for_feat2')(P5_upsample)
    P4_upsample = UpSampling2D()(P4)
    P4_upsample = Concatenate(axis = -1)([P4_upsample, feat1])
    P3_out      = C3_dw(P4_upsample, int(base_channels * 2), base_depth, shortcut = False, weight_decay=weight_decay, name = 'conv3_for_upsample2')

    P3_downsample   = ZeroPadding2D(((1, 0),(1, 0)))(P3_out)
    P3_downsample   = DarknetConv2D_BN_ReLU_dw(int(base_channels * 2), (3, 3), strides = (2, 2), weight_decay=weight_decay, name = 'down_sample1')(P3_downsample)
    P3_downsample   = Concatenate(axis = -1)([P3_downsample, P4])
    P4_out          = C3_dw(P3_downsample, int(base_channels * 2), base_depth, shortcut = False, weight_decay=weight_decay, name = 'conv3_for_downsample1') 

    P4_downsample   = ZeroPadding2D(((1, 0),(1, 0)))(P4_out)
    P4_downsample   = DarknetConv2D_BN_ReLU_dw(int(base_channels * 1), (3, 3), strides = (2, 2), weight_decay=weight_decay, name = 'down_sample2')(P4_downsample)
    P4_downsample   = Concatenate(axis = -1)([P4_downsample, P5])
    P5_out          = C3_dw(P4_downsample, int(base_channels * 2), base_depth, shortcut = False, weight_decay=weight_decay, name = 'conv3_for_downsample2')

    out2 = DarknetConv2D_dw(len(anchors_mask[2]) * (5 + num_classes), (1, 1), strides = (1, 1), weight_decay=weight_decay, name = 'yolo_head_P3')(P3_out)
    out1 = DarknetConv2D_dw(len(anchors_mask[1]) * (5 + num_classes), (1, 1), strides = (1, 1), weight_decay=weight_decay, name = 'yolo_head_P4')(P4_out)
    out0 = DarknetConv2D_dw(len(anchors_mask[0]) * (5 + num_classes), (1, 1), strides = (1, 1), weight_decay=weight_decay, name = 'yolo_head_P5')(P5_out)
    print("convert_model",anchors)
    if convert_model:
        anchor_mask     = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        out0_out = decode_box(out0,num_classes,anchors[anchor_mask[0]],0)       
        out1_out = decode_box(out1,num_classes,anchors[anchor_mask[1]],1)       
        out2_out = decode_box(out2,num_classes,anchors[anchor_mask[2]],2)
        return Model(inputs, [out0_out, out1_out, out2_out])
    return Model(inputs, [out0, out1, out2])
def decode_box(input_feat,num_classes,anchors,i):
    num_anchors = len(anchors)
    grid_shape = K.shape(input_feat)[1:3]
    grid_x  = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, num_anchors, 1])
    grid_y  = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], num_anchors, 1])
    grid    = K.cast(K.concatenate([grid_x, grid_y]), K.dtype(input_feat))
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, num_anchors, 2])
    anchors_tensor = K.tile(anchors_tensor, [grid_shape[0], grid_shape[1], 1, 1])
    input_feat     = K.reshape(input_feat, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    box_xy         = (K.sigmoid(input_feat[..., :2]) * 2 - 0.5 + grid) / K.cast([grid_shape[0],grid_shape[1]], K.dtype(input_feat))
    box_wh         = (K.sigmoid(input_feat[..., 2:4]) * 2) ** 2 * anchors_tensor / K.cast([480,480], K.dtype(input_feat))

    box_confidence  = K.sigmoid(input_feat[..., 4:5])
    box_class_probs = K.sigmoid(input_feat[..., 5:])

    box_out = K.concatenate([box_xy,box_wh],axis = 4)
    output = K.concatenate([K.reshape(box_out, (1, -1, 4)),K.reshape(box_confidence, (1, -1, 1)),K.reshape(box_class_probs,(1,-1,num_classes))],-1)

    return output
def get_train_model(model_body, input_shape, num_classes, anchors, anchors_mask, label_smoothing):
    y_true = [Input(shape = (input_shape[0] // {0:32, 1:16, 2:8}[l], input_shape[1] // {0:32, 1:16, 2:8}[l], \
                                len(anchors_mask[l]), num_classes + 5)) for l in range(len(anchors_mask))]
    model_loss  = Lambda(
        yolo_loss, 
        output_shape    = (1, ), 
        name            = 'yolo_loss', 
        arguments       = {
            'input_shape'       : input_shape, 
            'anchors'           : anchors, 
            'anchors_mask'      : anchors_mask, 
            'num_classes'       : num_classes, 
            'label_smoothing'   : label_smoothing, 
            'balance'           : [0.4, 1.0, 4],
            'box_ratio'         : 0.08,
            'obj_ratio'         : 1 * (input_shape[0] * input_shape[1]) / (480 ** 2), 
            'cls_ratio'         : 0.5 * (num_classes / 80)
        }
    )([*model_body.output, *y_true])
    model       = Model([model_body.input, *y_true], model_loss)
    return model

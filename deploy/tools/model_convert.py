#!/usr/bin/env python
# coding=utf-8
###############convert To tflite
import sys
import os
import tensorflow as tf
gpu_device_name = tf.test.gpu_device_name()
print(gpu_device_name)

sys.path.append("../")

convert = 'onnx'
if convert == "tflite":
    import tensorflow as tf
    import os 
    from nets.yolo import  yolo_body
    anchors_path    = './model_data/yolo_anchors.txt'
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    classes = 3
    model_body  = yolo_body((480, 480, 3), anchors_mask, classes,'s',5e-4)
    model_body.load_weights("../best_epoch_weights.h5")
    converter = tf.lite.TFLiteConverter.from_keras_model(model_body)
    tflite_model = converter.convert()

    open("./best_epoch_weights.tflite","wb").write(tflite_model)

if convert == "onnx":
    from nets.yolo import  yolo_body
    import tensorflow as tf
    import tf2onnx
    anchors_path    = 'model_data/yolo_anchors.txt'
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    classes = 3
    model_body  = yolo_body((480, 480, 3), anchors_mask, classes,'s',5e-4)
    model_body.load_weights("../best_epoch_weights.h5")

    # model = tf.keras.models.load_model("./logs/best_epoch_weights.h5")
    spec = (tf.TensorSpec((1, 480, 480, 3), tf.float32, name="input_1"),)  # 输入签名参数，(None, 128, 128, 3)决定输入的size
    output_path = "./best_epoch_weights.onnx"                                   # 输出路径
    
    # 转换并保存onnx模型，opset决定选用的算子集合
    model_proto, _ = tf2onnx.convert.from_keras(model_body, input_signature=spec, opset=11, output_path=output_path)
    output_names = [n.name for n in model_proto.graph.output]
    print(output_names)
if convert == "pb":
    import tensorflow as tf  
    from tensorflow.keras.models import load_model  
    from nets.yolo import  yolo_body
    import numpy as np
    anchors_path    = 'model_data/yolo_anchors.txt'
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    classes = 3
    model  = yolo_body((480, 480, 3), anchors_mask, classes,'s',5e-4)
    model.load_weights("../best_epoch_weights.h5")

    # 定义一个输入TensorSpec来指定模型的输入形状和类型  
    input_tensor = tf.convert_to_tensor(np.zeros(tuple(model.inputs[0].shape[1:])), dtype=model.inputs[0].dtype)  
    input_spec = tf.TensorSpec(shape=model.inputs[0].shape, dtype=model.inputs[0].dtype) 
    # 获取具体的函数表示  
    full_model = tf.function(lambda x: model(x))  
    concrete_func = full_model.get_concrete_function(input_spec) 
    # 将模型中的变量转换为常量（即冻结模型）  
    frozen_func =  tf.compat.v1.graph_util.extract_sub_graph(  
        sess=tf.compat.v1.Session(graph=concrete_func.graph),  
        input_graph_def=concrete_func.graph.as_graph_def(),  
        output_node_names=[out.op.name for out in concrete_func.outputs]  
    )
    with tf.io.gfile.GFile("./best_epoch_weights.pb", 'wb') as f:  
        f.write(frozen_func.SerializeToString())
# from tensorflow import keras
# import keras2onnx
# import onnx
# from tensorflow.keras.models import load_model
# model = load_model('./logs/best_epoch_weights.h5')
# onnx_model = keras2onnx.convert_keras(model, model.name)
# temp_model_file = './logs/best_epoch_weights.onnx'
# onnx.save_model(onnx_model, temp_model_file)
#!/usr/bin/env python
# coding=utf-8
###############convert To tflite
convert = 'onnx'
if convert == "tflite":
    import tensorflow as tf
    import os 
    from nets.yolo import  yolo_body
    anchors_path    = 'model_data/yolo_anchors.txt'
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    classes = 3
    model_body  = yolo_body((480, 480, 3), anchors_mask, classes,'s',5e-4)
    model_body.load_weights("./logs/best_epoch_weights.h5")

    # keras_model = tf.keras.models.load_model("./logs/best_epoch_weights.h5")

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
    model_body.load_weights("./logs/best_epoch_weights.h5")

    # model = tf.keras.models.load_model("./logs/best_epoch_weights.h5")
    spec = (tf.TensorSpec((1, 480, 480, 3), tf.float32, name="input_1"),)  # 输入签名参数，(None, 128, 128, 3)决定输入的size
    output_path = "./best_epoch_weights.onnx"                                   # 输出路径
    
    # 转换并保存onnx模型，opset决定选用的算子集合
    model_proto, _ = tf2onnx.convert.from_keras(model_body, input_signature=spec, opset=13, output_path=output_path)
    output_names = [n.name for n in model_proto.graph.output]
    print(output_names)
   

# from tensorflow import keras
# import keras2onnx
# import onnx
# from tensorflow.keras.models import load_model
# model = load_model('./logs/best_epoch_weights.h5')
# onnx_model = keras2onnx.convert_keras(model, model.name)
# temp_model_file = './logs/best_epoch_weights.onnx'
# onnx.save_model(onnx_model, temp_model_file)
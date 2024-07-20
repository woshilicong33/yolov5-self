#!/usr/bin/env python
# coding=utf-8
###############convert To tflite
import sys
import os
import tensorflow as tf



os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 这一句根据需要添加，作用是指定GPU

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

gpu_device_name = tf.test.gpu_device_name()
print(gpu_device_name)


print (os.getcwd())
convert = 'onnx'
if convert == "tflite":
    from yolo import YOLO
    yolo = YOLO()
    yolo.generate()
    yolo.conver_to_tflite("./best_epoch_weights.tflite")

if convert == "onnx":
    from yolo import YOLO
    yolo = YOLO()
    yolo.generate()
    yolo.convert_to_onnx(True,"./model.onnx")
if convert == "pb":
    from yolo import YOLO
    yolo = YOLO()
    yolo.generate()
    yolo.convert_to_pb("model_pb")
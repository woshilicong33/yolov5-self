#环境为tensorflow2.3
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
 
# 导入数据
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# 观察数据
print (x_train.shape)
plt.imshow(x_train[1000])
print (y_train[1000])
 
train_images=x_train/255.0
 
#(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# 归一化
x_train, x_test = x_train / 255.0, x_test / 255.0
 
class_names = ['0', '1', '2', '3', '4','5', '6', '7', '8', '9']
 
plt.imshow(x_train[2000])
 
 
x_train = x_train.reshape((x_train.shape[0],28,28,1)).astype('float32') 
x_test = x_test.reshape((x_test.shape[0],28,28,1)).astype('float32') #-1代表那个地方由其余几个值算来的
x_train = x_train/255
x_test = x_test/255
x_train = np.pad(x_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
x_test = np.pad(x_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
print (x_train.shape)
 
########################################################################################
#模型建立
 
#序贯模型（Sequential):单输入单输出
model = tf.keras.Sequential()
 
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import InputLayer, Dropout, Conv1D, Flatten
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 这一句根据需要添加，作用是指定GPU

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

#我加的这一层
model.add(InputLayer(input_shape=(32,32,1), name='x_input'))
 
#Layer 1
#Conv Layer 1
model.add(Conv2D(filters = 6, kernel_size = 5, strides = 1, activation = 'relu', input_shape = (32,32,1)))
#Pooling layer 1
model.add(MaxPooling2D(pool_size = 2, strides = 2))
 
#Layer 2
#Conv Layer 2
model.add(Conv2D(filters = 16, kernel_size = 5,strides = 1,activation = 'relu',input_shape = (14,14,6)))
#Pooling Layer 2
model.add(MaxPooling2D(pool_size = 2, strides = 2))
#Flatten
model.add(Flatten())
 
#Layer 3
#Fully connected layer 1
model.add(Dense(units = 120, activation = 'relu'))
 
#Layer 4
#Fully connected layer 2
model.add(Dense(units = 42, activation = 'relu'))######
 
#Layer 5
#Output Layer
model.add(Dense(units = 10, activation = 'softmax'))
model.compile(optimizer = 'adam',  loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics = ['accuracy'])
 
model.summary()
#########################################################################################
 
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
 
history = model.fit(x_train, y_train, epochs=1,batch_size=32,
                    validation_data=(x_test, y_test))
 
history.history.keys()#可视化
#准确率训练数据可视化
plt.plot(history.epoch, history.history.get('accuracy'),label='accuracy')
plt.plot(history.epoch, history.history.get('val_accuracy'),label='val_accuracy')
plt.legend()
 
model.save('./LeNet_5.h5')#keras保存模型，名字可以任取，但要由.h5后缀,可以更改为自己的路径
#测试模型
model.evaluate(x_test, y_test)
 
# 模型保存
save_path = "./pbpath"#pb模型保存路径
model.save(save_path)

train_images = x_train
train_images.shape[2]
from nets.yolo import  yolo_body

anchors_path    = 'model_data/yolo_anchors.txt'
anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
classes = 3
model_body  = yolo_body((480, 480, 3), anchors_mask, classes,'s',5e-4)
model_body.load_weights("./best_epoch_weights.h5") 
#int8真的量化成功了！！！！
#还可以部署到openMV上！！！
def representative_data_gen():
    for image in train_images[0:100,:,:]:
        yield[image.reshape(-1,480,480,3).astype("float32")]
 
converter = tf.lite.TFLiteConverter.from_keras_model(model_body)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
 
#--------新增加的代码--------------------------------------------------------
# 确保量化操作不支持时抛出异常
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# 设置输入输出张量为uint8格式
converter.inference_input_type = tf.int8 #or unit8
converter.inference_output_type = tf.int8 #or unit8
#----------------------------------------------------------------------------
 
tflite_model_quant = converter.convert()
#保存转换后的模型
FullInt_name = "int8.tflite"
open(FullInt_name, "wb").write(tflite_model_quant)
 
#查看输入输出类型
interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
input_type = interpreter.get_input_details()[0]['dtype']
print('input: ', input_type)
output_type = interpreter.get_output_details()[0]['dtype']
print('output: ', output_type)
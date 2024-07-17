import numpy as np
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 
def onnx_model():
    import os, sys
    import cv2
    sys.path.append(os.getcwd())
    import onnxruntime
    import numpy as np
    from PIL import Image

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    image   = Image.open("../2.jpg")
    image   = cvtColor(image)
    image   = image.resize((480,480), Image.BICUBIC)

    image  = np.array(image, np.float32) /255.0
    image_data = image[np.newaxis, :, :, :]
    onnx_model_path = "model.onnx"
    onnx_session = onnxruntime.InferenceSession(onnx_model_path)  
    inputs = {"input": image_data}  
    output = onnx_session.run(None, inputs)

    for i in range(3):
        for p in range(output[i].shape[1]):
            if output[i][0,p,4]  > 0.2:
                x = output[i][0,p,0]*640
                y = output[i][0,p,1]*480
                w = output[i][0,p,2]*640
                h = output[i][0,p,3]*480
                confid  = output[i][0,p,4]
                c1 = output[i][0,p,5]
                c2 = output[i][0,p,6]
                c3 = output[i][0,p,7]
                print("x:",int(x-w/2)," y:",int(y-h/2)," x2:",int(x+w/2)," y2:", int(y+h/2),"confid:",confid," c1:",c1," c2:",c2," c3:",c3)
                # cv2.rectangle (image_source, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0, 0, 255), 2)
    # cv2.imwrite("./result_onnx.jpg",image_source)
def h5(image_path,h5_model_path,model_shape=[480,480]):
    from model_body.body_unit.utils import get_anchors, get_classes
    from tensorflow.keras.models import load_model
    from model_body.yolov5 import yolo_body
    from PIL import Image
    import cv2
    import os
    import random
    from tensorflow.keras.layers import Input
    import numpy as np
    from PIL import ImageDraw
    def cvtColor(image):
        if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
            return image 
        else:
            image = image.convert('RGB')
            return image 
    classes_path    = 'info_data/xmclasses.txt'
    anchors_mask        = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    input_shape         = [model_shape[0], model_shape[1],3]
    class_names, num_classes = get_classes(classes_path)
    model = yolo_body(input_shape,anchors_mask,num_classes,phi='s')
    model.load_weights(h5_model_path, by_name=True, skip_mismatch=True)

    image_lines = os.listdir(image_path)
    random.shuffle(image_lines)
    if ".jpg" not in image_path:
        for images_line in image_lines:
            image_source   = Image.open(image_path+images_line)
            image   = image_source.convert('RGB')
            # image   = Image.open("./source.jpg")
            a=ImageDraw.ImageDraw(image_source)
            iw, ih  = image.size
            image   = image.resize((model_shape[0],model_shape[1]), Image.BICUBIC)
            image_data  = np.array(image, np.float32)/255.0
            image_data = image_data[np.newaxis, :, :, :]

            output= model.predict(image_data)


            for p in range(output.shape[1]):
                # print(output[0,p,:])
                if output[0,p,4]  > 0.5:
                    x = output[0,p,0]*640
                    y = output[0,p,1]*480
                    w = output[0,p,2]*640
                    h = output[0,p,3]*480
                    confid  = output[0,p,4]
                    c1 = output[0,p,5]
                    c2 = output[0,p,6]
                    c3 = output[0,p,7]
                    print("x:",int(x-w/2)," y:",int(y-h/2)," x2:",int(x+w/2)," y2:", int(y+h/2),"confid:",confid," c1:",c1," c2:",c2," c3:",c3)
                    # print("x:",x," y:",y," w:",w," h:",h,"confid:",confid)
                    a.rectangle(((int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2))),fill=None,outline='red',width=5)
                    # a.rectangle(((10, 20), (30,40)),outline='red',width=5)

            image_source.save("./result_h5.jpg")
            exit()
def mnn(image_path,mnn_model_path,output_name_list,model_shape=[480,480]):
    import time
    import MNN
    import cv2
    import os 
    import numpy as np
    import random
    interpreter = MNN.Interpreter(mnn_model_path)
    config = {}
    config['precision'] = 'high'
    config['backend'] = 'CPU'
    config['thread'] = 1
    session = interpreter.createSession(config)
    input_tensors = interpreter.getSessionInput(session)
    image_lines = os.listdir(image_path)
    random.shuffle(image_lines)
    if ".jpg" not in image_path:
        for images_line in image_lines:
            image  = cv2.imread(image_path+images_line)
            image_source = image
            image = cv2.resize(image,(model_shape[0],model_shape[1]))
            cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image  = np.array(image, np.float32) /255.
            # image = np.transpose(image, (2,0,1))
            image_data = image[np.newaxis, :, :, :]

            input_format = (1, 3,model_shape[0], model_shape[1])
            tmp_input = MNN.Tensor(input_format, MNN.Halide_Type_Float, image_data, MNN.Tensor_DimensionType_Caffe)
            input_tensors.copyFrom(tmp_input)
            interpreter.runSession(session)
            infer_result = interpreter.getSessionOutputAll(session)
            for layername in output_name_list:
                for p in range(infer_result[layername].getNumpyData().shape[1]):
                    if infer_result[layername].getNumpyData()[0,p,4]  > 0.2:
                        x = infer_result[layername].getNumpyData()[0,p,0]*640
                        y = infer_result[layername].getNumpyData()[0,p,1]*480
                        w = infer_result[layername].getNumpyData()[0,p,2]*640
                        h = infer_result[layername].getNumpyData()[0,p,3]*480
                        confid  = infer_result[layername].getNumpyData()[0,p,4]
                        c1 = infer_result[layername].getNumpyData()[0,p,5]
                        c2 = infer_result[layername].getNumpyData()[0,p,6]
                        c3 = infer_result[layername].getNumpyData()[0,p,7]
                        print("x:",int(x-w/2)," y:",int(y-h/2)," x2:",int(x+w/2)," y2:", int(y+h/2),"confid:",confid," c1:",c1," c2:",c2," c3:",c3)
                        # print("x:",x," y:",y," w:",w," h:",h,"confid:",confid)

                        cv2.rectangle (image_source, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0, 0, 255), 2)
            cv2.imwrite("./result_mnn.jpg",image_source)
            exit()
    image  = cv2.imread("../dataset/datamix/images/172.jpg2024-05-29-01:36:18.809775.jpg")
    image_source = image

    # 先获取输入输出
    
    



  
    # import MNN.expr as F
    # vars = F.load_as_dict("model.mnn")
    # inputVar = vars["input"]
    # # 查看输入信息
    # print(inputVar.shape)
    # print(inputVar.data_format)
image_path = "../dataset/datamix/images/"
mnn_model_path = './model.mnn'
h5_model_path = "./logs/ep030-loss0.989-val_loss0.994.h5"
output_name_list = [ "tf.reshape_7","tf.reshape_3", "tf.reshape_7" ]
# mnn(image_path,mnn_model_path,output_name_list)
# h5(image_path,h5_model_path)
onnx_model()
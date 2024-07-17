class onnx_model():
    def __init__(self,onnx_model_path):
        import os, sys
        import cv2
        import numpy as np
        sys.path.append(os.getcwd())
        import onnxruntime
        import numpy as np
        from PIL import Image,ImageDraw
        self.onnx_model_path = onnx_model_path
        self.onnx_session = onnxruntime.InferenceSession(onnx_model_path) 
        image_source = Image.open("1.jpg")
        draw = ImageDraw.Draw(image_source)
        image  = self.cvtColor(image_source)
        image_source = image
        image_data  =self.resize_image(image, (480, 480), True)
        image_data  = np.expand_dims(np.array(image_data, dtype='float32')/255., 0)

        input_feed = {"input":image_data}
        output     = self.onnx_session.run(None,input_feed)
        for row in range(np.shape(output[0])[0]):
            if output[2][row]  > 0.5:
                x = output[0][row][0]*640
                y = output[0][row][1]*480
                w = output[1][row][0]*640
                h = output[1][row][1]*480
                x1 = int(y-h//2)
                y1 = int(x-w//2)
                x2 = int(y+h//2)
                y2 = int(x+w//2)
                confid  = output[2][row][0]
                c1 = output[3][row]
                print("x:",x1," y:",y1," x2:",x2," y2:", y2,"confid:",confid," c1:",c1)
                draw.rectangle([y1,x1,y2,x2], outline=(0,0,0))
        image_source.save("./result_onnx.jpg")

        return output_name
    def cvtColor(self,image):
        import numpy as np
        if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
            return image 
        else:
            image = image.convert('RGB')
            return image 
    def resize_image(self,image, size, letterbox_image):
        from PIL import Image
        iw, ih  = image.size
        w, h    = size
        if letterbox_image:
            scale   = min(w/iw, h/ih)
            nw      = int(iw*scale)
            nh      = int(ih*scale)

            image   = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', size, (128,128,128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        else:
            new_image = image.resize((w, h), Image.BICUBIC)
        return new_image
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


class MNN():
    def __init__(self,image_path,mnn_model_path,output_name_list,model_shape=[480,480]):
        self.model_shape = model_shape
        self.mnn_model_path = mnn_model_path
        self.detec_image()
    def detec_image(self):
        import time
        import MNN
        import os 
        import random
        import numpy as np
        from PIL import Image,ImageDraw

        image_source = Image.open("1.jpg")
        draw = ImageDraw.Draw(image_source)
        image  = self.cvtColor(image_source)

        image_data  =self.resize_image(image, (480, 480), True)

        image_data  = np.expand_dims(np.array(image_data, dtype='float32')/255., 0)

        interpreter = MNN.Interpreter(self.mnn_model_path)
        config = {}
        config['precision'] = 'high'
        config['backend'] = 'CPU'
        config['thread'] = 1
        session = interpreter.createSession(config)
        input_tensors = interpreter.getSessionInput(session)

        input_format = (1, self.model_shape[0], self.model_shape[1],3)
        tmp_input = MNN.Tensor(input_format, MNN.Halide_Type_Float, image_data, MNN.Tensor_DimensionType_Caffe)
        input_tensors.copyFrom(tmp_input)
        interpreter.runSession(session)
        infer_result = interpreter.getSessionOutputAll(session)
        for row in range(infer_result["yolo_eval_2"].getNumpyData().shape[0]):
            if infer_result["yolo_eval_2"].getNumpyData()[row,0]  > 0.5:
                confid = infer_result["yolo_eval_2"].getNumpyData()[row,0]
                x = infer_result["yolo_eval"].getNumpyData()[row,1]*480
                y = infer_result["yolo_eval"].getNumpyData()[row,0]*640
                w = infer_result["yolo_eval_1"].getNumpyData()[row,1]*480
                h = infer_result["yolo_eval_1"].getNumpyData()[row,0]*640
                x1 = x-w//2
                y1 = y-h//2
                x2 = x+w//2
                y2 = y+w//2
                print("confid:",confid,"x1:",int(x1)," y:",int(y1),"x2",int(x2),"y2:",int(y2))
    def resize_image(self,image, size, letterbox_image):
        from PIL import Image
        import numpy as np
        iw, ih  = image.size
        w, h    = size
        if letterbox_image:
            scale   = min(w/iw, h/ih)
            nw      = int(iw*scale)
            nh      = int(ih*scale)

            image   = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', size, (128,128,128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        else:
            new_image = image.resize((w, h), Image.BICUBIC)

        return new_image
    def cvtColor(self,image):
        import numpy as np
        if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
            return image 
        else:
            image = image.convert('RGB')
            return image 

image_path = "../dataset/datamix/images/"

h5_model_path = "./logs/ep030-loss0.989-val_loss0.994.h5"
output_name_list = [ "yolo_eval","yolo_eval_1", "yolo_eval_2","yolo_eval_3" ]


mnn_model_path = './best_epoch_weights.mnn'
MNN(image_path,mnn_model_path,output_name_list)

# onnx_model_path = "./best_epoch_weights.onnx"
# onnx_model(onnx_model_path)
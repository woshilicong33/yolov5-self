
from keras.callbacks import Callback
import os 
from tqdm import tqdm
from PIL import Image
import numpy as np
import MNN
import glob
import shutil
import json
import math
import random
os.environ["CUDA_VISIBLE_DEVICES"]  = "1"
class onnx_model():
    def __init__(self,onnx_model_path,val_path):
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
        self.start_eval(self,val_path,map_out_path = ".temp_map_out")
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


class MNN_model():

    def __init__(self,image_path,val_path,mnn_model_path,output_name_list,model_shape=[480,480]):

        self.model_shape = model_shape
        self.mnn_model_path = mnn_model_path
        self.val_path = val_path
        self.class_names = ["ladder","leaf","drain"]


        self.interpreter = MNN.Interpreter(self.mnn_model_path)
        config = {}
        config['precision'] = 'high'
        config['backend'] = 'CPU'
        config['thread'] = 1
        self.session = self.interpreter.createSession(config)
        self.input_tensors = self.interpreter.getSessionInput(self.session)
        self.input_format = (1, self.model_shape[0], self.model_shape[1],3)

        self.start_eval(self.val_path)

    def start_eval(self,val_path,map_out_path = ".temp_map_out"):

        if not os.path.exists(map_out_path):
            os.makedirs(map_out_path)
        if not os.path.exists(os.path.join(map_out_path, "ground-truth")):
            os.makedirs(os.path.join(map_out_path, "ground-truth"))
        if not os.path.exists(os.path.join(map_out_path, "detection-results")):
            os.makedirs(os.path.join(map_out_path, "detection-results"))
        print("Get map.") 
        with open(val_path,"r") as f:
            val_lines = f.readlines()
            random.shuffle(val_lines)
            for annotation_line in tqdm(val_lines):
                line        = annotation_line.split()
                image_id    = os.path.basename(line[0]).split('.')[0]
                image       = Image.open(line[0])
                gt_boxes    = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

                self.get_map_txt(image_id, image, self.class_names, map_out_path)
                with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                    for box in gt_boxes:
                        left, top, right, bottom, obj = box
                        obj_name = self.class_names[obj]

                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))

            print("Calculate Map.")
            try:
                print("coco_map")
                temp_map = self.get_coco_map(class_names = self.class_names, path = map_out_path)[1]
            except:
                temp_map = self.get_map(50, False, path = map_out_path)
        # shutil.rmtree(map_out_path)
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
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #---------------------------------------------------------#
        image       = self.cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = self.resize_image(image, (480, 480), True)
        #---------------------------------------------------------#
        #   添加上batch_size维度，并进行归一化
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.array(image_data, dtype='float32')/255., 0)

        #---------------------------------------------------------#
        #   将图像输入网络当中进行预测！
        #---------------------------------------------------------#
        input_image_shape   = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
        out_boxes, out_scores, out_classes  = self.get_pred(image_data) 

        for i, c in enumerate(out_classes):

            predicted_class             = self.class_names[int(c)]
            try:
                score                   = str(out_scores[i].numpy())
            except:
                score                   = str(out_scores[i])
            top, left, bottom, right    = out_boxes[i]
            if predicted_class not in class_names:
                continue
            # print("pre save:",predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom)))

            # exit()
            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return  
    def get_pred(self,image):
        self.tmp_input = MNN.Tensor(self.input_format, MNN.Halide_Type_Float, image, MNN.Tensor_DimensionType_Caffe)
        self.input_tensors.copyFrom(self.tmp_input)
        self.interpreter.runSession(self.session)
        infer_result = self.interpreter.getSessionOutputAll(self.session)
        pre_xy,pre_wh, pre_scores, pre_classes = infer_result["yolo_eval"],infer_result["yolo_eval_1"],infer_result["yolo_eval_2"],infer_result["yolo_eval_3"]
        out_x1y1 = []
        out_x2y2 = []
        out_score = []
        out_class = []
        out_boxes = []
        for row in range(infer_result["yolo_eval_2"].getNumpyData().shape[0]):
            if infer_result["yolo_eval_2"].getNumpyData()[row,0]  > 0.5:
                confid = infer_result["yolo_eval_2"].getNumpyData()[row,0]
                x = pre_xy.getNumpyData()[row,1]*480
                y = pre_xy.getNumpyData()[row,0]*640
                w = pre_wh.getNumpyData()[row,1]*480
                h = pre_wh.getNumpyData()[row,0]*640
                x1 = x-w//2
                y1 = y-h//2
                x2 = x+w//2
                y2 = y+w//2
                out_x1y1.append([x1,y1])
                out_x2y2.append([x2,y2])

                class_score = np.max(pre_classes.getNumpyData()[row],axis = 0)*infer_result["yolo_eval_2"].getNumpyData()[row,0]
                class_num = np.argmax(np.expand_dims(pre_classes.getNumpyData()[row],axis=1),axis=0)
                out_score.append(class_score)
                out_class.append(class_num)
                
                # print("class_num:",self.class_names[class_num[0]],"class_score:",class_score,"x1:",int(x1)," y:",int(y1),"x2",int(x2),"y2:",int(y2))

        box_x1y1  = np.reshape(np.array(out_x1y1),[-1,2])
        box_x2y2  = np.reshape(np.array(out_x2y2),[-1,2])

        out_boxes = np.concatenate((box_x1y1,box_x2y2,np.reshape(np.array(out_score),[-1,1])),axis=1)

        keep = self.nms(out_boxes, 0.4)
        boxes_final=[]
        score_final=[]
        class_final=[]

        for i in range(len(keep)):

            boxes_final.append(out_boxes[keep[i]][0:4])
            score_final.append(out_boxes[keep[i]][4])
            class_final.append(out_class[keep[i]][0])
        return boxes_final,score_final,class_final

    def nms(self,dets, thresh):
        """Pure Python NMS baseline."""
        # x1、y1、x2、y2、以及score赋值
        x1 = dets[:, 0]  # xmin
        y1 = dets[:, 1]  # ymin
        x2 = dets[:, 2]  # xmax
        y2 = dets[:, 3]  # ymax
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        # argsort()返回数组值从小到大的索引值
        order = scores.argsort()[::-1]    
        keep = []
        while order.size > 0:  # 还有数据
            i = order[0]
            keep.append(i)
            if order.size==1:break
            # 计算当前概率最大矩形框与其他矩形框的相交框的坐标
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            # 计算相交框的面积
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            # 计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积）
            IOU = inter / (areas[i] + areas[order[1:]] - inter)
        
            # 找到重叠度不高于阈值的矩形框索引
            left_index = (np.where(IOU <= thresh))[0]
            
            # 将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
            order = order[left_index + 1]
            
        return keep
    def get_map(self,MINOVERLAP, draw_plot, score_threhold=0.4, path = './map_out'):
        GT_PATH             = os.path.join(path, 'ground-truth')
        DR_PATH             = os.path.join(path, 'detection-results')
        IMG_PATH            = os.path.join(path, 'images-optional')
        TEMP_FILES_PATH     = os.path.join(path, '.temp_files')
        RESULTS_FILES_PATH  = os.path.join(path, 'results')

        show_animation = True
        if os.path.exists(IMG_PATH): 
            for dirpath, dirnames, files in os.walk(IMG_PATH):
                if not files:
                    show_animation = False
        else:
            show_animation = False

        if not os.path.exists(TEMP_FILES_PATH):
            os.makedirs(TEMP_FILES_PATH)
            
        if os.path.exists(RESULTS_FILES_PATH):
            shutil.rmtree(RESULTS_FILES_PATH)
        else:
            os.makedirs(RESULTS_FILES_PATH)
        if draw_plot:
            try:
                matplotlib.use('TkAgg')
            except:
                pass
            os.makedirs(os.path.join(RESULTS_FILES_PATH, "AP"))
            os.makedirs(os.path.join(RESULTS_FILES_PATH, "F1"))
            os.makedirs(os.path.join(RESULTS_FILES_PATH, "Recall"))
            os.makedirs(os.path.join(RESULTS_FILES_PATH, "Precision"))
        if show_animation:
            os.makedirs(os.path.join(RESULTS_FILES_PATH, "images", "detections_one_by_one"))

        ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
        if len(ground_truth_files_list) == 0:
            error("Error: No ground-truth files found!")
        ground_truth_files_list.sort()
        gt_counter_per_class     = {}
        counter_images_per_class = {}

        for txt_file in ground_truth_files_list:
            file_id     = txt_file.split(".txt", 1)[0]
            file_id     = os.path.basename(os.path.normpath(file_id))
            temp_path   = os.path.join(DR_PATH, (file_id + ".txt"))
            if not os.path.exists(temp_path):
                error_msg = "Error. File not found: {}\n".format(temp_path)
                error(error_msg)
            lines_list      = self.file_lines_to_list(txt_file)
            bounding_boxes  = []
            is_difficult    = False
            already_seen_classes = []
            for line in lines_list:
                try:
                    if "difficult" in line:
                        class_name, left, top, right, bottom, _difficult = line.split()
                        is_difficult = True
                    else:
                        class_name, left, top, right, bottom = line.split()
                except:
                    if "difficult" in line:
                        line_split  = line.split()
                        _difficult  = line_split[-1]
                        bottom      = line_split[-2]
                        right       = line_split[-3]
                        top         = line_split[-4]
                        left        = line_split[-5]
                        class_name  = ""
                        for name in line_split[:-5]:
                            class_name += name + " "
                        class_name  = class_name[:-1]
                        is_difficult = True
                    else:
                        line_split  = line.split()
                        bottom      = line_split[-1]
                        right       = line_split[-2]
                        top         = line_split[-3]
                        left        = line_split[-4]
                        class_name  = ""
                        for name in line_split[:-4]:
                            class_name += name + " "
                        class_name = class_name[:-1]

                bbox = left + " " + top + " " + right + " " + bottom
                if is_difficult:
                    bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False, "difficult":True})
                    is_difficult = False
                else:
                    bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False})
                    if class_name in gt_counter_per_class:
                        gt_counter_per_class[class_name] += 1
                    else:
                        gt_counter_per_class[class_name] = 1

                    if class_name not in already_seen_classes:
                        if class_name in counter_images_per_class:
                            counter_images_per_class[class_name] += 1
                        else:
                            counter_images_per_class[class_name] = 1
                        already_seen_classes.append(class_name)

            with open(TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json", 'w') as outfile:
                json.dump(bounding_boxes, outfile)

        gt_classes  = list(gt_counter_per_class.keys())
        gt_classes  = sorted(gt_classes)
        n_classes   = len(gt_classes)

        dr_files_list = glob.glob(DR_PATH + '/*.txt')
        dr_files_list.sort()
        for class_index, class_name in enumerate(gt_classes):
            bounding_boxes = []
            for txt_file in dr_files_list:
                file_id = txt_file.split(".txt",1)[0]
                file_id = os.path.basename(os.path.normpath(file_id))
                temp_path = os.path.join(GT_PATH, (file_id + ".txt"))
                if class_index == 0:
                    if not os.path.exists(temp_path):
                        error_msg = "Error. File not found: {}\n".format(temp_path)
                        error(error_msg)
                lines = self.file_lines_to_list(txt_file)
                for line in lines:
                    try:
                        tmp_class_name, confidence, left, top, right, bottom = line.split()
                    except:
                        line_split      = line.split()
                        bottom          = line_split[-1]
                        right           = line_split[-2]
                        top             = line_split[-3]
                        left            = line_split[-4]
                        confidence      = line_split[-5]
                        tmp_class_name  = ""
                        for name in line_split[:-5]:
                            tmp_class_name += name + " "
                        tmp_class_name  = tmp_class_name[:-1]

                    if tmp_class_name == class_name:
                        bbox = left + " " + top + " " + right + " " +bottom
                        bounding_boxes.append({"confidence":confidence, "file_id":file_id, "bbox":bbox})

            bounding_boxes.sort(key=lambda x:float(x['confidence']), reverse=True)
            with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
                json.dump(bounding_boxes, outfile)

        sum_AP = 0.0
        ap_dictionary = {}
        lamr_dictionary = {}
        with open(RESULTS_FILES_PATH + "/results.txt", 'w') as results_file:
            results_file.write("# AP and precision/recall per class\n")
            count_true_positives = {}

            for class_index, class_name in enumerate(gt_classes):
                count_true_positives[class_name] = 0
                dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
                dr_data = json.load(open(dr_file))

                nd          = len(dr_data)
                tp          = [0] * nd
                fp          = [0] * nd
                score       = [0] * nd
                score_threhold_idx = 0
                for idx, detection in enumerate(dr_data):
                    file_id     = detection["file_id"]
                    score[idx]  = float(detection["confidence"])
                    if score[idx] >= score_threhold:
                        score_threhold_idx = idx

                    if show_animation:
                        ground_truth_img = glob.glob1(IMG_PATH, file_id + ".*")
                        if len(ground_truth_img) == 0:
                            error("Error. Image not found with id: " + file_id)
                        elif len(ground_truth_img) > 1:
                            error("Error. Multiple image with id: " + file_id)
                        else:
                            img = cv2.imread(IMG_PATH + "/" + ground_truth_img[0])
                            img_cumulative_path = RESULTS_FILES_PATH + "/images/" + ground_truth_img[0]
                            if os.path.isfile(img_cumulative_path):
                                img_cumulative = cv2.imread(img_cumulative_path)
                            else:
                                img_cumulative = img.copy()
                            bottom_border = 60
                            BLACK = [0, 0, 0]
                            img = cv2.copyMakeBorder(img, 0, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)

                    gt_file             = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
                    ground_truth_data   = json.load(open(gt_file))
                    ovmax       = -1
                    gt_match    = -1
                    bb          = [float(x) for x in detection["bbox"].split()]
                    for obj in ground_truth_data:
                        if obj["class_name"] == class_name:
                            bbgt    = [ float(x) for x in obj["bbox"].split() ]
                            bi      = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                            iw      = bi[2] - bi[0] + 1
                            ih      = bi[3] - bi[1] + 1
                            if iw > 0 and ih > 0:
                                ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                                ov = iw * ih / ua
                                if ov > ovmax:
                                    ovmax = ov
                                    gt_match = obj

                    if show_animation:
                        status = "NO MATCH FOUND!" 
                        
                    min_overlap = MINOVERLAP
                    if ovmax >= min_overlap:
                        if "difficult" not in gt_match:
                            if not bool(gt_match["used"]):
                                tp[idx] = 1
                                gt_match["used"] = True
                                count_true_positives[class_name] += 1
                                with open(gt_file, 'w') as f:
                                        f.write(json.dumps(ground_truth_data))
                                if show_animation:
                                    status = "MATCH!"
                            else:
                                fp[idx] = 1
                                if show_animation:
                                    status = "REPEATED MATCH!"
                    else:
                        fp[idx] = 1
                        if ovmax > 0:
                            status = "INSUFFICIENT OVERLAP"

                    """
                    Draw image to show animation
                    """
                    if show_animation:
                        height, widht = img.shape[:2]
                        white           = (255,255,255)
                        light_blue      = (255,200,100)
                        green           = (0,255,0)
                        light_red       = (30,30,255)
                        margin          = 10
                        # 1nd line
                        v_pos           = int(height - margin - (bottom_border / 2.0))
                        text            = "Image: " + ground_truth_img[0] + " "
                        img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
                        text            = "Class [" + str(class_index) + "/" + str(n_classes) + "]: " + class_name + " "
                        img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), light_blue, line_width)
                        if ovmax != -1:
                            color       = light_red
                            if status   == "INSUFFICIENT OVERLAP":
                                text    = "IoU: {0:.2f}% ".format(ovmax*100) + "< {0:.2f}% ".format(min_overlap*100)
                            else:
                                text    = "IoU: {0:.2f}% ".format(ovmax*100) + ">= {0:.2f}% ".format(min_overlap*100)
                                color   = green
                            img, _ = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)
                        # 2nd line
                        v_pos           += int(bottom_border / 2.0)
                        rank_pos        = str(idx+1)
                        text            = "Detection #rank: " + rank_pos + " confidence: {0:.2f}% ".format(float(detection["confidence"])*100)
                        img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
                        color           = light_red
                        if status == "MATCH!":
                            color = green
                        text            = "Result: " + status + " "
                        img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        if ovmax > 0: 
                            bbgt = [ int(round(float(x))) for x in gt_match["bbox"].split() ]
                            cv2.rectangle(img,(bbgt[0],bbgt[1]),(bbgt[2],bbgt[3]),light_blue,2)
                            cv2.rectangle(img_cumulative,(bbgt[0],bbgt[1]),(bbgt[2],bbgt[3]),light_blue,2)
                            cv2.putText(img_cumulative, class_name, (bbgt[0],bbgt[1] - 5), font, 0.6, light_blue, 1, cv2.LINE_AA)
                        bb = [int(i) for i in bb]
                        cv2.rectangle(img,(bb[0],bb[1]),(bb[2],bb[3]),color,2)
                        cv2.rectangle(img_cumulative,(bb[0],bb[1]),(bb[2],bb[3]),color,2)
                        cv2.putText(img_cumulative, class_name, (bb[0],bb[1] - 5), font, 0.6, color, 1, cv2.LINE_AA)

                        cv2.imshow("Animation", img)
                        cv2.waitKey(20) 
                        output_img_path = RESULTS_FILES_PATH + "/images/detections_one_by_one/" + class_name + "_detection" + str(idx) + ".jpg"
                        cv2.imwrite(output_img_path, img)
                        cv2.imwrite(img_cumulative_path, img_cumulative)

                cumsum = 0
                for idx, val in enumerate(fp):
                    fp[idx] += cumsum
                    cumsum += val
                    
                cumsum = 0
                for idx, val in enumerate(tp):
                    tp[idx] += cumsum
                    cumsum += val

                rec = tp[:]
                for idx, val in enumerate(tp):
                    rec[idx] = float(tp[idx]) / np.maximum(gt_counter_per_class[class_name], 1)

                prec = tp[:]
                for idx, val in enumerate(tp):
                    prec[idx] = float(tp[idx]) / np.maximum((fp[idx] + tp[idx]), 1)

                ap, mrec, mprec = self.voc_ap(rec[:], prec[:])
                F1  = np.array(rec)*np.array(prec)*2 / np.where((np.array(prec)+np.array(rec))==0, 1, (np.array(prec)+np.array(rec)))

                sum_AP  += ap
                text    = "{0:.2f}%".format(ap*100) + " = " + class_name + " AP " #class_name + " AP = {0:.2f}%".format(ap*100)

                if len(prec)>0:
                    F1_text         = "{0:.2f}".format(F1[score_threhold_idx]) + " = " + class_name + " F1 "
                    Recall_text     = "{0:.2f}%".format(rec[score_threhold_idx]*100) + " = " + class_name + " Recall "
                    Precision_text  = "{0:.2f}%".format(prec[score_threhold_idx]*100) + " = " + class_name + " Precision "
                else:
                    F1_text         = "0.00" + " = " + class_name + " F1 " 
                    Recall_text     = "0.00%" + " = " + class_name + " Recall " 
                    Precision_text  = "0.00%" + " = " + class_name + " Precision " 

                rounded_prec    = [ '%.2f' % elem for elem in prec ]
                rounded_rec     = [ '%.2f' % elem for elem in rec ]
                results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n")
                
                if len(prec)>0:
                    print(text + "\t||\tscore_threhold=" + str(score_threhold) + " : " + "F1=" + "{0:.2f}".format(F1[score_threhold_idx])\
                        + " ; Recall=" + "{0:.2f}%".format(rec[score_threhold_idx]*100) + " ; Precision=" + "{0:.2f}%".format(prec[score_threhold_idx]*100))
                else:
                    print(text + "\t||\tscore_threhold=" + str(score_threhold) + " : " + "F1=0.00% ; Recall=0.00% ; Precision=0.00%")
                ap_dictionary[class_name] = ap

                n_images = counter_images_per_class[class_name]
                lamr, mr, fppi = self.log_average_miss_rate(np.array(rec), np.array(fp), n_images)
                lamr_dictionary[class_name] = lamr

                if draw_plot:
                    plt.plot(rec, prec, '-o')
                    area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
                    area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
                    plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')

                    fig = plt.gcf()
                    fig.canvas.set_window_title('AP ' + class_name)

                    plt.title('class: ' + text)
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                    axes = plt.gca()
                    axes.set_xlim([0.0,1.0])
                    axes.set_ylim([0.0,1.05]) 
                    fig.savefig(RESULTS_FILES_PATH + "/AP/" + class_name + ".png")
                    plt.cla()

                    plt.plot(score, F1, "-", color='orangered')
                    plt.title('class: ' + F1_text + "\nscore_threhold=" + str(score_threhold))
                    plt.xlabel('Score_Threhold')
                    plt.ylabel('F1')
                    axes = plt.gca()
                    axes.set_xlim([0.0,1.0])
                    axes.set_ylim([0.0,1.05])
                    fig.savefig(RESULTS_FILES_PATH + "/F1/" + class_name + ".png")
                    plt.cla()

                    plt.plot(score, rec, "-H", color='gold')
                    plt.title('class: ' + Recall_text + "\nscore_threhold=" + str(score_threhold))
                    plt.xlabel('Score_Threhold')
                    plt.ylabel('Recall')
                    axes = plt.gca()
                    axes.set_xlim([0.0,1.0])
                    axes.set_ylim([0.0,1.05])
                    fig.savefig(RESULTS_FILES_PATH + "/Recall/" + class_name + ".png")
                    plt.cla()

                    plt.plot(score, prec, "-s", color='palevioletred')
                    plt.title('class: ' + Precision_text + "\nscore_threhold=" + str(score_threhold))
                    plt.xlabel('Score_Threhold')
                    plt.ylabel('Precision')
                    axes = plt.gca()
                    axes.set_xlim([0.0,1.0])
                    axes.set_ylim([0.0,1.05])
                    fig.savefig(RESULTS_FILES_PATH + "/Precision/" + class_name + ".png")
                    plt.cla()
                    
            if show_animation:
                cv2.destroyAllWindows()
            if n_classes == 0:
                print("未检测到任何种类，请检查标签信息与get_map.py中的classes_path是否修改。")
                return 0
            results_file.write("\n# mAP of all classes\n")
            mAP     = sum_AP / n_classes
            text    = "mAP = {0:.2f}%".format(mAP*100)
            results_file.write(text + "\n")
            print(text)

        shutil.rmtree(TEMP_FILES_PATH)

        """
        Count total of detection-results
        """
        det_counter_per_class = {}
        for txt_file in dr_files_list:
            lines_list = self.file_lines_to_list(txt_file)
            for line in lines_list:
                class_name = line.split()[0]
                if class_name in det_counter_per_class:
                    det_counter_per_class[class_name] += 1
                else:
                    det_counter_per_class[class_name] = 1
        dr_classes = list(det_counter_per_class.keys())

        """
        Write number of ground-truth objects per class to results.txt
        """
        with open(RESULTS_FILES_PATH + "/results.txt", 'a') as results_file:
            results_file.write("\n# Number of ground-truth objects per class\n")
            for class_name in sorted(gt_counter_per_class):
                results_file.write(class_name + ": " + str(gt_counter_per_class[class_name]) + "\n")

        """
        Finish counting true positives
        """
        for class_name in dr_classes:
            if class_name not in gt_classes:
                count_true_positives[class_name] = 0

        """
        Write number of detected objects per class to results.txt
        """
        with open(RESULTS_FILES_PATH + "/results.txt", 'a') as results_file:
            results_file.write("\n# Number of detected objects per class\n")
            for class_name in sorted(dr_classes):
                n_det = det_counter_per_class[class_name]
                text = class_name + ": " + str(n_det)
                text += " (tp:" + str(count_true_positives[class_name]) + ""
                text += ", fp:" + str(n_det - count_true_positives[class_name]) + ")\n"
                results_file.write(text)

        """
        Plot the total number of occurences of each class in the ground-truth
        """
        if draw_plot:
            window_title = "ground-truth-info"
            plot_title = "ground-truth\n"
            plot_title += "(" + str(len(ground_truth_files_list)) + " files and " + str(n_classes) + " classes)"
            x_label = "Number of objects per class"
            output_path = RESULTS_FILES_PATH + "/ground-truth-info.png"
            to_show = False
            plot_color = 'forestgreen'
            draw_plot_func(
                gt_counter_per_class,
                n_classes,
                window_title,
                plot_title,
                x_label,
                output_path,
                to_show,
                plot_color,
                '',
                )

        # """
        # Plot the total number of occurences of each class in the "detection-results" folder
        # """
        # if draw_plot:
        #     window_title = "detection-results-info"
        #     # Plot title
        #     plot_title = "detection-results\n"
        #     plot_title += "(" + str(len(dr_files_list)) + " files and "
        #     count_non_zero_values_in_dictionary = sum(int(x) > 0 for x in list(det_counter_per_class.values()))
        #     plot_title += str(count_non_zero_values_in_dictionary) + " detected classes)"
        #     # end Plot title
        #     x_label = "Number of objects per class"
        #     output_path = RESULTS_FILES_PATH + "/detection-results-info.png"
        #     to_show = False
        #     plot_color = 'forestgreen'
        #     true_p_bar = count_true_positives
        #     draw_plot_func(
        #         det_counter_per_class,
        #         len(det_counter_per_class),
        #         window_title,
        #         plot_title,
        #         x_label,
        #         output_path,
        #         to_show,
        #         plot_color,
        #         true_p_bar
        #         )

        """
        Draw log-average miss rate plot (Show lamr of all classes in decreasing order)
        """
        if draw_plot:
            window_title = "lamr"
            plot_title = "log-average miss rate"
            x_label = "log-average miss rate"
            output_path = RESULTS_FILES_PATH + "/lamr.png"
            to_show = False
            plot_color = 'royalblue'
            draw_plot_func(
                lamr_dictionary,
                n_classes,
                window_title,
                plot_title,
                x_label,
                output_path,
                to_show,
                plot_color,
                ""
                )

        """
        Draw mAP plot (Show AP's of all classes in decreasing order)
        """
        if draw_plot:
            window_title = "mAP"
            plot_title = "mAP = {0:.2f}%".format(mAP*100)
            x_label = "Average Precision"
            output_path = RESULTS_FILES_PATH + "/mAP.png"
            to_show = True
            plot_color = 'royalblue'
            draw_plot_func(
                ap_dictionary,
                n_classes,
                window_title,
                plot_title,
                x_label,
                output_path,
                to_show,
                plot_color,
                ""
                )
        return mAP
    def file_lines_to_list(self,path):
        # open txt file lines to a list
        with open(path) as f:
            content = f.readlines()
        # remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]
        return content
    def voc_ap(self,rec, prec):
        """
        --- Official matlab code VOC2012---
        mrec=[0 ; rec ; 1];
        mpre=[0 ; prec ; 0];
        for i=numel(mpre)-1:-1:1
                mpre(i)=max(mpre(i),mpre(i+1));
        end
        i=find(mrec(2:end)~=mrec(1:end-1))+1;
        ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
        """
        rec.insert(0, 0.0) # insert 0.0 at begining of list
        rec.append(1.0) # insert 1.0 at end of list
        mrec = rec[:]
        prec.insert(0, 0.0) # insert 0.0 at begining of list
        prec.append(0.0) # insert 0.0 at end of list
        mpre = prec[:]
        """
        This part makes the precision monotonically decreasing
            (goes from the end to the beginning)
            matlab: for i=numel(mpre)-1:-1:1
                        mpre(i)=max(mpre(i),mpre(i+1));
        """
        for i in range(len(mpre)-2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i+1])
        """
        This part creates a list of indexes where the recall changes
            matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
        """
        i_list = []
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i-1]:
                i_list.append(i) # if it was matlab would be i + 1
        """
        The Average Precision (AP) is the area under the curve
            (numerical integration)
            matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
        """
        ap = 0.0
        for i in i_list:
            ap += ((mrec[i]-mrec[i-1])*mpre[i])
        return ap, mrec, mpre
    def log_average_miss_rate(self,precision, fp_cumsum, num_images):
        """
            log-average miss rate:
                Calculated by averaging miss rates at 9 evenly spaced FPPI points
                between 10e-2 and 10e0, in log-space.

            output:
                    lamr | log-average miss rate
                    mr | miss rate
                    fppi | false positives per image

            references:
                [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
                State of the Art." Pattern Analysis and Machine Intelligence, IEEE
                Transactions on 34.4 (2012): 743 - 761.
        """

        if precision.size == 0:
            lamr = 0
            mr = 1
            fppi = 0
            return lamr, mr, fppi

        fppi = fp_cumsum / float(num_images)
        mr = (1 - precision)

        fppi_tmp = np.insert(fppi, 0, -1.0)
        mr_tmp = np.insert(mr, 0, 1.0)

        ref = np.logspace(-2.0, 0.0, num = 9)
        for i, ref_i in enumerate(ref):
            j = np.where(fppi_tmp <= ref_i)[-1][-1]
            ref[i] = mr_tmp[j]

        lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

        return lamr, mr, fppi

    def get_coco_map(self,class_names, path):
        GT_PATH     = os.path.join(path, 'ground-truth')
        DR_PATH     = os.path.join(path, 'detection-results')
        COCO_PATH   = os.path.join(path, 'coco_eval')

        if not os.path.exists(COCO_PATH):
            os.makedirs(COCO_PATH)

        GT_JSON_PATH = os.path.join(COCO_PATH, 'instances_gt.json')
        DR_JSON_PATH = os.path.join(COCO_PATH, 'instances_dr.json')

        with open(GT_JSON_PATH, "w") as f:
            results_gt  = preprocess_gt(GT_PATH, class_names)
            json.dump(results_gt, f, indent=4)

        with open(DR_JSON_PATH, "w") as f:
            results_dr  = preprocess_dr(DR_PATH, class_names)
            json.dump(results_dr, f, indent=4)
            if len(results_dr) == 0:
                print("未检测到任何目标。")
                return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        cocoGt      = COCO(GT_JSON_PATH)
        cocoDt      = cocoGt.loadRes(DR_JSON_PATH)
        cocoEval    = COCOeval(cocoGt, cocoDt, 'bbox') 
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        return cocoEval.stats
image_path = "../dataset/datamix/images/"
h5_model_path = "./logs/ep030-loss0.989-val_loss0.994.h5"


output_name_list = [ "yolo_eval","yolo_eval_1", "yolo_eval_2","yolo_eval_3" ]
mnn_model_path = './best_epoch_weights.mnn'
val_path = "../dataset/datamix_test.txt"
mnn  = MNN_model(image_path,val_path,mnn_model_path,output_name_list)

# onnx_model_path = "./best_epoch_weights.onnx"
# onnx_model(onnx_model_path)
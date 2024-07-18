# 计算IoU，矩形框的坐标形式为xyxy
# (x1,y1)是矩形框左上角的坐标,(x2,y2)是矩形框右下角的坐标
import numpy as np
def box_iou_xyxy(box1, box2):
    # 获取box1左上角和右下角的坐标
    x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
    # 计算box1的面积
    s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
    # 获取box2左上角和右下角的坐标
    x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]
    # 计算box2的面积
    s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)

    # 计算相交矩形框的坐标
    xmin = np.maximum(x1min, x2min)  # 左上角的横坐标
    ymin = np.maximum(y1min, y2min)  # 左上角的纵坐标
    xmax = np.minimum(x1max, x2max)  # 右下角的横坐标
    ymax = np.minimum(y1max, y2max)  # 右下角的纵坐标

    # 计算相交矩形的高度、宽度、面积
    inter_h = np.maximum(ymax - ymin + 1., 0.)
    inter_w = np.maximum(xmax - xmin + 1., 0.)
    intersection = inter_h * inter_w

    # 计算相并面积
    union = s1 + s2 - intersection

    # 计算交并比
    iou = intersection / union
    return iou


bbox1 = [ 359,265,  541,351]
bbox2 = [ 385,285,  639,415]
iou = box_iou_xyxy(bbox1, bbox2)
print('IoU is {}'.format(iou))

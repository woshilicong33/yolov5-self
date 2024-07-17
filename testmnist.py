import numpy as np
np.set_printoptions(threshold=np.inf)
def get_near_points(x, y, i, j):
    sub_x = x - i
    sub_y = y - j
    if sub_x > 0.5 and sub_y > 0.5:
        return [[0, 0], [1, 0], [0, 1]]
    elif sub_x < 0.5 and sub_y > 0.5:
        return [[0, 0], [-1, 0], [0, 1]]
    elif sub_x < 0.5 and sub_y < 0.5:
        return [[0, 0], [-1, 0], [0, -1]]
    else:
        return [[0, 0], [1, 0], [0, -1]]
def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    #-----------------------------------------------------------#
    #   获得框的坐标和图片的大小
    #-----------------------------------------------------------#
    true_boxes  = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    print("true_boxes",true_boxes)
    print("true_boxes.shape",true_boxes.shape)
    print("input_shape",input_shape)
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    #-----------------------------------------------------------#
    #   一共有三个特征层数
    #-----------------------------------------------------------#
    num_layers  = len(anchors_mask)
    print("num_layers",num_layers)
    true_boxes = np.expand_dims(true_boxes, 0)
    #-----------------------------------------------------------#
    #   m为图片数量，grid_shapes为网格的shape
    #-----------------------------------------------------------#
    m           = true_boxes.shape[0]
    print("m",m)


    grid_shapes = [input_shape // {0:32, 1:16, 2:8}[l] for l in range(num_layers)]
    print("grid_shapes",grid_shapes)

    #-----------------------------------------------------------#
    #   y_true的格式为(m,13,13,3,85)(m,26,26,3,85)(m,52,52,3,85)
    #-----------------------------------------------------------#
    print("y_true:",grid_shapes[0])

    y_true = [np.zeros((m, grid_shapes[l], grid_shapes[l], len(anchors_mask[l]), 5 + num_classes),
                dtype='float32') for l in range(num_layers)]
    print("y_true:",y_true[0].shape)
    print("y_true:",y_true[1].shape)
    print("y_true:",y_true[2].shape)

    #-----------------------------------------------------#
    #   anchors_best_ratio
    #-----------------------------------------------------#
    box_best_ratios = [np.zeros((m, grid_shapes[l], grid_shapes[l], len(anchors_mask[l])),
                dtype='float32') for l in range(num_layers)]

    #-----------------------------------------------------------#
    #   通过计算获得真实框的中心和宽高
    #   中心点(m,n,2) 宽高(m,n,2)
    #-----------------------------------------------------------#
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh =  true_boxes[..., 2:4] - true_boxes[..., 0:2]
    print("boxes_xy:",boxes_xy)
    print("boxes_wh:",boxes_wh)

    #-----------------------------------------------------------#
    #   将真实框归一化到小数形式
    #-----------------------------------------------------------#
    true_boxes[..., 0:2] = boxes_xy / input_shape
    true_boxes[..., 2:4] = boxes_wh / input_shape
    print("boxes_xy:",true_boxes)
    print("boxes_wh:",true_boxes)

    #-----------------------------------------------------------#
    #   [9,2] -> [9,2]
    #-----------------------------------------------------------#
    anchors         = np.array(anchors, np.float32)

    #-----------------------------------------------------------#
    #   长宽要大于0才有效
    #-----------------------------------------------------------#
    valid_mask = boxes_wh[..., 0]>0
    print(valid_mask)

    for b in range(m):
        #-----------------------------------------------------------#
        #   对每一张图进行处理
        #-----------------------------------------------------------#
        wh = boxes_wh[b, valid_mask[b]]
        print("wh;",wh)

        if len(wh) == 0: continue
        #-------------------------------------------------------#
        #   wh                      : num_true_box, 2
        #   anchors                 : 9, 2
        #
        #   ratios_of_gt_anchors    : num_true_box, 9, 2
        #   ratios_of_anchors_gt    : num_true_box, 9, 2
        #
        #   ratios                  : num_true_box, 9, 4
        #   max_ratios              : num_true_box, 9
        #-------------------------------------------------------#

        ratios_of_gt_anchors = np.expand_dims(wh, 1) / np.expand_dims(anchors, 0)
        print(ratios_of_gt_anchors)
        print(np.expand_dims(wh, 1))
        print(np.expand_dims(anchors, 0))

        ratios_of_anchors_gt = np.expand_dims(anchors, 0) / np.expand_dims(wh, 1)
        ratios               = np.concatenate([ratios_of_gt_anchors, ratios_of_anchors_gt], axis = -1)
        max_ratios           = np.max(ratios, axis = -1)

        print("ratios_of_anchors_gt:",ratios_of_anchors_gt)
        print("ratios_of_gt_anchors:",ratios_of_gt_anchors)
        print("ratios:",ratios)
        print("max_ratios:",max_ratios)
       
        for t, ratio in enumerate(max_ratios):
            #-------------------------------------------------------#
            #   ratio : 9
            #-------------------------------------------------------#
            over_threshold = ratio < 4
            print("over_threshold:",over_threshold)

            over_threshold[np.argmin(ratio)] = True
            print("over_threshold:",over_threshold) 
            print("ratio:",ratio)
            print("np.argmin(ratio):",np.argmin(ratio))

            #-----------------------------------------------------------#
            #   找到每个真实框所属的特征层
            #-----------------------------------------------------------#
            for l in range(num_layers):
                print("l:",l)

                for k, n in enumerate(anchors_mask[l]):
                    if not over_threshold[n]:
                        continue
                    #-----------------------------------------------------------#
                    #   floor用于向下取整，找到真实框所属的特征层对应的x、y轴坐标
                    #-----------------------------------------------------------#
                    print("true_boxes:",true_boxes)
                    i = np.floor(true_boxes[b,t,0] * grid_shapes[l]).astype('int32')
                    print("i:",i)
                    print("grid_shapes[l][1]:",grid_shapes[l])
                    print("true_boxes[b,t,0]:",true_boxes[b,t,0])  
                 
                    j = np.floor(true_boxes[b,t,1] * grid_shapes[l]).astype('int32')
                    print("j:",j)
                    print("grid_shapes[l][1]:",grid_shapes[l])
                    print("true_boxes[b,t,0]:",true_boxes[b,t,1])  

                    offsets = get_near_points(true_boxes[b,t,0] * grid_shapes[l], true_boxes[b,t,1] * grid_shapes[l], i, j)
                    print("offsets;",offsets)

                    for offset in offsets:
                        local_i = i + offset[0]
                        local_j = j + offset[1]
                        print("local_i:",local_i)
                        print("local_j:",local_j)
                        print("grid_shapes[l]:",grid_shapes[l])
                        if local_i >= grid_shapes[l] or local_i < 0 or local_j >= grid_shapes[l] or local_j < 0:

                            continue

                        if box_best_ratios[l][b, local_j, local_i, k] != 0:

                            if box_best_ratios[l][b, local_j, local_i, k] > ratio[n]:
                                y_true[l][b, local_j, local_i, k, :] = 0
                            else:
                                continue

                        #-----------------------------------------------------------#
                        #   c指的是当前这个真实框的种类
                        #-----------------------------------------------------------#
                        c = true_boxes[b, t, 4].astype('int32')
                        #-----------------------------------------------------------#
                        #   y_true的shape为(m,13,13,3,85)(m,26,26,3,85)(m,52,52,3,85)
                        #   最后的85可以拆分成4+1+80，4代表的是框的中心与宽高、
                        #   1代表的是置信度、80代表的是种类
                        #-----------------------------------------------------------#
                        y_true[l][b, local_j, local_i, k, 0:4] = true_boxes[b, t, 0:4]
                        y_true[l][b, local_j, local_i, k, 4] = 1
                        y_true[l][b, local_j, local_i, k, 5+c] = 1
                        box_best_ratios[l][b, local_j, local_i, k] = ratio[n]

                        print("y_true;",len(y_true))
                        print("y_true;",np.shape(y_true[0]))
                        # print("y_true;",len(y_true[1]))
                        # exit()
    return y_true
line = ["/yolo/compil/yolov5-self/dataset/datamix/images/2024_04_30_11_42_01_656.jpg2024-05-29-01:36:18.991027.jpg", "538,259,556,277,1","195,72,221,86,1"]
box= np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

box_data = np.zeros((10,5))
box_data[:len(box)] = box
print(box_data)

input_shape = 480
anchors = [[10,13],[16,30], [33,23],  [30,61], [62,45], [59,119],  [116,90], [156,198], [373,326]]

preprocess_true_boxes(box_data, input_shape, anchors, 3)
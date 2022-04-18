# coding=utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import tensorflow as tf
import tool

def model_init(model_path , img_path):
    tflite_model = tf.lite.Interpreter(model_path)
    tflite_model.allocate_tensors()
    tflife_input_details = tflite_model.get_input_details()[0]
    tflite_model.set_tensor(tflife_input_details['index'], data_input(img_path))
    return tflite_model

def model_run(tflite_model):
    tflite_model.invoke()
    tflife_output_details = tflite_model.get_output_details()[0]
    output_tflite = tflite_model.get_tensor(tflife_output_details['index'])[0]
    #print("output 0 data: ", output_tflite)
    yy_bboxes = output_tflite
    tflife_output_details = tflite_model.get_output_details()[1]
    output_tflite = tflite_model.get_tensor(tflife_output_details['index'])[0]
    #print("output 1 data: ", output_tflite)
    y_scores = output_tflite
    return yy_bboxes, y_scores

def data_input(img_path):
    img_raw = cv2.imread(str(img_path))
    #frame = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY) #彩色转灰色
    frame = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB) #彩色转RGB
    frame = cv2.resize(frame, (260, 260), cv2.INTER_AREA) # 改变尺寸  
    frame = np.expand_dims(frame, 0)	# 扩展dim=0维度
    #frame = np.expand_dims(frame, 3)	# 灰度图需要扩展dim=3维度
    frame = frame/255.0 #做归一化
    return np.float32(frame)		# 类型转为float32  

def main():
    conf_thresh=0.5
    iou_thresh=0.4
    target_shape=[260, 260]
    draw_result=True
    show_result=True
    output_info = []
    height, width=target_shape

    # anchor configuration
    feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
    anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
    anchor_ratios = [[1, 0.62, 0.42]] * 5
    # generate anchors
    anchors = tool.generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)
    # for inference , the batch size is 1, the model output shape is [1, N, 4],
    # so we expand dim for anchors to [1, anchor_num, 4]
    anchors_exp = np.expand_dims(anchors, axis=0)
    # id
    id2class = {0: 'Mask', 1: 'NoMask'}

    #img_path = '../img1_260x260.jpg'
    img_path = '../me.jpg'
    img_path = '../test1.jpg'
    img_path = '../test2.jpg'
    model_path = '../Model/face_mask_detection.tflite'
    img_raw = cv2.imread(str(img_path))
    height = img_raw.shape[0]
    width = img_raw.shape[1]
    #image = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB) #彩色转RGB
    image = cv2.resize(img_raw, (260, 260), cv2.INTER_AREA) # 改变尺寸  

    model = model_init(model_path, img_path)
    y_bboxes_output, y_cls_output = model_run(model)
    y_bboxes_output = np.expand_dims(y_bboxes_output, axis=0)
    y_cls_output = np.expand_dims(y_cls_output, axis=0)
    y_bboxes = tool.decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)    
    # keep_idx is the alive bounding box after nms.
    keep_idxs = tool.single_class_non_max_suppression(y_bboxes,
                                                 bbox_max_scores,
                                                 conf_thresh=conf_thresh,
                                                 iou_thresh=iou_thresh,
                                                 )
    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)

        if draw_result:
            if class_id == 0:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            cv2.rectangle(img_raw, (xmin, ymin), (xmax, ymax), color, 2)
            print("xmin=", xmin, "ymin=", ymin, "xmax=", xmax, "ymax=", ymax)
            cv2.putText(img_raw, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
        output_info.append([class_id, conf, xmin, ymin, xmax, ymax])
    cv2.imshow('image', img_raw)
    cv2.waitKey(10000)

if __name__ == "__main__":
    main()
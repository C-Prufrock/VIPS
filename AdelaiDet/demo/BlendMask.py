# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import argparse
import glob
import multiprocessing as mp
import matplotlib.pyplot as plt
import atexit
import bisect

import time
import cv2
import tqdm
import torch
import numpy as np

from detectron2.data.detection_utils import read_image

from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from adet.utils.visualizer import TextVisualizer
from collections import deque
from adet.config import get_cfg

##print("I can work with cv::imshow")


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        ##print(self.metadata)
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.vis_text = cfg.MODEL.ROI_HEADS.NAME == "TextHead"

        self.parallel = parallel

        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            #print("VisualizationDemo works here")
            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        #print(predictions)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        if self.vis_text:
            visualizer = TextVisualizer(image, self.metadata, instance_mode=self.instance_mode)
            
        else:
            visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
            print("visualizer._instance_mode is :",visualizer._instance_mode)
            print("alpha is :",visualizer.overlay_instances)
          
        ##下列if语句辅助输出vis_output
        if "bases" in predictions:
            self.vis_bases(predictions["bases"])
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device))
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

ROOT_DIR = "/home/lxy/SLAM/VIO/VINS-Course-master/AdelaiDet/demo/"
#print("ROOT_DIR is : ",ROOT_DIR)

#定义读取参数的函数；
def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    #print("setup_cfg(args): ends running")
    return cfg

class args():
    def __init__(self,config_file,confidence_threshold,opts,ImageInput=None):
        self.config_file = config_file
        self.opts = opts
        self.confidence_threshold=confidence_threshold
        self.input = ImageInput

class Blendmask:
    def __init__(self):
        print('Initializing BlendMask network')
        mp.set_start_method("spawn", force=True)
    ## load blendmaks相关参数；
        Args = args('/home/lxy/SLAM/VIO/VINS-Course-master/AdelaiDet/configs/BlendMask/R_101_dcni3_5x.yaml',0.35,['MODEL.WEIGHTS', '/home/lxy/SLAM/VIO/VINS-Course-master/AdelaiDet/demo/blendmask_r101_dcni3_5x.pth'])
        #Args.input = os.path.join(ROOT_DIR,'datasets/coco/000061.png')
        #print(Args.config_file)
        #print(Args.opts)
        #print(Args.confidence_threshold)
        self.cfg = setup_cfg(Args)
        #print(self.cfg.MODEL.RETINANET.SCORE_THRESH_TEST)
        self.demo = VisualizationDemo(self.cfg)
        print('Initialated BLendMask network')

    def GetDynSeg(self,array):
        a = array[:, 0:len(array[0] - 2):3]
        b = array[:, 1:len(array[0] - 2):3]
        c = array[:, 2:len(array[0] - 2):3]
        a = a[:, :, None]
        b = b[:, :, None]
        c = c[:, :, None]
        m = np.concatenate((a, b, c), axis=2)
        predictions, visualized_output = self.demo.run_on_image(m)
        pred_masks=predictions['instances'].pred_masks
        channel = pred_masks.shape[0]
        h = pred_masks.shape[1]
        w = pred_masks.shape[2]
        Integration_mask = np.zeros((h,w))

        i=0
        for exi_clas in predictions['instances'].pred_classes:
            if  exi_clas == 0:
                Integration_mask += pred_masks[i].cpu().numpy()
            if  exi_clas == 1:
                Integration_mask += pred_masks[i].cpu().numpy()
            if  exi_clas == 2:
                Integration_mask += pred_masks[i].cpu().numpy()
            i+=1
        return Integration_mask

    def GetInsSeg(self, array):
        a = array[:, 0:len(array[0] - 2):3]
        b = array[:, 1:len(array[0] - 2):3]
        c = array[:, 2:len(array[0] - 2):3]
        a = a[:, :, None]
        b = b[:, :, None]
        c = c[:, :, None]
        m = np.concatenate((a, b, c), axis=2)
        predictions, visualized_output =  self.demo.run_on_image(m)
        print("here works")
        pred_masks=predictions['instances'].pred_masks
        pred_masks_result = []
        pred_boxes = predictions['instances'].pred_boxes

        #print(pred_masks.cpu().numpy().shape[0])
        a = pred_masks.cpu().numpy().shape[0]
        print("here works 1")
        print(pred_boxes.tensor)
        for x in range(a):
            print(x)
            Integration_mask = pred_masks[x].cpu().numpy()
            pred_masks_result.append(Integration_mask)
            #pred_boxes_tensor = predictions['instance'].pred_boxes[x].cpu().numpy().tolist()
            #pred_boxes.append(pred_boxes_tensor)
        #return [pred_boxes]
        print("here works 2")
        return [pred_masks_result,predictions['instances'].pred_classes.cpu().numpy().tolist(),pred_boxes.tensor.cpu().numpy().tolist()]
        #return [,,pred_boxes]
    ##return predictions['instances'].pred_classes




'''
def GetDynSeg(self,array):
    a = array[:, 0:len(array[0] - 2):3]
    b = array[:, 1:len(array[0] - 2):3]
    c = array[:, 2:len(array[0] - 2):3]
    a = a[:, :, None]
    b = b[:, :, None]
    c = c[:, :, None]
    m = np.concatenate((a, b, c), axis=2)
    predictions, visualized_output = self.demo.run_on_image(m)
    pred_masks=predictions['instances'].pred_masks
    channel = pred_masks.shape[0]
    h = pred_masks.shape[1]
    w = pred_masks.shape[2]
    Integration_mask = np.zeros((h,w))

    i=0
    for exi_clas in predictions['instances'].pred_classes:
        if  exi_clas == 0:
            Integration_mask += pred_masks[i].cpu().numpy()
        if  exi_clas == 1:
            Integration_mask += pred_masks[i].cpu().numpy()
        if  exi_clas == 2:
            Integration_mask += pred_masks[i].cpu().numpy()
        i+=1
        print("let's know the type of predictions['instances'].pred_classes",predictions['instances'].pred_classes.type())

    pred_classes = predictions['instances'].pred_classes.cpu().numpy()
    print(pred_classes)
    return Integration_mask
'''

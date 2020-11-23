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
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from adet.utils.visualizer import TextVisualizer

from adet.config import get_cfg
from collections import deque

print("you work me")

# constants
WINDOW_NAME = "Python-C Test"


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
        print(self.metadata)
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.vis_text = cfg.MODEL.ROI_HEADS.NAME == "TextHead"

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
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
    return cfg

class args():
    def __init__(self,config_file,confidence_threshold,opts,ImageInput=None):
        self.config_file = config_file
        self.opts = opts
        self.confidence_threshold=confidence_threshold
        self.input = ImageInput

ROOT_DIR = "/home/lxy/SLAM/VIO/VINS-Course-master/AdelaiDet/demo/"




print("ROOT_DIR is : ",ROOT_DIR)
class BlendMask:
    def __init__(self):
        print('Initializeing BlendMask network')
        ##mp.set_start_method("spawn", force=True)
    ## load blendmaks相关参数；
        Args = args('configs/BlendMask/R_101_dcni3_5x.yaml',0.35,['MODEL.WEIGHTS', 'blendmask_r101_dcni3_5x.pth'])
        #Args.input = os.path.join(ROOT_DIR,'datasets/coco/000061.png')
        print(Args.input)
        logger = setup_logger()
        logger.info("Arguments: " + str(Args))
        self.cfg = setup_cfg(Args)
        self.demo = VisualizationDemo(cfg)
        print('Initialated BLendMask network')

    #demo = VisualizationDemo(cfg)
    ##注意此处的Args.input为相对路径；可考虑更改为绝对路径；
    def GetInstanceSeg(self,image):
        predictions, visualized_output = self.demo.run_on_image(image)
        print("here works")
        cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
        return visualized_output.get_image()


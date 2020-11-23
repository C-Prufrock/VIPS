# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from adet.config import get_cfg

# constants
WINDOW_NAME = "Python-C Test"

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
class BlendMask:
    def __init__(self):
        print('Initializeing BlendMask network')
        ##mp.set_start_method("spawn", force=True)
        
        print("ROOT_DIR is : ",ROOT_DIR)

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
        predictions, visualized_output = self.demo.run_on_image(img)
        logger.info(
                "{}: detected {} instances in {:.2f}s".format(
                    path, len(predictions["instances"]), time.time() - start_time
                )
            )
        cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
        return visualized_output.get_image()
        

    
    




    

    
    

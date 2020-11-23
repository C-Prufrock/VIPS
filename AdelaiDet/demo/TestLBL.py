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
WINDOW_NAME = "COCO detections"


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



def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


##需要把主函数封装为一个函数；
if __name__ == "__main__":

##设置进程启动的方式；
    mp.set_start_method("spawn", force=True)

    args = get_parser().parse_args()

    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
#VisualizationDemo函数的主要运行函数为DefaultPredictor,该函数来自于detectron2.engine.defaults;
#1. 该函数的作用在于load checkpoints from ‘cfg.MODEL.WEIGHTS’.
#2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
#3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
#4. Take one input image and produce a single output, instead of a batch.
#Examples:
#    ::
#        pred = DefaultPredictor(cfg)
#        inputs = cv2.imread("input.jpg")
#        outputs = pred(inputs)
#`    """
    #输出输入的参数信息；
    #print(args.input[0])
    if args.input:
        if os.path.isdir(args.input[0]):
            args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
        elif len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")

            print("------type(img):------",img.shape)
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)   
            #显示实例分割数目以及，时间；
            logger.info(
                "{}: detected {} instances in {:.2f}s".format(
                    path, len(predictions["instances"]), time.time() - start_time
                )
            )
            #type(predictions["instances"])
            #print(predictions['instances'].pred_masks)

            pred_masks=predictions['instances'].pred_masks
            print(pred_masks.shape)
            print(predictions['instances'].scores)
            print(predictions['instances'].pred_classes)
            #print(predictions['instances'].pred_boxes)

            ##显示实例分割图像；
            cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
            #整合具有运动物体的mask
            c = pred_masks.shape[0]
            h = pred_masks.shape[1]
            w = pred_masks.shape[2]

            Integration_mask = np.zeros((h,w))

            '''
            print("predictions['instances'].pred_classes is : ", type(predictions['instances'].pred_classes[0]))
            if predictions['instances'].pred_classes[0] == 2:
                print("predictions['instances'].pred_classes[0]) is just a number")
            '''
            #只识别人，自行车，car三种0,1,2；
            #tensor to numpy过程中，gpu的数据不可以直接转换，需要加.cpu()
            i=0
            for exi_clas in predictions['instances'].pred_classes:
                if  exi_clas == 0:
                    Integration_mask += pred_masks[i].cpu().numpy()
                if  exi_clas == 1:
                    Integration_mask += pred_masks[i].cpu().numpy()
                if  exi_clas == 2:
                    Integration_mask += pred_masks[i].cpu().numpy()
                i+=1

            plt.imshow(Integration_mask)
            plt.show()
           
            
     
            
          
            
            
            '''
            

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
            '''

    

    
    

//
// Created by lxy on 20-8-27.
//
#pragma once

#ifndef VINS_ESTIMATOR_INSTANCESEG_H
#define VINS_ESTIMATOR_INSTANCESEG_H

#include <python3.6/Python.h>
#include<iostream>
#include<fstream>
#include<iomanip>
#include<dirent.h>
#include<errno.h>
#include<opencv2/opencv.hpp>
#include<vector>
#include<cstdio>
#include<boost/thread.hpp>
#include<string>
#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include "ndarrayobject.h"
#include <numpy/arrayobject.h>
#include<Conversion.h>

class Ins_Seg_result{
public:
    std::vector<cv::Mat> InsSeg_Mask;
    std::vector<int> MaskId;
    std::vector<std::vector<double>>pred_boxes;
};

class InstanceSeg {

    private:
        NDArrayConverter *cvt; 	/*!< Converter to NumPy Array from cv::Mat */
        PyObject *py_module; 	/*!< Module of python where the Mask algorithm is implemented */
        PyObject *py_class; 	/*!< Class to be instanced */
        PyObject *net; 			/*!< Instance of the class */
        PyObject *pFunc;
        PyObject *InsSeg;

        std::string py_path; 	/*!< Path to be included to the environment variable PYTHONPATH */
        std::string module_name; /*!< Detailed description after the member */
        std::string class_name; /*!< Detailed description after the member */
        std::string get_dyn_seg; 	/*!< Detailed description after the member */
        std::string get_Ins_seg;
        void ImportSettings();
    public:

        InstanceSeg();
        ~InstanceSeg();
        cv::Mat GetDynSeg(cv::Mat &image, std::string dir="no_save", std::string rgb_name="no_file");
        Ins_Seg_result GetInsSeg(cv::Mat &image, std::string dir = "no_save",std::string rgb_name = "no_file");
};




#endif //VINS_ESTIMATOR_INSTANCESEG_H

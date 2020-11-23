/**
* This file is part of DynaSLAM.
* Copyright (C) 2018 Berta Bescos <bbescos at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/bertabescos/DynaSLAM>.
*
*/


#include <python3.6/Python.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#include "ndarrayobject.h"
#include <numpy/arrayobject.h>

//#include "__multiarray_api.h"

    class NDArrayConverter {
    private:
        void* init();
    public:
        NDArrayConverter();

        std::vector<cv::Mat> toMatVec(PyObject* o);
        std::vector<int> toIdVec(PyObject* o);
        std::vector<std::vector<double>> toVecBox(PyObject* o);
        cv::Mat toMat(PyObject* o);

        PyObject* mat2Pyobejct(cv::Mat mat);

    };

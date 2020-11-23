/**
* This file is part of DynaSLAM.
* Copyright (C) 2018 Berta Bescos <bbescos at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/bertabescos/DynaSLAM>.
*
*/


#include "Conversion.h"
#include <iostream>
#include<vector>
NDArrayConverter::NDArrayConverter() { init(); }

using namespace std;

void* NDArrayConverter::init()
{
    import_array();
}


PyObject* NDArrayConverter::mat2Pyobejct(cv::Mat Img){
    //读取mat数据，并进行封装；
    int m,n;
  //std::cout<<"img.cols is "<< Img.cols<<std::endl;
  //std::cout<<"img.ross is "<< Img.rows<<std::endl;
    n = Img.cols*3;
    m = Img.rows;
    unsigned  char *data = (unsigned char*)malloc(sizeof(unsigned char*)*m*n);
    int p=0;
    for(int i=0;i<m;i++) {
        for (int j = 0; j < n; j++) {
            data[p] = Img.at<unsigned char>(i, j);
            //std::cout<<"value in img: "<< img.at<unsigned char>(i,j);
            p++;
        }
    }

    npy_intp Dims[2]={m,n};

    //导入PyObject；
    PyObject*PyArray = PyArray_SimpleNewFromData(2, Dims, NPY_UBYTE, data);
    //std::cout<<"here works"<<std::endl;

    PyObject *ArgArray = PyTuple_New(1);
    PyObject *arg = PyLong_FromLong(10);
    PyTuple_SetItem(ArgArray, 0, PyArray);
    PyTuple_SetItem(ArgArray, 1, arg);
    //cout<<"数据被准确的转换为PyObject了"<<endl;
    return ArgArray;
};

//负责把一个包含多个mat矩阵的PyList转换为一个cv::Mat的vector；
std::vector<cv::Mat> NDArrayConverter::toMatVec(PyObject* o) {
    int SizeOfList = PyList_Size(o);
    vector<cv::Mat>Ins_Mask;
    for (int channels = 0; channels < SizeOfList; channels++) {
        //std::cout << "channels is " << channels << std::endl;
        PyArrayObject *MaskItem;
        PyObject * mask = PyList_GetItem(o, channels);
        PyArray_OutputConverter(mask, &MaskItem);
        //int typenum = PyArray_TYPE(MaskItem);
        int Rows = MaskItem->dimensions[0], columns = MaskItem->dimensions[1];
        //cout << "Rows is " << Rows << endl;
        //cout << "colums us " << columns << endl;
        //std::cout << "The " << channels << "th Array is:" << std::endl;
        cv::Mat Mask = cv::Mat::zeros(Rows, columns, CV_32F);
       // cout << "type of Mask is " << Mask.type() << endl;
        int count = 0;
        //cout << "strides[0]" << MaskItem->strides[0] << endl;
        //cout << "strides[1]" << MaskItem->strides[1] << endl;
        for (int Index_m = 0; Index_m < Rows; Index_m++) {

            for (int Index_n = 0; Index_n < columns; Index_n++) {
                //cout << *(double *) (MaskItem->data + Index_m * MaskItem->strides[0] + Index_n * MaskItem->strides[1])
                     //<< " ";
                if (*(double *) (MaskItem->data + Index_m * MaskItem->strides[0] + Index_n * MaskItem->strides[1]) !=
                    0.0) {
                    Mask.at<float>(Index_m, Index_n) = 255.0;
                    count++;
                }
                //Mask.at//cout<<*(double *)(MaskItem->data + Index_m * MaskItem->strides[0] + Index_n *MaskItem->strides[1])<<" ";//访问数据，Index_m 和 Index_n 分别是数组元素的坐标，乘上相应维度的步长，即可以访问数组元素
            }
            //cout << endl;
        }
        //cout << "count is " << count << endl;
        //cv::imshow("instance_mask:", Mask);
        //cv::waitKeyEx(0);
        Ins_Mask.push_back(Mask);
    }
    return Ins_Mask;
};
std::vector<std::vector<double>>NDArrayConverter::toVecBox(PyObject* o){
     vector<vector<double>>pre_boxes;
     int SizeOfbox = PyList_Size(o);
     //cout<<"SizeOfbox is"<< SizeOfbox<<endl;
     for(int i = 0;i<SizeOfbox;i++){

         PyObject * mask = PyList_GetItem(o, i);
         int sizeofmask = PyList_Size(mask);
         //cout<<"sizeofmask is "<<sizeofmask<<endl;
         //int typenum = PyArray_TYPE(MaskItem);
         for(int i = 0;i < sizeofmask;i++){
             PyObject* item = PyList_GetItem(mask, i);
             double val;
             PyArg_Parse(item,"d",&val);
             //cout<<"val is" <<val<<endl;
         }
     }
     return pre_boxes;
};
std::vector<int> NDArrayConverter::toIdVec(PyObject* o){
    std::vector<int>MaskId;
    int SizeOfMask = PyList_Size(o);
    for(int i =0; i< SizeOfMask;i++){
        int result;
        PyObject *Item = PyList_GetItem(o, i);
        PyArg_Parse(Item,"i",&result);
        MaskId.push_back(result);
        //printf("%d",result);
    }
    return MaskId;
};

cv::Mat NDArrayConverter::toMat(PyObject *o) {
    PyArrayObject *ret_array;
    PyArray_OutputConverter(o, &ret_array);
    npy_intp *shape = PyArray_SHAPE(ret_array);
    //std::cout<<"shape [0] is "<<shape[0]<<endl;
    //std::cout<<"shape [1] is "<<shape[1]<<endl;
    //下列代码以一个二位的PyArray_DATA来初始化cv::Mat数据；保证其元素类型为正确即可；
    cv::Mat big_img(shape[0], shape[1], CV_64F, PyArray_DATA(ret_array));
    //std::cout<<"here works"<<std::endl;
    return big_img;
}





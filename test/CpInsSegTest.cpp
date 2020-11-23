//
// Created by lxy on 20-8-21.
//
#include<iostream>
#include<fstream>
#include<iomanip>
#include<dirent.h>
#include<errno.h>
#include<python3.6/Python.h>
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


using namespace std;
using namespace cv;


int main(){
    std::cout<<"we begin to test te P2C"<<std::endl;
    //获取需要调用的python文件的工作位置；
    //获取python工作的路径；
    std::string py_path = "/home/lxy/SLAM/VIO/VINS-Course-master/AdelaiDet/demo/";
    //调用某个python文件；
    std::string module_name="BlendMask";
    //调用python文件中的哪个类；
    //std::string class_name  = "run";
    //调用python文件中某个类的某个函数；
    //std::string get_instance_seg = "GetInstanceSeg";
    std::string x;
    setenv("PYTHONPATH", py_path.c_str(), 1);
    x = getenv("PYTHONPATH");

    std::cout<<x<<std::endl;
    cv::Mat img = cv::imread("/home/lxy/SLAM/VIO/VINS-Course-master/AdelaiDet/datasets/coco/000061.png",CV_LOAD_IMAGE_UNCHANGED);

    /*
    int m,n;
    std::cout<<"img.cols is "<< img.cols<<std::endl;
    std::cout<<"img.ross is "<< img.rows<<std::endl;
    n = img.cols*3;
    m = img.rows;
    //分配空间；并将图像记录到data的内存空间中；
    std::cout<<"m is :"<< m  <<"n is :"<<  n  << std::endl;
    unsigned  char *data = (unsigned char*)malloc(sizeof(unsigned char*)*m*n);
    int p=0;
    for(int i=0;i<m;i++) {
        for (int j = 0; j < n; j++) {
            data[p] = img.at<unsigned char>(i, j);
            //std::cout<<"value in img: "<< img.at<unsigned char>(i,j);
            p++;
        }
    }

    //cv::imshow("lets see",img);
    //cv::waitKey(0);*/
    Py_Initialize();
//import_array();
    NDArrayConverter* cvt = new NDArrayConverter();
    PyObject* py_image = cvt->mat2Pyobejct(img);

    PyObject* py_module = PyImport_ImportModule("BlendMask");
    assert(py_module!=NULL);

    PyObject* pClass = PyObject_GetAttrString(py_module,"Blendmask");
    PyObject* pObject = PyEval_CallObject(pClass,NULL); //对“BLendmask”类 进行实例化；
    PyObject* pFunc = PyObject_GetAttrString(pObject, "GetInsSeg");
    //获取分割实例并接受返回值；
    cout<<"get pred_classes "<<endl;
    PyObject* pyValue = PyObject_CallObject(pFunc, py_image);//函数调用
    cout<<"successfully get result"<<endl;
    /*
    if(PyList_Check(pyValue)){
        cout<<"返回值是一个pylist"<<endl;
    }*/
    //vector<int>MaskId;
    //int Size =PyList_Size(pyValue);
    //cout<<"Size  is "<<Size<<endl;
    /*
    PyObject* pred_class = PyList_GetItem(pyValue, 1);
    int SizeOfMask = PyList_Size(pred_class);
    for(int i =0; i< SizeOfMask;i++){
        int result;
        PyObject *Item = PyList_GetItem(pred_class, i);
        PyArg_Parse(Item,"i",&result);
        MaskId.push_back(result);
        printf("%d",result);
    }

    PyObject* pred_mask = PyList_GetItem(pyValue, 0);
    cvt->toMatVec(pred_mask);*/
    //PyArrayObject *ret_array;
    //PyArray_OutputConverter(pyValue, &ret_array);




    //for(int i =0;i<SizeOfList;i++){
       //PyArrayObject* D2_array;
       //D2_array = ret_array(i);
   // }
    //cvt->toMatVec(pyValue);
    //解析PyList
    /*
    vector<int>MaskId;
    int SizeOfMask =PyList_Size(pyValue);
    cout<<"Size of Mask is "<<SizeOfMask<<endl;
    for(int i =0; i< SizeOfMask;i++){
        int result;
        PyObject *Item = PyList_GetItem(pyValue, i);
        PyArg_Parse(Item,"i",&result);
        MaskId.push_back(result);
        printf("%d",result);
    }
    */

    //cv::Mat big_img = cvt->toMat(pyValue);
    //cv::imshow("big_img ",big_img);
    //cv::waitKey(0);
    Py_Finalize();

    return 0;
    /*
    int count=0;
    for(int i=0;i<shape[0];i++){
        for(int j=0;j<shape[1];j++){
            if(big_img.at<uchar>(i,j)!=0){
                count++;
            }else{
                continue;
            }
        }
    }*/
    //cout<<"count is "<<count<<endl;

    //int ndims = PyArray_NDIM(ret_array);
    //cout<<"ndims in ret_array is "<< ndims<<endl;

    //const int CV_MAX_DIM = 32;

}


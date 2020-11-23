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

using namespace std;
using namespace cv;


int main(){
    std::cout<<"we begin to test te P2C"<<std::endl;
    //获取需要调用的python文件的工作位置；
    //获取python工作的路径；
    std::string py_path = "/home/lxy/SLAM/VIO/VINS-Course-master/AdelaiDet/demo/";
    //调用某个python文件；
    std::string module_name="test1";
    //调用python文件中的哪个类；
    //std::string class_name  = "run";
    //调用python文件中某个类的某个函数；
    //std::string get_instance_seg = "GetInstanceSeg";
    std::string x;
    setenv("PYTHONPATH", py_path.c_str(), 1);
    x = getenv("PYTHONPATH");

    std::cout<<x<<std::endl;
    cv::Mat img = cv::imread("/home/lxy/SLAM/VIO/VINS-Course-master/AdelaiDet/datasets/coco/000061.png",CV_LOAD_IMAGE_UNCHANGED);
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
    //cv::waitKey(0);
    Py_Initialize();
    import_array();

    npy_intp Dims[2]={m,n};
    //导入PyObject；
    PyObject*PyArray = PyArray_SimpleNewFromData(2, Dims, NPY_UBYTE, data);
    //导入当前路径
    //PyRun_SimpleString("import cv2");
    //PyRun_SimpleString("sys.path.append('./')");
    //PyRun_SimpleString("print(sys.path)");
    //PyRun_SimpleString("print(os.listdir())");
    //PyObject* m_pPythonName   = PyString_FromString(py_path);

    PyObject* py_module = PyImport_ImportModule("test2");
    assert(py_module!=NULL);
    //PyObject *pDict = PyModule_GetDict(py_module);
    //assert(pDict!=NULL);
    //PyObject* pFunc = PyDict_GetItemString(pDict, "load_image");

    //将cv中image的格式转换为Pyobejct;
    //准备传入python3的参数；
     PyObject *ArgArray = PyTuple_New(2);
     PyObject *arg = PyLong_FromLong(10);
     PyTuple_SetItem(ArgArray, 0, PyArray);
     PyTuple_SetItem(ArgArray, 1, arg);
    //读取目标函数；
    PyObject* pClass = PyObject_GetAttrString(py_module,"Blendmask");
    PyObject* pObject = PyEval_CallObject(pClass,NULL); //对“BLendmask”类 进行实例化；
    PyObject* pFunc = PyObject_GetAttrString(pObject, "GetInstanceSeg");
    //获取分割实例并接受返回值；
    PyObject* pyValue = PyObject_CallObject(pFunc, ArgArray);//函数调用
    //解析返回值；
    //PyArg_ParseTuple(pyValue, "i", &Integration_mask);
    if(pyValue == NULL)
    {
        Py_Finalize();
        return false;
    }
    std::cout<<"pyValue is correct"<<std::endl;
    PyArrayObject *ret_array;
    PyArray_OutputConverter(pyValue, &ret_array);
    int typenum = PyArray_TYPE(ret_array), new_typenum = typenum;
    int type = typenum == NPY_UBYTE ? CV_8U :
               typenum == NPY_BYTE ? CV_8S :
               typenum == NPY_USHORT ? CV_16U :
               typenum == NPY_SHORT ? CV_16S :
               typenum == NPY_INT ? CV_32S :
               typenum == NPY_INT32 ? CV_32S :
               typenum == NPY_FLOAT ? CV_32F :
               typenum == NPY_DOUBLE ? CV_64F : -1;
    cout<<"type is "<<typenum<<endl;
    cout<<"NPY_INT32"<<NPY_DOUBLE<<endl;

    npy_intp *shape = PyArray_SHAPE(ret_array);

    cv::Mat big_img(shape[0], shape[1], CV_64F, PyArray_DATA(ret_array));

    cout<<"shape[0] is "<< shape[0]<<endl;
    cout<<"shape[1] is "<< shape[1]<<endl;
    cout<<"big_img.cols is "<<big_img.cols<<endl;
    cout<<"big_img.rows is "<<big_img.rows<<endl;

    Mat_<uchar>::iterator it = big_img.begin<uchar>();
    cout<<"value in big_img(200,800) "<<*(it + 800*shape[1] + 200)<<endl;

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
    cout<<"count is "<<count<<endl;
    cv::imshow("big_img ",big_img);
    cv::waitKey(0);
    //int ndims = PyArray_NDIM(ret_array);
    //cout<<"ndims in ret_array is "<< ndims<<endl;

    //const int CV_MAX_DIM = 32;





    Py_Finalize();

    return 0;
}


//
// Created by lxy on 20-8-27.
//
#pragma once
#include<InstanceSeg.h>
#include<object.h>
InstanceSeg::InstanceSeg() {
    std::cout << "Importing BlendMask Settings..." << std::endl;
    ImportSettings();
    std::string x;
    //std::cout<< this->py_path.c_str()<<std::endl;
    setenv("PYTHONPATH", this->py_path.c_str(), 1);
    x = getenv("PYTHONPATH");
    std::cout<<"python work path is :"<< x  <<std::endl;
    Py_Initialize();
    this->cvt = new NDArrayConverter();
    this->py_module=PyImport_ImportModule(this->module_name.c_str());
    assert(this->py_module != NULL);
    this->py_class = PyObject_GetAttrString(this->py_module, this->class_name.c_str());
    //对BLendMask进行实例化；
    this->net = PyEval_CallObject(this->py_class, NULL);
    assert(this->net != NULL);
    this->pFunc = PyObject_GetAttrString(this->net, this->get_dyn_seg.c_str());
    assert(this->pFunc!=NULL);
    this->InsSeg = PyObject_GetAttrString(this->net, this->get_Ins_seg.c_str());
    std::cout << "Creating net instance..." << std::endl;
    cv::Mat image  = cv::Mat::zeros(375,1242,CV_64F); //Be careful with size!!
    std::cout << "Loading net parameters..." << std::endl;
    GetDynSeg(image);
    //GetInsSeg(image);
    //Py_Finalize();
};

cv::Mat InstanceSeg::GetDynSeg(cv::Mat &image, std::string dir, std::string rgb_name) {

        PyObject* py_image =cvt->mat2Pyobejct(image.clone());
        assert(this->pFunc!=NULL);
        /*
        std::cout<<"what wrong with the fig"<<std::endl;
        cv::imshow("fig is ",image);
        cv::waitKeyEx(0);*/

        //根据修正过的参数，传入python中的blendmask，获得分割结果，并返回py_mask_img；
        PyObject* py_mask_img = PyObject_CallObject(this->pFunc, py_image);

        //std::cout<<"是否成功调用？"<<std::endl;
        cv::Mat big_img = this->cvt->toMat(py_mask_img);
        /*
        cv::imshow("big_img is ",big_img);
        cv::waitKeyEx(0);*/
        return big_img;
}
Ins_Seg_result InstanceSeg::GetInsSeg(cv::Mat &image, std::string dir,std::string rgb_name){
    PyObject* py_image = cvt->mat2Pyobejct(image.clone());
    assert(py_image != NULL);
    PyObject* py_mask_img = PyObject_CallObject(this->InsSeg, py_image);
    PyObject* pred_mask = PyList_GetItem(py_mask_img, 0);
    PyObject* pred_class = PyList_GetItem(py_mask_img, 1);
    PyObject* pred_boxes = PyList_GetItem(py_mask_img,2);
    Ins_Seg_result result_Mask;
    result_Mask.InsSeg_Mask = cvt->toMatVec(pred_mask);
    result_Mask.MaskId = cvt->toIdVec(pred_class);
    result_Mask.pred_boxes = cvt->toVecBox(pred_boxes);
    return result_Mask;
}

void InstanceSeg::ImportSettings(){
    std::string strSettingsFile = "/home/lxy/SLAM/VIO/VINS-Course-master/config/BlendMaskSettings.yaml";
    std::cout<<strSettingsFile<<std::endl;
    cv::FileStorage fs(strSettingsFile.c_str(), cv::FileStorage::READ);
    fs["py_path"] >> this->py_path;
    fs["module_name"] >> this->module_name;
    fs["class_name"] >> this->class_name;
    fs["get_dyn_seg"] >> this->get_dyn_seg;
    fs["get_Ins_seg"] >> this->get_Ins_seg;

    //std::cout << "    py_path: "<< this->py_path << std::endl;
    //std::cout << "    module_name: "<< this->module_name << std::endl;
    //std::cout << "    class_name: "<< this->class_name << std::endl;
    //std::cout << "    get_dyn_seg: "<< this->get_dyn_seg << std::endl;
}

//系统结束，释放内存地址；
InstanceSeg::~InstanceSeg() {
    delete this->py_module;
    delete this->py_class;
    delete this->net;
    delete this->cvt;
}

//
// Created by lxy on 20-8-27.
//

//
// Created by lxy on 20-8-27.
//
#include<InstanceSeg.h>


using namespace std;
int main(){
    std::cout<<"we begin to test InstanceSeg"<<endl;
    //InstanceSeg网络初始化；
    InstanceSeg* DynSeg;
    string dir = "/home/lxy/SLAM/VIO/VINS-Course-master/AdelaiDet/datasets/coco";
    string name = "000061.png";
    DynSeg= new InstanceSeg();

    cout<<"初始化成功"<<endl;
    cv::Mat img = cv::imread("/home/lxy/SLAM/VIO/VINS-Course-master/config/image_02/data/0000000002.png",CV_LOAD_IMAGE_UNCHANGED);
    cout<<"图像读取成功"<<endl;

    cv::imshow("img",img);
    //cv::waitKeyEx(0);
    Ins_Seg_result Dynmask;
    Dynmask = DynSeg->GetInsSeg(img);

    //cv::imshow("Mask",Dynmask);
    cv::waitKeyEx(0);
    /*
    Ins_Seg_result Class_ID_Mask;
    Class_ID_Mask = DynSeg->GetInsSeg(img);
    cout<<"Class_Id_Mask size is "<<Class_ID_Mask.MaskId.size()<<endl;
    for(int i =0;i<Class_ID_Mask.MaskId.size();i++){
        cv::Mat Mask;
        Mask = Class_ID_Mask.InsSeg_Mask[i];
        cv::imshow("End result",Mask);
        cv::waitKeyEx(0);
    }*/

    return 0;
}
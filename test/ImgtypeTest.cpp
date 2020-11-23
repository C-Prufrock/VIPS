//
// Created by lxy on 20-9-14.
//

#include <iostream>


#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(){
    cout<<"hello we work"<<endl;
    cv::Mat img,Transfer_img,dst;
    img = imread("/home/lxy/SLAM/VIO/VINS-Course-master/config/image_02/data/0000000000.png");
    cv::imshow("img",img);
    cvtColor(img,Transfer_img,CV_BGR2GRAY);

    //cvtColor(img,Transfer_img,CV_8UC1,0);
    cout<<"let's see img type is "<<Transfer_img.type()<<endl;
    cout<<"let's see img channels is "<<img.channels()<<endl;
    cout<<"let's see img type is "<<img.type()<<endl;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(Transfer_img, dst);
    cv::waitKey(0);
    return 0;
}


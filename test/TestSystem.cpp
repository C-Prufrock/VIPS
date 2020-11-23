//
// Created by lxy on 20-8-31.
//

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <thread>
#include <iomanip>

#include <cv.h>
#include <opencv2/opencv.hpp>
#include <highgui.h>
#include <eigen3/Eigen/Dense>
#include "parameters.h"

using namespace std;
using namespace cv;
int main(){
    cout<<" we begin to test System readparameters"<<endl;
    string sConfig_path = ("/home/lxy/SLAM/VIO/VINS-Course-master/config/KITTI-00-unrectified.yaml");
    cout << "1 System() sConfig_file: " << sConfig_path << endl;
    readParameters(sConfig_path);
    


    //下面code块将实现读取P0，p2的相机、imu基本参数；


    return 0;
}
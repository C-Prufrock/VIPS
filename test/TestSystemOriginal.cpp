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
    string sConfig_path = ("/home/lxy/KITTI tracking/KITTI_dev/data_tracking_calib/testing/calib/0000.txt");
    cout << "1 System() sConfig_file: " << sConfig_path << endl;
    //readParameters(sConfig_path); 必须建立kitti的config之后方才能利用readParameters函数；
    //建立两个矩阵
    //建立P_rect;

    float p_rect[12]={721.5377,0.0000,609.5593,44.85,0.0000,721.5377,172.8540,0.2163,0.0000,0.0000,1.0000,0.0027};
//P_rect
    cv::Mat P_rect =cv::Mat(3,4,CV_32F,p_rect);
    //cout<<P_rect<<endl;

    //建立R_rect
    float r_rect[16]={0.9998,0.0151,-0.0028,0,-0.0151, 0.9998, -0.0009,0,0.002827,0.0009,0.9999,0.00,0.00,0.00,0.00,1.00};
    cv::Mat R_rect = cv::Mat(4,4,CV_32F,r_rect);
    //cout<<R_rect<<endl;
    cv::Mat Project_matrix;
    Project_matrix = P_rect*R_rect;
    //cout<<Project_matrix<<endl;

    //读取y = P_rect×R_rect*T_vel_cam*T_imu_vel
    //此处我们计算T_cam_vel*T_vel_imu
    float t_velo_cam[16]= {0.007533745,-0.9999714,-0.000616602, -0.004069766,0.01480249,0.0007280733,-0.9998902,-0.07631618,0.9998621,0.00752379,0.01480755,-0.2717806,0.00,0.00,0.00,1.00};
    cv::Mat Tr_velo_vam= cv::Mat(4,4,CV_32F,t_velo_cam);

    float t_imu_vel[16]={0.9999976,0.0007553071, -0.002035826,-0.8086759,-0.0007854027,0.9998898,-0.01482298,0.3195559,0.002024406, 0.01482454,0.9998881,-0.7997231,0.00,0.00,0.00,1.00};
    cv::Mat Tr_imu_vel = cv::Mat(4,4,CV_32F,t_imu_vel);
    cv::Mat IMU_Gray01;
    IMU_Gray01 = Tr_velo_vam*Tr_imu_vel;
    //cout<<IMU_Gray01<<endl;

    float R_rect_00[16] = {0.9999239,0.00983776,-0.007445048,0.0,-0.009869795,0.9999421,-0.004278459,0.0,0.007402527,0.004351614,0.9999631,0.0,0.0,0.0,0.0,1.0};
    cv::Mat r_rect_001 = cv::Mat(4,4,CV_32F,R_rect_00);
    cv::Mat IMU_Gray02;
    IMU_Gray02 = r_rect_001*IMU_Gray01;

    float tt[12] = {1.0,0.0,0.0,0.06,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0};
    cv::Mat cam_cam_tt = cv::Mat(3,4,CV_32F,tt);
    cv::Mat cam_Imu  =  cam_cam_tt * IMU_Gray02;
    cv::Mat cam_Imu_R = cam_Imu(Range(0,3),Range(0,3));
    cv::Mat cam_Imu_t = cam_Imu(Range(0,3),Range(3,4));
    cout<<" cam_Imu "<<cam_Imu<<endl;
    cout<<" cam_Imu_R "<<cam_Imu_R<<endl;
    cout<<" cam_Imu_t "<<cam_Imu_t<<endl;
    cv::Mat Imu_cam_R,Imu_cam_t;
    invert(cam_Imu_R,Imu_cam_R,DECOMP_LU);
    cout<<" Imu_cam_R "<<Imu_cam_R<<endl;
    Imu_cam_t = -Imu_cam_R*cam_Imu_t;
    cout<<"Imu_cam_t"<<Imu_cam_t<<endl;

    //cv::Mat Imu_cam;
    //cv::invert(cam_Imu, Imu_cam,DECOMP_LU );
    //cout<<"cam_Imu is "<< Imu_cam<<endl;

    /*[00, -0.2540769;
 0.0084169023, -0.0042508226, -0.99995565, 0.71945202;
 0.99996406, 0.0010345527, 0.0084125763, -1.0890831]*/

/*
    [0.0083178589, -0.99986464, 0.014190687, -0.32921579;
    0.012777699, -0.014083739, -0.99981928, 0.71158135;
    0.99988377, 0.0084976787, 0.012658823, -1.0897827;
    0, 0, 0, 1]*/

    //下面code块将实现读取P0，p2的相机、imu基本参数；


    return 0;
}
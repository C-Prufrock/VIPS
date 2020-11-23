#pragma once

#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>

#include <fstream>
#include <condition_variable>
#include <InstanceSeg.h>
#include <cv.h>
#include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>
#include <MSG.h>
#include "estimator.h"
#include "parameters.h"
#include "feature_tracker.h"
#include "Instance_Object.h"


//imu for vio

class System
{
public:
    System(std::string sConfig_files);
    TicToc t_r;
    ~System();

    Ins_Seg_result InstanceMask;
    //动态接收返回的Mask；可以是空，也可以是很多；
    //pair<cv::Mat,double> FirstImg;
    pair<cv::Mat,double> SegFrame;
    bool Seg_Conver = false;
    bool Track_begin = true;
    InstanceSeg* DynSeg;


    //cv::Mat DynMask;
    vector<cv::Mat>SegMask;
    //用以进行分割类型选择与do分割；
    void SegImg();
    void DynaTrack();

    void PubImageData(double dStampSec, cv::Mat &img);

    void PubImuData(double dStampSec, const Eigen::Vector3d &vGyr, 
        const Eigen::Vector3d &vAcc,const Eigen::Vector3d &vVel);

    // thread: visual-inertial odometry
    void ProcessBackEnd();
    void Draw();
    
    pangolin::OpenGlRenderState s_cam;
    pangolin::View d_cam;

    FeatureTracker trackerData[NUM_OF_CAM]; //just set camera paraments;



#ifdef __APPLE__
    void InitDrawGL(); 
    void DrawGLFrame();
#endif

private:

    //feature tracker
    std::vector<uchar> r_status;
    std::vector<float> r_err;
    // std::queue<ImageConstPtr> img_buf;

    // ros::Publisher pub_img, pub_match;
    // ros::Publisher pub_restart;
    int Img_quene_len = 5;
    //int Max_Object = 100;


    double first_image_time;

    double first_cal_opt_image_time;//第一个保留图像特征进入feature_buff的图像；

    int pub_count = 1;


    bool first_image_flag = true;
    bool first_cal_opt_flag = true;


    double last_image_time = 0;
    bool init_pub = 0;

    //estimator
    Estimator estimator;

    //全局变量；
    std::condition_variable con;

    double current_time = -1;
    std::queue<ImuConstPtr> imu_buf;     //存储pub进入的imu信息；
    //std::queue<ORI_IMGPtr>Img_buf;          //存储pub进入的图像信息；
    std::queue<pair<pre_integration_ImuPtr,vector<IMG_MSG>>>feature_buf;
    std::queue<pair<pre_integration_ImuPtr,vector<shared_ptr<IMG_MSG>>>>Instance_Objects;
    vector<pair<pre_integration_ImuPtr, ORI_MSGPtr>>measurements;//for intergrate Img and Imu;
    std::queue<pair<pre_integration_ImuPtr, ORI_MSGPtr>>Img_12_buf;       //存储delay处理的12幅图像；

    // std::queue<PointCloudConstPtr> relo_buf;
    int sum_of_wait = 0;

    std::mutex m_buf;
    std::mutex m_state;
    std::mutex i_buf;
    std::mutex m_estimator;

    double latest_time;
    Eigen::Vector3d tmp_P;
    Eigen::Quaterniond tmp_Q;
    Eigen::Vector3d tmp_V;
    Eigen::Vector3d tmp_Ba;
    Eigen::Vector3d tmp_Bg;
    Eigen::Vector3d acc_0;
    Eigen::Vector3d gyr_0;
    //init_feature 应该
    bool init_feature = 0;
    bool init_imu = 1;
    double last_imu_t = 0;
    std::ofstream ofs_pose;
    std::vector<Eigen::Vector3d> vPath_to_draw;
    bool bStart_backend;
    bool bStart_Seg;
    std::vector<std::pair<std::vector<ImuConstPtr>, ImgConstPtr>> getMeasurements();
    vector<pair<pre_integration_ImuPtr, ORI_MSGPtr>>Get_Prosess();
};

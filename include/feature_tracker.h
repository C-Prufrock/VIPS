#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>
#include<fstream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "Instance_Object.h"
#include "utility/tic_toc.h"
#include<InstanceSeg.h>
#include<MSG.h>



using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(cv::Point2f point,cv::Mat mask);
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

void reduceObject(vector<Instance_Object>&Instance_Objects,vector<int>&v);

double NumberInMask(cv::Mat Mask,vector<cv::Point2f>points);

void FeatureGet(pair<const cv::Mat &,double>Img,Instance_Object& Object,camodocal::CameraPtr m_camera);
void TrackObject(pair<pre_integration_ImuPtr, ORI_MSGPtr>measurement,Instance_Object& Object,camodocal::CameraPtr m_camera,cv::Mat& F);


cv::Mat rejectWithF(Instance_Object& Object,camodocal::CameraPtr m_camera);
void setMask(Instance_Object& Object);
void addPoints(Instance_Object& Object);
void undistortedPoints(Instance_Object& Object,camodocal::CameraPtr m_camera);


class FeatureTracker
{

  public:
    FeatureTracker();
    void setK(string &calib_file);
    cv::Mat K = cv::Mat::eye(3,3,CV_32F);

    //void FeatureTracker::readImage(pair<const cv::Mat&, double>Img);
    //void readInsImage(pair<const cv::Mat&, double>Img,Ins_Seg_result Mask);

    void readMaskImage(pair<const cv::Mat &,double>Img,vector<cv::Mat>Mask,vector<int> Mask_id,vector<vector<double>>pred_boxes,cv::Mat& BGMask);
    void trackImg(pair<pre_integration_ImuPtr, ORI_MSGPtr>measurement);

    void readIntrinsicParameter(const string &calib_file);
    camodocal::CameraPtr m_camera;

    void updateID(vector<Instance_Object>&Instance_Objects);
    bool update_Object_ID(unsigned int i);

    vector<Instance_Object>New_Instance_Objects;
    vector<Instance_Object>Instance_Objects;

    vector<int>ids;
    int Object_id;
    bool Dyna_judge = false;

    vector<Instance_Object>addObjects(vector<Instance_Object>Instance_Objects);
    vector<Instance_Object>addObjects(Instance_Object Bg);
    /*
    cv::Mat mask;
    cv::Mat fisheye_mask;

    cv::Mat prev_img, cur_img, forw_img;


    vector<cv::Point2f> n_pts;

    //里面一层int 代表是什么物体；外面一层int代表是第几个物体，注意不是同类的第几个，而是场景中的第几个；

    /*
    vector<Object>prev_pts, cur_pts, forw_pts;
    vector<Object> prev_un_pts, cur_un_pts;



    //两层嵌套：第一层记录是第几个物体，第二层记录是该物体的第几个特征点；
    vector<vector<int>> ids;
    vector<vector<int>> track_cnt;
    vector<map<int, cv::Point2f>> cur_un_pts_map;
    vector<map<int, cv::Point2f>> prev_un_pts_map;
    vector<vector<cv::Point2f>> pts_velocity;

    //camera model is just for function liftprojective;


    double cur_time;
    double prev_time;

    static int n_id;*/
};

//
// Created by lxy on 20-9-26.
//

#ifndef VINS_ESTIMATOR_INSTANCE_OBJECT_H
#define VINS_ESTIMATOR_INSTANCE_OBJECT_H

#include<opencv2/opencv.hpp>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>
using namespace std;


class Instance_Object {
public:
    cv::Mat mask;
    //cv::Mat fisheye_mask;
    cv::Mat prev_img, cur_img, forw_img;
    vector<cv::Point2f> n_pts;
    vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
    vector<cv::Point2f> prev_un_pts, cur_un_pts;
    vector<cv::Point2f> pts_velocity;
    vector<int> ids;
    vector<int> track_cnt;
    map<int, cv::Point2f> cur_un_pts_map;
    map<int, cv::Point2f> prev_un_pts_map;
    double cur_time;
    double prev_time;

    int id_for_class;//recording what the Instance_Object is;
    //NeedObstract is for juding whether there is new mask;
    bool NeedObstract = false;
    //Dyna_judge is for when new Mask(new body appears),whether we need to do dyna_judge;
    // we addObject according to the Dyna_judge while we will delete according to the Dyna_judge;
    bool Dyna_judge = false;
    //isDyna is for judging this new object is dyna;
    bool isDyna = false;
    //Tracking is for judging whther there is enough features in cur_Object
    bool Tracking = true;
    //NewObject is for judging whether this object is new object;
    bool NewObject = false;

    int n_id = 0;
};


#endif //VINS_ESTIMATOR_INSTANCE_OBJECT_H

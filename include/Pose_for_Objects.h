//
// Created by lxy on 20-10-25.
//

#ifndef VINS_ESTIMATOR_POSE_FOR_OBJECTS_H
#define VINS_ESTIMATOR_POSE_FOR_OBJECTS_H

#include "feature_manager.h"

class Pose_for_Objects{
public:
    int frame_count = 0;
    int id_for_objects;
    int id_for_class;
    int last_track_num = 0;
    void addFeature(Pose_for_Objects& object_pose, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td);
    //double computeparalex(FeaturePerId &it_per_id, int frame_count);
    //used to deal with features belonging to current Object;

    list<FeaturePerId> feature;

    vector<double>headers;
    vector<Vector3d> Ps;
    vector<Matrix3d> Rs;
    vector<Vector3d> Vs;
};


/*
double Pose_for_Objects::computeparalex(FeaturePerId &it_per_id, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and current frame
     FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
     FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    double ans = 0;
    Vector3d p_j = frame_j.point;

    double u_j = p_j(0);  //3D点在当前帧计算时的x坐标；
    double v_j = p_j(1);  //3D点在当前帧计算时的y坐标；

    Vector3d p_i = frame_i.point;
    Vector3d p_i_comp;

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    p_i_comp = p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;

    double du = u_i - u_j, dv = v_i - v_j;

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}*/

#endif //VINS_ESTIMATOR_POSE_FOR_OBJECTS_H

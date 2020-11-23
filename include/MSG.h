//
// Created by lxy on 20-10-4.
//

#ifndef VINS_ESTIMATOR_MSG_H
#define VINS_ESTIMATOR_MSG_H

#include "estimator.h"

struct IMU_MSG
{
    double header;  //header应该是时间戳
    Eigen::Vector3d linear_acceleration;
    Eigen::Vector3d angular_velocity;
    Eigen::Vector3d velocity;
};

typedef std::shared_ptr<IMU_MSG const> ImuConstPtr;


//image for vio
struct IMG_MSG {
    double header;
    vector<Vector3d> points;
    vector<int> id_of_point;
    vector<float> u_of_point;
    vector<float> v_of_point;
    vector<float> velocity_x_of_point;
    vector<float> velocity_y_of_point;
    int ids_for_Object;
    int ids_for_class;
};

typedef std::shared_ptr <IMG_MSG const > ImgConstPtr;

//用以存储图像信息,进行IMU与IMG信息的匹配；里面存储的就是一个原始图像与其时间戳；
struct ORI_MSG{
    double header;
    cv::Mat img;
};

typedef std::shared_ptr<ORI_MSG>ORI_MSGPtr ;

typedef std::shared_ptr<IntegrationBase>pre_integration_ImuPtr;


#endif //VINS_ESTIMATOR_MSG_H

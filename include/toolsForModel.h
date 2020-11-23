//
// Created by lxy on 20-10-15.
//

#ifndef VINS_ESTIMATOR_TOOLSFORMODEL_H
#define VINS_ESTIMATOR_TOOLSFORMODEL_H


#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>

using  namespace Eigen;
using namespace std;

cv::Mat Cal_F(vector<cv::Point2f>&points1, vector<cv::Point2f>&points2,vector<uchar>&status);
void Normalize(vector<cv::Point2f> &vKeys, vector<Vector2f> &vNormalizedPoints, Matrix3f &T);
// Ransac max iterations
Matrix3f ComputeF21(vector<Vector2f> &vP1, const vector<Vector2f> &vP2);
float CheckFundamental( Matrix3f &F21, vector<uchar> &vbMatchesInliers, vector<cv::Point2f>points1,vector<cv::Point2f>points2,float sigma);
void show(Matrix3f &F21, vector<uchar>&vbMatchesInliers,vector<cv::Point2f>points1,vector<cv::Point2f>points2,float sigma);



// Ransac sets


#endif //VINS_ESTIMATOR_TOOLSFORMODEL_H

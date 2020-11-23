#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
#include <map>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

// #include <ros/console.h>
// #include <ros/assert.h>

#include "parameters.h"

class FeaturePerFrame
{
public:
  FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td)
  {
    point.x() = _point(0);
    point.y() = _point(1);
    point.z() = _point(2);
    uv.x() = _point(3);
    uv.y() = _point(4);
    velocity.x() = _point(5);
    velocity.y() = _point(6);
    cur_td = td;
  }
  double cur_td;

  //3D点、对应图像上二维点，对应速度；
  Vector3d point;
  Vector2d uv;
  Vector2d velocity;

  double z;
  bool is_used;
  double parallax;
  MatrixXd A;
  VectorXd b;

  double dep_gradient;
};

class FeaturePerId
{
public:
  const int feature_id;
  //起始帧；
  int start_frame;

  vector<FeaturePerFrame> feature_per_frame;

  int used_num;
  //是否为外点；
  bool is_outlier;
  //是否已经边缘化；
  bool is_margin;
  //估计深度如何？
  double estimated_depth;
  //是否已经成功解算？作为成熟的地图点？
  int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;

  //3D点坐标？
  Vector3d gt_p;

  //采用列表进行初始化；
  FeaturePerId(int _feature_id, int _start_frame)
      : feature_id(_feature_id), start_frame(_start_frame),
        used_num(0), estimated_depth(-1.0), solve_flag(0)
  {
  }

  int endFrame();
};

//特征管理的操作类；
class FeatureManager
{
public:
  FeatureManager(Matrix3d _Rs[]);

  void setRic(Matrix3d _ric[]);

  void clearState();

  int getFeatureCount();

  bool addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td);
  void debugShow();
  vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);

  //void updateDepth(const VectorXd &x);
  //设定深度；
  void setDepth(const VectorXd &x);
  //剔除深度化失败的点；
  void removeFailures();
  //清除深度；
  void clearDepth(const VectorXd &x);
  //获得深度矢量；
  VectorXd getDepthVector();
  //进行三角化；
  void triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);
  //？下面这个函数没搞清楚作用；
  void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
  void removeBack();
  void removeFront(int frame_count);
  void removeOutlier();

  list<FeaturePerId> feature;

  int last_track_num;

private:
  double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
  const Matrix3d *Rs;
  Matrix3d ric[NUM_OF_CAM];
};

#endif
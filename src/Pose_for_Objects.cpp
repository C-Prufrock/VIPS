//
// Created by lxy on 20-11-15.
//
#include<Pose_for_Objects.h>
void Pose_for_Objects::addFeature(Pose_for_Objects& object_pose, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td) {
//ROS_DEBUG("input feature: %d", (int)image.size());
//ROS_DEBUG("num of feature: %d", getFeatureCount());
    object_pose.frame_count++;
    double parallax_sum = 0;
    object_pose.last_track_num++;

    int parallax_num = 0;

    for (auto &id_pts : image) {
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td);

        int feature_id = id_pts.first;

        auto it = find_if(object_pose.feature.begin(), object_pose.feature.end(), [feature_id](const FeaturePerId &it) {
            return it.feature_id == feature_id;
        });

        if (it == object_pose.feature.end()) {
            object_pose.feature.push_back(FeaturePerId(feature_id, object_pose.frame_count));
            object_pose.feature.back().feature_per_frame.push_back(f_per_fra);
        } else if (it->feature_id == feature_id) {
            it->feature_per_frame.push_back(f_per_fra);
        }
    }
}

//第二帧，或者上一帧经过特征点提取后跟踪特征数目仍小于20，则设定为关键点；
//cout<<"last_track_num is "<<last_track_num<<endl;
/*
if (object_pose.frame_count < 2)
    return true;

for (auto &it_per_id : object_pose.feature)
{
    if (it_per_id.start_frame <= object_pose.frame_count - 2 &&
        it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= object_pose.frame_count - 1)
    {

        //检验每个3D点的最后一帧与倒数第二帧之间的视差判断视差是否足够大，倒数第二帧是否为关键帧；
        parallax_sum += computeparalex(it_per_id, object_pose.frame_count);
        parallax_num++;

    }
}


int count = 0;
for(auto &it_per_id : object_pose.feature){
    if(it_per_id.start_frame <= frame_count - 2 &&
       it_per_id.start_frame + int(it_per_id.feature_per_frame.size())-1 == frame_count)
    {
        count++;
    }
}
//cout<<"frame_Count is"<<frame_count<<endl;
//std::cout<<"we can see the number in last frame_count is "<< count<<endl;*/
/*
if (parallax_num == 0)
{
   return true;
}
else
{
   //ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
   //ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
   return parallax_sum / parallax_num >= MIN_PARALLAX;
}*/


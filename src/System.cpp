#include "System.h"

#include <pangolin/pangolin.h>
#include <MSG.h>

using namespace std;
using namespace cv;
using namespace pangolin;

System::System(string sConfig_file_)
    :bStart_backend(true)
{
    //string sConfig_file = sConfig_file_ + "euroc_config.yaml";
    bStart_Seg  = true;
    string sConfig_file = sConfig_file_ + "KITTI-00-unrectified.yaml";

    cout << "1 System() sConfig_file: " << sConfig_file << endl;
    readParameters(sConfig_file);

    trackerData[0].readIntrinsicParameter(sConfig_file);
    trackerData[0].setK(sConfig_file);

    estimator.setParameter();
    ofs_pose.open("./pose_output.txt",fstream::out);
    if(!ofs_pose.is_open())
    {
        cerr << "ofs_pose is not open" << endl;
    }
    // thread thd_RunBackend(&System::process,this);
    // thd_RunBackend.detach();
    //DynSeg = new InstanceSeg();
    //系统开始的时候就配置实例分割相关数据；

    cout << "2 System() end" << endl;
}

System::~System()
{
    bStart_backend = false;
    bStart_Seg = false;
    pangolin::QuitAll();

    //m_buf负责保护feature_buf 与 imu_buf的数据安全；
    m_buf.lock();
    while (!feature_buf.empty())
        feature_buf.pop();
    while (!imu_buf.empty())
        imu_buf.pop();
    while(!Img_12_buf.empty())
        Img_12_buf.pop();
    m_buf.unlock();

    i_buf.lock();
    while(Seg_Conver == true)
        Seg_Conver = false;
    i_buf.unlock();

    SegMask.clear();

    m_estimator.lock();
    estimator.clearState();
    m_estimator.unlock();

    ofs_pose.close();
}

void System::PubImageData(double dStampSec, Mat &img){
    //仍然进行一些图像判断；根据判断选择要处理的图像；

    if (!init_feature)
    {
        cout << "1 PubImageData skip the first detected feature, which doesn't contain optical flow speed" << endl;
        init_feature = 1;
        cout<<"Tic_toc"<<t_r.toc()<<endl;
        return;
    }

    if (first_image_flag) //用来判断图像的频率；
    {
        cout << "2 PubImageData first_image_flag" << endl;
        first_image_flag = false;
        first_image_time = dStampSec;
        last_image_time = dStampSec;
        return;
    }

    // detect unstable camera stream
    if (dStampSec - last_image_time > 1.0 || dStampSec < last_image_time)
    {
        cerr << "3 PubImageData image discontinue! reset the feature tracker!" << endl;
        first_image_flag = true;
        last_image_time = 0;
        pub_count = 1;
        return;
    }
    //cout<<"last_image_time is "<<last_image_time<<endl;
    last_image_time = dStampSec;
    // frequency control　　　－－－－　此处代码块
    if (round(1.0 * pub_count / (dStampSec - first_image_time)) <= FREQ)
    {
        //如果频率过高，则直接删除该图像；
        //PUB_THIS_FRAME = true;
        // reset the frequency control
        if (abs(1.0 * pub_count / (dStampSec - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = dStampSec;
            pub_count = 0;
        }
    }
    else
    {
        return;
    }
    //此处出现第一幅图像；
    //每增加一副图像，measurements增加一元素；


    //等待Imu数据不为空；
    unique_lock<mutex>lk(m_buf);
    while(imu_buf.empty() || (imu_buf.back()->header <= dStampSec + estimator.td))
    {
        //cout<<"我们正在等待imu数据"<<endl;
        con.wait(lk);}
    lk.unlock();

    //如果imu的首帧时间戳尚且大于该帧时间戳;则重新发布图像；
    if (!(imu_buf.front()->header < dStampSec + estimator.td))
    {
        cerr << "throw img, only should happen at the beginning,reset pubImage"<< endl;
        //reset
        init_feature=0;
        first_image_flag = false;
        return;
    }

    vector<ImuConstPtr>IMUs;
    if (!init_pub)
    {
        //发布的首帧;
        cout << "首次发布图像" << endl;
        first_cal_opt_image_time = dStampSec;
        cout<<" we can see first ImgTimeStamp: "<<setprecision(10)<<first_cal_opt_image_time<<endl;
        //剔除所有小于该帧的imu数据；
        ImuConstPtr LastImuForFirstImage;
        shared_ptr<IMU_MSG> FirstImuForFirstImage(new IMU_MSG());
        //cout<<"here works1"<<endl;
        int imu_count = 0;
        while (imu_buf.front()->header < first_cal_opt_image_time + estimator.td){
            //找到第一个大于第一个Img时间戳的图像；
            LastImuForFirstImage = imu_buf.front();
            imu_buf.pop();
            imu_count++;
        }
        cout<<"imu_count is "<<imu_count<<endl;
        FirstImuForFirstImage->header = first_cal_opt_image_time + estimator.td ;
        double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
        double dt_1 =  first_cal_opt_image_time - LastImuForFirstImage->header;
        double dt_2 = imu_buf.front()->header  - first_cal_opt_image_time;
        assert(dt_1 >= 0);
        assert(dt_2 >= 0);
        assert(dt_1 + dt_2 > 0);

        double w1 = dt_2 / (dt_1 + dt_2);
        double w2 = dt_1 / (dt_1 + dt_2);


        Vector3d da = imu_buf.front()->linear_acceleration - LastImuForFirstImage->linear_acceleration;

        //利用相机时间戳 与相邻的两个imu数据时间戳对相机处时间imu数据进行差值估计；
        FirstImuForFirstImage->linear_acceleration.x() = w1 * LastImuForFirstImage->linear_acceleration.x()+w2 * imu_buf.front()->linear_acceleration.x();
        FirstImuForFirstImage->linear_acceleration.y() = w1 * LastImuForFirstImage->linear_acceleration.y()+w2 * imu_buf.front()->linear_acceleration.y();
        FirstImuForFirstImage->linear_acceleration.z() = w1 * LastImuForFirstImage->linear_acceleration.z()+w2 * imu_buf.front()->linear_acceleration.z();
        FirstImuForFirstImage->angular_velocity.x() =  w1 * LastImuForFirstImage->angular_velocity.x() +w2 * imu_buf.front()->angular_velocity.x();
        FirstImuForFirstImage->angular_velocity.y() =  w1 * LastImuForFirstImage->angular_velocity.y() +w2 * imu_buf.front()->angular_velocity.y();
        FirstImuForFirstImage->angular_velocity.z() =  w1 * LastImuForFirstImage->angular_velocity.z() +w2 * imu_buf.front()->angular_velocity.z();
        FirstImuForFirstImage->velocity.x() = w1 * LastImuForFirstImage->velocity.x() + w2* imu_buf.front()->velocity.x();
        FirstImuForFirstImage->velocity.y() = w1 * LastImuForFirstImage->velocity.y() + w2* imu_buf.front()->velocity.y();
        FirstImuForFirstImage->velocity.z() = w1 * LastImuForFirstImage->velocity.z() + w2* imu_buf.front()->velocity.z();
        cout<<"LastImuForFirstImage->velocity.x() is :"<<LastImuForFirstImage->velocity.x()<<endl;
        cout<<"LastImuForFirstImage->velocity.y() is :"<<LastImuForFirstImage->velocity.y()<<endl;
        cout<<"LastImuForFirstImage->velocity.z() is :"<<LastImuForFirstImage->velocity.z()<<endl;

        cout<<"FirstImuForImg->velocity.x() is :"<<FirstImuForFirstImage->velocity.x()<<endl;
        cout<<"FirstImuForImg->velocity.y() is :"<<FirstImuForFirstImage->velocity.y()<<endl;
        cout<<"FirstImuForImg->velocity.z() is :"<<FirstImuForFirstImage->velocity.z()<<endl;

        cout<<"imu_buf->velocity.x() is :"<<imu_buf.front()->velocity.x()<<endl;
        cout<<"imu_buf->velocity.y() is :"<<imu_buf.front()->velocity.y()<<endl;
        cout<<"imu_buf->velocity.z() is :"<<imu_buf.front()->velocity.z()<<endl;

        IMUs.emplace_back(FirstImuForFirstImage);
        imu_buf.pop();

        //此时可以进行分割了；
        SegFrame.first =img;
        SegFrame.second = dStampSec;
        Seg_Conver = true;
        cout<<"Seg_Conver is true now"<<endl;
        con.notify_all();
        init_pub = 1;
        return;
    }
    //cout<<"here works"<<endl;

    //cout<<"last_image_time is "<<last_image_time<<endl;
    while (imu_buf.front()->header < dStampSec + estimator.td)
    {
        //cout<<"我们正在循环　"<<endl;
        //cout<<"imu_buf.front() is "<<imu_buf.front()->header<<endl;
        IMUs.emplace_back(imu_buf.front());
        imu_buf.pop();
    }
    //cout<<"current image time is"<<dStampSec<<endl;

    IMUs.emplace_back(imu_buf.front());//把下一个buf中的首帧也加入了，但没有在imu_buf中剔除，便于对当前帧进行预积分的差值；
    if (IMUs.empty()){
        cerr << "no imu between two image" << endl;
    }

    //输出保留在vector<vector<IMU_MSG>,ORI_IMG>measurements;
    std::shared_ptr<ORI_MSG>Img(new ORI_MSG());
    Img->header = dStampSec;
    Img->img = img;

    //begin to pre_inter_gration;
    IntegrationBase* cur_pre_integration;
    Eigen::Vector3d acc_w(ACC_W,ACC_W,ACC_W);
    Eigen::Vector3d gyr_w(GYR_W,GYR_W,GYR_W);
    cur_pre_integration = new IntegrationBase{IMUs.front()->linear_acceleration,IMUs.front()->linear_acceleration,acc_w,gyr_w};


    double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0,vx = 0,vy = 0,vz = 0;
    //mean to utilize imu to seg dynamic environment, while we do not do it.
    //cout<<"first know measurement.first.size"<<measurement.first.size()<<endl;
    for (auto &imu_msg : IMUs)//measurement.first的第一要素是vectro<ImuConsPtr>// 的地址；
    {
        //cout<<"imu_msg->header  :"<<setprecision(10) <<imu_msg->header;
        double t = imu_msg->header;
        double img_t = dStampSec + estimator.td;
        if (t <= img_t)
        {
            if (current_time < 0)
                current_time = t;
            double dt = t - current_time;
            assert(dt >= 0);
            current_time = t;
            dx = imu_msg->linear_acceleration.x();
            dy = imu_msg->linear_acceleration.y();
            dz = imu_msg->linear_acceleration.z();

            rx = imu_msg->angular_velocity.x();
            ry = imu_msg->angular_velocity.y();
            rz = imu_msg->angular_velocity.z();

            vx = imu_msg->velocity.x();
            vy = imu_msg->velocity.y();
            vz = imu_msg->velocity.z();

            cur_pre_integration->push_back(dt, Vector3d(dx, dy, dz),Vector3d(rx, ry, rz),Vector3d(vx,vy,vz));
            //cur_pre_integration = estimator.pre_IMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
            // printf("1 BackEnd imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);
        }
        else
        {
            double dt_1 = img_t - current_time;
            double dt_2 = t - img_t;
            current_time = img_t;
            assert(dt_1 >= 0);
            assert(dt_2 >= 0);
            assert(dt_1 + dt_2 > 0);
            double w1 = dt_2 / (dt_1 + dt_2);
            double w2 = dt_1 / (dt_1 + dt_2);


            vx = vx + dx * dt_1 + 0.5 * (imu_msg->linear_acceleration.x() - dx) * dt_1 *w2;
            vy = vy + dy * dt_1 + 0.5 * (imu_msg->linear_acceleration.y() - dy) * dt_1 *w2;
            vz = vz + dz * dt_1 + 0.5 * (imu_msg->linear_acceleration.z() - dz) * dt_1 *w2;


            dx = w1 * dx + w2 * imu_msg->linear_acceleration.x();
            dy = w1 * dy + w2 * imu_msg->linear_acceleration.y();
            dz = w1 * dz + w2 * imu_msg->linear_acceleration.z();
            rx = w1 * rx + w2 * imu_msg->angular_velocity.x();
            ry = w1 * ry + w2 * imu_msg->angular_velocity.y();
            rz = w1 * rz + w2 * imu_msg->angular_velocity.z();



            cur_pre_integration->push_back(dt_1, Vector3d(dx, dy, dz),Vector3d(rx, ry, rz),Vector3d(vx,vy,vz));
            //cur_pre_integration = estimator.pre_IMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
            //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
        }
    }
    //cout<<"我们插入了　："<< measurements[0].first.size()<<endl;

    measurements.emplace_back(make_pair(cur_pre_integration,Img));
    // cout << "1 PubIsmuData t: " << fixed << imu_msg->header
    //     << " imu_buf size:" << imu_buf.size() << endl;
}

vector<pair<pre_integration_ImuPtr, ORI_MSGPtr>> System::Get_Prosess(){
    vector<pair<pre_integration_ImuPtr, ORI_MSGPtr>>Get_measurement;
    Get_measurement = measurements;
    measurements.clear();
    return Get_measurement;
}

void System::DynaTrack() {
    while(Track_begin) {

        vector<pair<pre_integration_ImuPtr, ORI_MSGPtr>>prosess_measurements;
        unique_lock<mutex> lk(m_buf);
        con.wait(lk, [&] {
            return ((prosess_measurements = Get_Prosess()).size()!=0);
        });
        //measurements = getMeasurements()).size() != 0;
        //prosess_measurements = measurements;
        //measurements.clear();
        for (auto measurement:prosess_measurements) {
            if (Img_12_buf.size() < Img_quene_len) {
                    Img_12_buf.push(measurement);
                    cout << "Img_12_buf.size() :   " << Img_12_buf.size() << endl;
                    continue;
            }else{
                    //begin to goTrack dynaObject;
                    cout<<"begin to track"<<endl;
                    if(first_cal_opt_flag){
                        cout<<"we begin to waiting"<<endl;
                        first_cal_opt_flag = false;
                        //differet lock means differet mutex;
                        unique_lock<mutex>lk(i_buf);
                        while(Seg_Conver)
                        {
                            cout << "我们正在等待首帧分割结果" << endl;
                            con.wait(lk);
                        }
                        pair<cv::Mat,double>Last_SFrame;
                        Last_SFrame = SegFrame;
                        SegFrame.first = Img_12_buf.back().second->img;
                        SegFrame.second = Img_12_buf.back().second->header;
                        Seg_Conver = true;
                        lk.unlock();
                        con.notify_all();

                        if(Seg_Conver){
                            cout<<"Seg_Conver is true"<<endl;
                        }

                        //transfer Img_12_buf to SegFrame;

                        //read Segframe and cal_front_opt;
                        //cal optical then creat object

                        //multi-thread is for accelerating opti-speed;
                        //read optical_flow for every InstaceMask;

                        cout<<"InstanceMask has been all read"<<endl;
                        cv::Mat BGMask = cv::Mat::zeros(375,1242,CV_32F); //Be careful with size!!;
                        trackerData[0].readMaskImage(Last_SFrame, InstanceMask.InsSeg_Mask, InstanceMask.MaskId,InstanceMask.pred_boxes,BGMask);
                        //no Seg just track;
                        trackerData[0].trackImg(Img_12_buf.front());
                    }else{
                        if (Img_12_buf.front().second->header == SegFrame.second) {
                            cout<<"we are just reading Mask_Img"<<endl;
                            //如果第一帧还没有分割完成,则进行等待;

                            unique_lock<mutex>lk(i_buf);
                            //判断帧已经分割完成；如果没有，则等待;
                            while(Seg_Conver) {
                                cout << "我们正在等待分割结果" << endl;
                                con.wait(lk);
                            }
                            //对分割帧进行更新；
                            pair<cv::Mat,double> Last_SFrame;
                            Last_SFrame = SegFrame;
                            SegFrame.first = Img_12_buf.back().second->img;
                            SegFrame.second = Img_12_buf.back().second->header;
                            Seg_Conver = true;
                            //lk.unlock();
                            //con.notify_all();


                            cv::Mat BGMask = cv::Mat::zeros(375,1242,CV_32F); //Be careful with size!!;
                            trackerData[0].readMaskImage(Last_SFrame, InstanceMask.InsSeg_Mask, InstanceMask.MaskId,InstanceMask.pred_boxes,BGMask);
                        }  //SegConver;
                        else {
                            cout<<"we just trackIMg"<<endl;
                            trackerData[0].trackImg(Img_12_buf.front());
                        }
                    }
                pair<pre_integration_ImuPtr,vector<shared_ptr<IMG_MSG>>>Cur_Objects;
                Cur_Objects.first = Img_12_buf.front().first;

                //first we update Object_ID;
                for (unsigned int i = 0;; i++)
                {
                    bool completed = false;
                    completed |= trackerData[0].update_Object_ID(i);;

                    if (!completed)
                        break;
                }
                trackerData[0].updateID(trackerData[0].Instance_Objects);
                //cout<<"ids in Object is " << trackerData[0].Instance_Objects[0].ids[2]<<endl;

                for(int i = 0 ;i<trackerData[0].Instance_Objects.size();i++) {
                    shared_ptr<IMG_MSG>feature_Objects(new IMG_MSG());
                    feature_Objects->header = Img_12_buf.front().second->header;
                    //record how much dyna_Objects; Instance_Objects[j].ids[i] = Instance_Objects[j].n_id++;
                    //cout<<"n_id is " << i <<endl;
                    feature_Objects->ids_for_Object = trackerData[0].ids[i];
                    //record class for objects;
                    feature_Objects->ids_for_class = trackerData[0].Instance_Objects[i].id_for_class;

                    auto &un_pts = trackerData[0].Instance_Objects[i].cur_un_pts;
                    auto &cur_pts = trackerData[0].Instance_Objects[i].cur_pts;
                    auto &ids = trackerData[0].Instance_Objects[i].ids;
                    auto &pts_velocity = trackerData[0].Instance_Objects[i].pts_velocity;
                    for (unsigned int j = 0; j < trackerData[0].Instance_Objects[i].ids.size(); j++) {
                        //cout<<" track_cnt "<<trackerData[0].Instance_Objects[i].track_cnt[j]<<endl;
                        if (trackerData[0].Instance_Objects[i].track_cnt[j] > 1) {

                            int p_id = ids[j];
                            //cout <<"id for point : "<< ids[j] <<endl;
                            double x = un_pts[j].x;
                            double y = un_pts[j].y;
                            double z = 1;
                            feature_Objects->points.push_back(Vector3d(x, y, z));
                            feature_Objects->id_of_point.push_back(p_id);
                            feature_Objects->u_of_point.push_back(cur_pts[j].x);
                            feature_Objects->v_of_point.push_back(cur_pts[j].y);
                            feature_Objects->velocity_x_of_point.push_back(pts_velocity[j].x);
                            feature_Objects->velocity_y_of_point.push_back(pts_velocity[j].y);
                            if(feature_Objects->points.back().z()!=1.0){
                                cout<<" z is "<< feature_Objects->points.back().z()<<endl;
                            }

                        }
                    }
                    Cur_Objects.second.push_back(feature_Objects);
                }

                cout<<"here we give the Object to feature_buf"<<endl;
                Instance_Objects.push(Cur_Objects); //get new measurements;
                cv::Mat show_img = Img_12_buf.front().second->img;
                cout<<"here we can see "<<trackerData[0].Instance_Objects.size()<<" Objects "<<endl;
                for(int i =0 ;i<trackerData[0].Instance_Objects.size();i++){
                    if( trackerData[0].Instance_Objects[i].id_for_class != -1){
                        //cout<<"here is one dyna_object: "<<"Size is "<< trackerData[0].Instance_Objects[i].cur_pts.size() <<endl;
                        cout<<" we have "<< trackerData[0].Instance_Objects[i].cur_pts.size()<<" features on img "<<endl;
                        for (unsigned int j = 0; j < trackerData[0].Instance_Objects[i].cur_pts.size(); j++)
                        {
                            cv::circle(show_img, trackerData[0].Instance_Objects[i].cur_pts[j], 2, cv::Scalar(0, 255, 255), 2);
                        }
                    }else{
                        cout<<" we have "<< trackerData[0].Instance_Objects[i].cur_pts.size()<<" features on Bgimg "<<endl;
                        for (unsigned int j = 0; j < trackerData[0].Instance_Objects[i].cur_pts.size(); j++)
                        {
                            cv::circle(show_img, trackerData[0].Instance_Objects[i].cur_pts[j], 2, cv::Scalar(255,0,255), 2);
                        }
                    }
                }
                //cv::cvtColor(Img_12_buf.front().second->img, show_img, CV_GRAY2RGB);


                cv::namedWindow("IMAGE", CV_WINDOW_AUTOSIZE);
                cv::imshow("IMAGE", show_img);
                cv::waitKey(0);

                Img_12_buf.pop();
                Img_12_buf.push(measurement);
            }  //begin to Track;
        }  // go through measurements;
    }
}

void System::SegImg()
{
    //语义分割的结果主要体现在稀疏特征点的选择上,理解该店也就理解了分割线程与特征提取线程的关系；
    //一个原则是buf中图像数据不能利用后面的分割数据，而只能被后续的分割数据矫正；
    while(bStart_Seg){
        unique_lock<mutex> lk(i_buf);
        //判断帧已经分割完成；如果没有，则等待;
        while(!Seg_Conver) {
            //cout << "我们正在img_thread" << endl;
            con.wait(lk);
        }
        lk.unlock();
        con.notify_all();
        if(Seg_Conver){
            //如果是第一次进行分割；进行DynSeg；
            //只有进行分割了，该标识才可以被置false；
            cout<<"Tic_toc when Seg works"<<t_r.toc()<<endl;
            InstanceMask = DynSeg->GetInsSeg(SegFrame.first);
            //DynMask = DynSeg->GetDynSeg(FirstImg.first);

            cout<<"here works？"<<endl;
            Seg_Conver = false; //Seg_Conver false 标识上一分割帧已经实现分割，需要进行更新；
            if(!InstanceMask.InsSeg_Mask.empty()){
                cout<<"InsSeg_Mask 已经不为空了"<<InstanceMask.InsSeg_Mask[0].size()<<endl;
            }
            con.notify_all();
        }
    }
};

void System::PubImuData(double dStampSec, const Eigen::Vector3d &vGyr, 
    const Eigen::Vector3d &vAcc,const Eigen::Vector3d &vVel)
{
    shared_ptr<IMU_MSG> imu_msg(new IMU_MSG());
	imu_msg->header = dStampSec;
	imu_msg->linear_acceleration = vAcc;
	imu_msg->angular_velocity = vGyr;
	imu_msg->velocity = vVel;

    if (dStampSec <= last_imu_t)
    {
        cout<<"dStampSec is "<<dStampSec<<endl;
        cout<<"last_imu_tis "<<last_imu_t<<endl;
        cerr << "imu message in disorder!" << endl;
        return;
    }
    last_imu_t = dStampSec;
    //cout<<"Tic_toc in IMu is "<<t_r.toc()<<endl;
    // cout << "1 PubImuData t: " << fixed << imu_msg->header
    //     << " acc: " << imu_msg->linear_acceleration.transpose()
    //     << " gyr: " << imu_msg->angular_velocity.transpose() << endl;
    m_buf.lock();
    imu_buf.push(imu_msg);
    // cout << "1 PubImuData t: " << fixed << imu_msg->header 
    //     << " imu_buf size:" << imu_buf.size() << endl;
    m_buf.unlock();
    con.notify_all();
}



// thread: visual-inertial odometry

void System::ProcessBackEnd()
{
    cout << "1 ProcessBackEnd start" << endl;
    while (bStart_backend)
    {

        vector<pair<pre_integration_ImuPtr,vector<shared_ptr<IMG_MSG>>>>Objects_measurement;

        unique_lock<mutex> lk(m_state);
        while(Instance_Objects.empty()){
            con.wait(lk);
        }
        for(int i = 0 ;i<Instance_Objects.size();i++){
            Objects_measurement.push_back(Instance_Objects.front());
            Instance_Objects.pop();
        }
        lk.unlock();
        con.notify_all();
        m_estimator.lock();

        for (auto & measurement : Objects_measurement)
        {
            TicToc t_endprocess;
            pre_integration_ImuPtr cur_pre_integration;
            cur_pre_integration = measurement.first;
            vector<shared_ptr<IMG_MSG>> cur_objects;
            cur_objects = measurement.second;

            //deal with Imu_information
            for(int i = 0; i < cur_pre_integration->acc_buf.size();i++){
                estimator.processIMU(cur_pre_integration->dt_buf[i], cur_pre_integration->acc_buf[i], cur_pre_integration->gyr_buf[i],cur_pre_integration->vel_buf[i]);
            }

            // deal with Img_information;
            // first deal with BG ground;
            shared_ptr<IMG_MSG>BG_objects;
            //the last one is BG ground;
            BG_objects = cur_objects.back();
            cout<<"we can see class of BG_Objects "<<BG_objects->ids_for_class;
            //cout<<"we can see ids for BG_Objects"<<BG_objects->ids_for_Object<<endl;
            TicToc t_processImage;
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> BG_image;
            //cout <<"number of BG_objects is  : "<< BG_objects->points.size() <<endl;
            for(int i = 0 ; i < BG_objects->points.size();i++){
                //first
                int id = BG_objects->id_of_point[i];

                //cout<< " id is "<< id <<endl;
                double x = BG_objects->points[i].x();
                double y = BG_objects->points[i].y();
                double z = BG_objects->points[i].z();//此时，所有的z值都设定为1；
                double p_u = BG_objects->u_of_point[i];
                double p_v = BG_objects->v_of_point[i];
                double velocity_x = BG_objects->velocity_x_of_point[i];
                double velocity_y = BG_objects->velocity_y_of_point[i];
                assert(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                BG_image[id].emplace_back(BG_objects->ids_for_class, xyz_uv_velocity);
                //cout<<"id is"<<id <<endl;
            }

            estimator.processImage(BG_image, BG_objects->header);

            //utilize camera model to cal dyna_objects;
            /*
            if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR){
                vector<map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>>dyna_objects_features;
                for(int i = 0;i<cur_objects.size()-1;i++){
                    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>object_features;
                    shared_ptr<IMG_MSG>dyna_object = cur_objects[i];
                    for(int i = 0 ;i < dyna_object->points.size();i++){
                        int id = BG_objects->id_of_point[i];
                        double x = BG_objects->points[i].x();
                        double y = BG_objects->points[i].y();
                        double z = BG_objects->points[i].z();//此时，所有的z值都设定为1；
                        double p_u = BG_objects->u_of_point[i];
                        double p_v = BG_objects->v_of_point[i];
                        double velocity_x = BG_objects->velocity_x_of_point[i];
                        double velocity_y = BG_objects->velocity_y_of_point[i];
                        if( z !=1 ){
                            cout<<"here is one on 1 z" <<endl;
                            continue;
                        }
                        //cout<<" here is i points" << i << " and the z is "<< z << endl;
                        //assert(z == 1);
                        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                        object_features[id].emplace_back(dyna_object->ids_for_Object,xyz_uv_velocity);
                        //cout<<"id is "<<id<<endl;
                        //cout<<"z is "<< z <<endl;
                    }

                    dyna_objects_features.push_back(object_features);
                }
                //just do after function;
                estimator.processObject(dyna_objects_features,BG_objects->header);
            }*/




            if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            {
                Vector3d p_wi;
                Quaterniond q_wi;
                q_wi = Quaterniond(estimator.Rs[WINDOW_SIZE]);
                p_wi = estimator.Ps[WINDOW_SIZE];
                vPath_to_draw.push_back(p_wi);
                double dStamp = estimator.Headers[WINDOW_SIZE];
                cout << "1 BackEnd processImage dt: " << fixed << t_processImage.toc() << " stamp: " <<  dStamp << " p_wi: " << p_wi.transpose() << endl;
                ofs_pose << fixed << dStamp << " " << p_wi(0) << " " << p_wi(1) << " " << p_wi(2) << " " 
                         << q_wi.w() << " " << q_wi.x() << " " << q_wi.y() << " " << q_wi.z() << endl;
            }
            cout<<"endprocess one round cost :  "<<t_endprocess.toc()<<" .ms "<<endl;
        }
        m_estimator.unlock();
    }
}


void System::Draw() 
{   
    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    s_cam = pangolin::OpenGlRenderState(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 384, 0.1, 1000),
            pangolin::ModelViewLookAt(-5, 0, 15, 7, 0, 0, 1.0, 0.0, 0.0)
    );

    d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    // pangolin::OpenGlRenderState s_cam(
    //         pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 384, 0.1, 1000),
    //         pangolin::ModelViewLookAt(-5, 0, 15, 7, 0, 0, 1.0, 0.0, 0.0)
    // );

    // pangolin::View &d_cam = pangolin::CreateDisplay()
    //         .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
    //         .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(0.75f, 0.75f, 0.75f, 0.75f);
        glColor3f(0, 0, 1);
        pangolin::glDrawAxis(3);
         
        // draw poses
        glColor3f(0, 0, 0);
        glLineWidth(2);
        glBegin(GL_LINES);
        int nPath_size = vPath_to_draw.size();
        for(int i = 0; i < nPath_size-1; ++i)
        {        
            glVertex3f(vPath_to_draw[i].x(), vPath_to_draw[i].y(), vPath_to_draw[i].z());
            glVertex3f(vPath_to_draw[i+1].x(), vPath_to_draw[i+1].y(), vPath_to_draw[i+1].z());
        }
        glEnd();
        
        // points
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
        {
            glPointSize(5);
            glBegin(GL_POINTS);
            for(int i = 0; i < WINDOW_SIZE+1;++i)
            {
                Vector3d p_wi = estimator.Ps[i];
                glColor3f(1, 0, 0);
                glVertex3d(p_wi[0],p_wi[1],p_wi[2]);
            }
            glEnd();
        }
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
}

#ifdef __APPLE__
void System::InitDrawGL() 
{   
    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    s_cam = pangolin::OpenGlRenderState(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 384, 0.1, 1000),
            pangolin::ModelViewLookAt(-5, 0, 15, 7, 0, 0, 1.0, 0.0, 0.0)
    );

    d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));
}

void System::DrawGLFrame() 
{  

    if (pangolin::ShouldQuit() == false)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(0.75f, 0.75f, 0.75f, 0.75f);
        glColor3f(0, 0, 1);
        pangolin::glDrawAxis(3);
            
        // draw poses
        glColor3f(0, 0, 0);
        glLineWidth(2);
        glBegin(GL_LINES);
        int nPath_size = vPath_to_draw.size();
        for(int i = 0; i < nPath_size-1; ++i)
        {        
            glVertex3f(vPath_to_draw[i].x(), vPath_to_draw[i].y(), vPath_to_draw[i].z());
            glVertex3f(vPath_to_draw[i+1].x(), vPath_to_draw[i+1].y(), vPath_to_draw[i+1].z());
        }
        glEnd();
        
        // points
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
        {
            glPointSize(5);
            glBegin(GL_POINTS);
            for(int i = 0; i < WINDOW_SIZE+1;++i)
            {
                Vector3d p_wi = estimator.Ps[i];
                glColor3f(1, 0, 0);
                glVertex3d(p_wi[0],p_wi[1],p_wi[2]);
            }
            glEnd();
        }
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
}
#endif

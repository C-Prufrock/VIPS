#include "System.h"

#include <pangolin/pangolin.h>


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

    SegMask.clear();


    m_estimator.lock();
    estimator.clearState();
    m_estimator.unlock();

    ofs_pose.close();
}


void System::PubImageData(double dStampSec, Mat &img)
{
    //PubImageData图像只传输图像；而非分割结果；
    //因为实例分割网络的结果与后端处理进程直接相关，因此


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
    last_image_time = dStampSec;
    // frequency control
    if (round(1.0 * pub_count / (dStampSec - first_image_time)) <= FREQ)
    {
        PUB_THIS_FRAME = true;
        // reset the frequency control
        if (abs(1.0 * pub_count / (dStampSec - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = dStampSec;
            pub_count = 0;
        }
    }
    else
    {
        PUB_THIS_FRAME = false;
    }
    //first_cal_opt_image_time  以及第一次计算光流的bool值；用来对齐Img与Imu数据；
    //只有第一次分割的时候，对Img_12_buf进行首帧分割；

    //只是处理首帧时运行,光流提取首帧也是分割的首帧；
    if(first_image_imread)
    {
        //该代码块只负责把首帧传递给doSeg线程；
       FirstImg.first =img;
       FirstImg.second = dStampSec;
       Img_12_buf.push(FirstImg);
       SegFrame = FirstImg;
       first_image_imread = false;
       return;
       //此时应根据Mask进行光流点提取,便于后续图像跟踪；
       //每一帧读取的Mask应是首帧分割的Mask；
       //trackerData[0].readSegImage(Img_12_buf.front());
    }
    //如果Img_12_buf的数量不足12,不要进行跟踪，填满，填满的时间为delay时间；
    if(Img_12_buf.size()<Img_quene_len)
    {
        pair<cv::Mat,double>Image;
        Image.first  = img;
        Image.second = dStampSec;
        Img_12_buf.push(Image);
        cout<<"Img_12_buf.size() :   "<< Img_12_buf.size() <<endl;
        //只添加不处理；
        return;
        //此处初始化完成；
    }else{
        cout<<"Segging and Tracking begin to works"<<endl;
        //step1 ：等待DynaMask不为空；应维持两个DynaMask;
        //判断是否是第一次分割，应用分割结果读取光流；
        if(first_image_seg){
            //如果第一帧还没有分割完成,则进行等待。
            unique_lock<mutex>lk(m_buf);
            //判断帧已经分割完成；如果没有，则等待；
            while(!first_image_seg_in_seg)
            {
                //cout<<"我们正在等待分割结果"<<endl;
                con.wait(lk);}
            lk.unlock();
            first_image_seg = false;
            trackerData[0].readMaskImage(Img_12_buf.front(),InstanceMask);
        }else{
            //分割完之后才可以把Img_12_buf中序列的末帧传递到SegImg;
            //暂时忽略哪个花费时间的长或者短
            //不过是两种情况，一个是分割时间大于12帧光流跟踪花费时间；如果SegFrame不是一个序列（用来等待分割完成）,则认为分割时间大于序列的光流时间；
            //一个则是分割时间小于12帧光流花费时间；
            // 当Img_12_buf首帧等于上一帧分割帧时
            //当前Img_12_buf中quene的末帧将被视更视为新的分割帧；分割帧更新；
            if( SegFrame.second ==  Img_12_buf.front().second ){
                //此时跟踪帧为分割帧；Mask被重置；
                //此时需要对SegFrame进行更新；此时需要标识符来认识这种更新；
                unique_lock<mutex>lk(m_buf);
                ///需要等待的时候应该是上一帧分割帧已经进入了等待序列；
                while(Seg_Conver)
                {
                    cout<<"你电脑分割的时候比较慢，12帧光流提取已经完成，但分割未完成，可通过改变Img_quene_len,考虑增加quene长度；"<<endl;
                    con.wait(lk);}
                lk.unlock();
                SegFrame = Img_12_buf.back();  //对分割帧进行更新；
                Seg_Conver = true;//true说明出现了需要分割的帧;需要判断上一帧是否已经完成分割;
                //分割帧改变了；
                //如此更新 需要保证SegFrame的数据安全；
                //运动物体特征提取方式肯定要区别于背景；旺旺区域很小，需要重新提取图像；
                //运动物体的判定必然来自分割帧:背后的逻辑是,如果是回溯运动物体上的特征点,而必须利用稠密或半稠密的提取方式方能实现回溯判定；
                //两种方法：1.对所有Mask内的光流进行多个提取，区域过小（光流点过少的此帧不做判定；）.然后利用Imu预积分信息进行运动判定；
                //2 不针对Mask做光流提取,根据后端判定后（有可能Mask中特征点过少，导致判定不准确）
                //因此，必须针对Mask提取特征点；
                cv::Mat BGMask;
                for(int i =0;i<InstanceMask.MaskId.size();i++){
                    //此处保留的操作为
                    //第三个是什么物体，第四个是第几个物体..
                    //＋１表明物体的种类从１开始，０留给背景；
                    trackerData[0].readMaskImage(Img_12_buf.front(),InstanceMask.InsSeg_Mask[i],InstanceMask.MaskId[i],i+1);
                    BGMask = BGMask + InstanceMask.InsSeg_Mask[i];
                }
                //背景永远是第一个物体，在各个feature的vector中是第一，而其物体类别为-1；
                trackerData[0].readMaskImage(Img_12_buf.front(),BGMask,-1,0);

            }else{
                //此时出现帧为跟踪帧；
                cv::Mat mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
                trackerData[0].readMaskImage(Img_12_buf.front(),mask);
            }

        }
        cout<<"begin to do quene"<<endl;
        //每进入一个新帧,对Img_12_buf的首帧进行特征提取；
        //读取图像之后，删除Img_12_buf内的信息；
        Img_12_buf.pop();
        //并把当前帧信息加入到quene序列中；
        pair<cv::Mat,double>Image;
        Image.first  = img;
        Image.second = dStampSec;
        Img_12_buf.push(Image);
        //利用Mask和首帧
    }
    /*
     * for(int n =0;;n++){
     * for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        completed |= trackerData[0].updateID(i,n);

        if (!completed)
            break;
    }
     *
     *
     * }

    if (PUB_THIS_FRAME)
    {
        pub_count++;
        shared_ptr<IMG_MSG> feature_points(new IMG_MSG());
        feature_points->header = dStampSec;
        vector<set<int>> hash_ids(NUM_OF_CAM);
        for (int i = 0; i < NUM_OF_CAM; i++)//虽然采用循环，但由于是单目，仅采用一次；
        {
            auto &un_pts = trackerData[i].cur_un_pts;//畸变点；
            auto &cur_pts = trackerData[i].cur_pts;//经矫正的点；
            auto &ids = trackerData[i].ids;       //
            auto &pts_velocity = trackerData[i].pts_velocity;
            for (unsigned int j = 0; j < ids.size(); j++)
            {
                if (trackerData[i].track_cnt[j] > 1)
                {
                    int p_id = ids[j];
                    hash_ids[i].insert(p_id);
                    double x = un_pts[j].x;
                    double y = un_pts[j].y;
                    double z = 1;
                    feature_points->points.push_back(Vector3d(x, y, z));
                    feature_points->id_of_point.push_back(p_id * NUM_OF_CAM + i);
                    feature_points->u_of_point.push_back(cur_pts[j].x);
                    feature_points->v_of_point.push_back(cur_pts[j].y);
                    feature_points->velocity_x_of_point.push_back(pts_velocity[j].x);
                    feature_points->velocity_y_of_point.push_back(pts_velocity[j].y);
                }
            }
            if (!init_pub)
            {
                cout << "4 PubImage init_pub skip the first image!" << endl;
                first_cal_opt_image_time = feature_points->header;
                cout<<" we can see first ImgTimeStamp: "<<setprecision(10)<<first_cal_opt_image_time<<endl;
                init_pub = 1;
            }
            else
            {
                m_buf.lock();
                feature_buf.push(feature_points);
                // cout << "5 PubImage t : " << fixed << feature_points->header
                //     << " feature_buf size: " << feature_buf.size() << endl;
                m_buf.unlock();
                con.notify_one();
            }
        }
    }*/
         /*
#ifdef __linux__
    cv::Mat show_img;
	cv::cvtColor(img, show_img, CV_GRAY2RGB);
	if (SHOW_TRACK)
	{
		for (unsigned int j = 0; j < trackerData[0].cur_pts.size(); j++)
        {
			double len = min(1.0, 1.0 * trackerData[0].track_cnt[j] / WINDOW_SIZE);
			cv::circle(show_img, trackerData[0].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
		}

        cv::namedWindow("IMAGE", CV_WINDOW_AUTOSIZE);
		cv::imshow("IMAGE", show_img);
        cv::waitKey(1);
	}
#endif    
   */
}

void System::SegImg()
{
    //语义分割的结果主要体现在稀疏特征点的选择上,理解该店也就理解了分割线程与特征提取线程的关系；
    //一个原则是buf中图像数据不能利用后面的分割数据，而只能被后续的分割数据矫正；
    while(bStart_Seg){
        //cout<<"Tic_toc in Seg_th"<<t_r.toc()<<endl;
        //首帧分割；
        if(!first_image_seg_in_seg && !FirstImg.first.empty()){
            //如果是第一次进行分割；进行DynSeg；
            //此时如果Img_12_buf中已经读入图像，则进行分割；
            //只有进行分割了，该标识才可以被置false；
            //InstanceMask = DynSeg->GetInsSeg(SegFrame.first);
            //DynMask = DynSeg->GetDynSeg(FirstImg.first);
            first_image_seg_in_seg = true;  //first_image_seg_in_seg 用来表征是否第一次进行分割；
            cout<<"Tic_toc when Seg works"<<t_r.toc()<<endl;
            cout<<"here works？"<<endl;
            Seg_Conver = false; //Seg_Conver false 标识上一分割帧已经实现分割，需要进行更新；
            //cv::imshow("Fist Seg Img2:",DynMask);
            //cv::waitKey(3000);

            if(!InstanceMask.InsSeg_Mask.empty()){
                cout<<"InsSeg_Mask 已经不为空了"<<DynMask.size()<<endl;
            }
            con.notify_all();
        }else{
            //进行实例分割；
            //如果分割帧已经实现了传递;则进行新的分割；
            if(Seg_Conver){
                //DynMask = DynSeg->GetDynSeg(SegFrame.first);
                InstanceMask = DynSeg->GetInsSeg(SegFrame.first);
                Seg_Conver = false;
                con.notify_all();
            }
        }
    }
};

vector<pair<vector<ImuConstPtr>, ImgConstPtr>> System::getMeasurements()
{
    vector<pair<vector<ImuConstPtr>, ImgConstPtr>> measurements;

    while (true)
    {
        if (imu_buf.empty() || feature_buf.empty())
        {
            // cerr << "1 imu_buf.empty() || feature_buf.empty()" << endl;
            return measurements;
        }

        if (!(imu_buf.back()->header > feature_buf.front()->header + estimator.td))
        {
            cerr << "wait for imu, only should happen at the beginning sum_of_wait: " 
                << sum_of_wait << endl;
            sum_of_wait++;
            return measurements;
        }

        if (!(imu_buf.front()->header < feature_buf.front()->header + estimator.td))
        {
            cerr << "throw img, only should happen at the beginning" << endl;
            feature_buf.pop();
            continue;
        }
        ImgConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        vector<ImuConstPtr> IMUs;
        ImuConstPtr LastImuForFirstImage;
        shared_ptr<IMU_MSG> FirstImuForFirstImage(new IMU_MSG());
        //则实际得到的第一imu为差值结果；
        if (first_cal_opt_flag)//此处为实际计算的第一幅图像；
        {
            cout << "最开始的时候有大量与Img无关的imu数据需要剔除掉" << endl;
            first_cal_opt_flag = false;
            while (imu_buf.front()->header < first_cal_opt_image_time + estimator.td){
                //找到第一个大于第一个Img时间戳的图像；
                LastImuForFirstImage = imu_buf.front();
                imu_buf.pop();
            }
            FirstImuForFirstImage->header = first_cal_opt_image_time + estimator.td ;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            double dt_1 =  first_cal_opt_image_time - LastImuForFirstImage->header;
            double dt_2 = imu_buf.front()->header  - first_cal_opt_image_time;
            assert(dt_1 >= 0);
            assert(dt_2 >= 0);
            assert(dt_1 + dt_2 > 0);

            double w1 = dt_2 / (dt_1 + dt_2);
            double w2 = dt_1 / (dt_1 + dt_2);

            //利用相机时间戳 与相邻的两个imu数据时间戳对相机处时间imu数据进行差值估计；
            FirstImuForFirstImage->linear_acceleration.x() =  w2 * imu_buf.front()->linear_acceleration.x();
            FirstImuForFirstImage->linear_acceleration.y() =  w2 * imu_buf.front()->linear_acceleration.y();
            FirstImuForFirstImage->linear_acceleration.z() =  w2 * imu_buf.front()->linear_acceleration.z();
            FirstImuForFirstImage->angular_velocity.x() =  w2 * imu_buf.front()->angular_velocity.x();
            FirstImuForFirstImage->angular_velocity.y() =  w2 * imu_buf.front()->angular_velocity.y();
            FirstImuForFirstImage->angular_velocity.z() =  w2 * imu_buf.front()->angular_velocity.z();
            IMUs.emplace_back(FirstImuForFirstImage);
        }

        while (imu_buf.front()->header < img_msg->header + estimator.td)
        {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        // cout << "1 getMeasurements IMUs size: " << IMUs.size() << endl;
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty()){
            cerr << "no imu between two image" << endl;
        }
        // cout << "1 getMeasurements img t: " << fixed << img_msg->header
        //     << " imu begin: "<< IMUs.front()->header 
        //     << " end: " << IMUs.back()->header
        //     << endl;
        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

void System::PubImuData(double dStampSec, const Eigen::Vector3d &vGyr, 
    const Eigen::Vector3d &vAcc)
{
    shared_ptr<IMU_MSG> imu_msg(new IMU_MSG());
	imu_msg->header = dStampSec;
	imu_msg->linear_acceleration = vAcc;
	imu_msg->angular_velocity = vGyr;

    if (dStampSec <= last_imu_t)
    {
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
    con.notify_one();
}

// thread: visual-inertial odometry
/*
void System::ProcessBackEnd()
{
    cout << "1 ProcessBackEnd start" << endl;
    while (bStart_backend)
    {
        // cout << "1 process()" << endl;
        vector<pair<vector<ImuConstPtr>, ImgConstPtr>> measurements;
        
        unique_lock<mutex> lk(m_buf);
        con.wait(lk, [&] {
            return (measurements = getMeasurements()).size() != 0;
        });
        if( measurements.size() > 1){
        cout << "1 getMeasurements size: " << measurements.size() 
            << " imu sizes: " << measurements[0].first.size()
            << " feature_buf size: " <<  feature_buf.size()
            << " imu_buf size: " << imu_buf.size() << endl;
        }
        lk.unlock();
        m_estimator.lock();
        for (auto &measurement : measurements)
        {
            TicToc t_endprocess;
            auto img_msg = measurement.second;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            cout<<"first know measurement.first.size"<<measurement.first.size()<<endl;
            for (auto &imu_msg : measurement.first)//measurement.first的第一要素是vectro<ImuConsPtr>// 的地址；
            {
                cout<<"imu_msg->header  :"<<setprecision(10) <<imu_msg->header;
                double t = imu_msg->header;
                double img_t = img_msg->header + estimator.td;
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
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
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
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x();
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y();
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z();
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x();
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y();
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z();
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }

            // cout << "processing vision data with stamp:" << img_msg->header 
            //     << " img_msg->points.size: "<< img_msg->points.size() << endl;

            // TicToc t_s;
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;

            cout<<"img_msg - >stamp :"<<img_msg->header<<endl;
            for (unsigned int i = 0; i < img_msg->points.size(); i++) 
            {
                int v = img_msg->id_of_point[i] + 0.5;
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                double x = img_msg->points[i].x();
                double y = img_msg->points[i].y();
                double z = img_msg->points[i].z();//此时，所有的z值都设定为1；
                double p_u = img_msg->u_of_point[i];
                double p_v = img_msg->v_of_point[i];
                double velocity_x = img_msg->velocity_x_of_point[i];
                double velocity_y = img_msg->velocity_y_of_point[i];
                assert(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id, xyz_uv_velocity);
            }
            TicToc t_processImage;
            estimator.processImage(image, img_msg->header);
            
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
*/
/*
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
*/
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

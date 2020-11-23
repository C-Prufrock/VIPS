#include "feature_tracker.h"



int FeatureTracker::n_id = 0;

//判断点经过畸变校正后是否还在图像空间内；
bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

/* 测试代码，用来查看究竟是否利用了Mask中的特征点；
bool inMask(cv::Mat Mask,vector<cv::Point2f>points){
    cout<<"inMask works  "<<points.size()<<endl;
    for(int i=0;i<points.size();i++){
        if(Mask.at<unsigned char>(points[i].y,points[i].x)!=255){
            swap(*(std::begin(points)+i),*(std::end(points)-1));
            points.pop_back();
            cout<<"vector delte one"<<endl;
        }else{
            continue;
        }
    }
    cout<<"inMask endl"<<endl;
}*/
//根据检测的状态【如畸变处理后是否在图像中，如是否通过F矩阵检验。】
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    //只保留状态良好的光流点；
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}


FeatureTracker::FeatureTracker()
{
}

void FeatureTracker::setMask(cv::Mat Mask,int n)
{
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = Mask;
    
     //此处直接添加实例分割代码；

    // prefer to keep features that are tracked for long time
    //第一帧的时候,forw_pts.size()为0,mask为全部.
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;
    //cout<<"第一次看一下forw_pts.size()"<<forw_pts.size()<<endl;
    //遍历物体的特征点；
    for (unsigned int i = 0; i < forw_pts[n].p.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[n][i], make_pair(forw_pts[n].p[i], ids[n][i])));

    //第二帧图像进入时，所有特征点的跟踪个数相同;
    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    forw_pts[n].p.clear();
    ids[n].clear();
    track_cnt[n].clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {

            forw_pts[n].p.push_back(it.second.first);
            ids[n].push_back(it.second.second);
            track_cnt[n].push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

//如果跟踪的角点数目比较小，则加点；
void FeatureTracker::addPoints(int m,int n)
{
    for (auto &p : n_pts)
    {
        //将当前帧探测到的特征点加到当前序列中；track_cnt记录的跟踪角点的数量；ids初始化的时候记录的是（-1）

        forw_pts[n].p.push_back(p); //保留位置；
        forw_pts[n].m = m;   //什么类；
        forw_pts[n].n = n;   //第几个物体；
        ids[n].push_back(-1);     //初始化为-1
        track_cnt[n].push_back(1); //加入1；
    }
}

//读图，更新光流点，如果点数小于固定值，则添加角点；进行畸变矫正；计算x，y方向上的光流速度；
/*
void FeatureTracker::readImage(pair<const cv::Mat&, double>Img)
{
    TicToc ImageProcess;
    cv::Mat _img,img;
    cur_time = Img.second;

    //是否对图像进行均衡化处理？

    cvtColor(Img.first,_img,CV_BGR2GRAY);
    cout<<"_Img.type(): "<<_img.type()<<endl;
    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        //TicToc t_c;
        //cout<<"here works 1"<<endl;
        clahe->apply(_img, img);
        //cout<<"here works 2"<<endl;
        //ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;

    //如果是第一帧，则prev_img cur_img forw_img三者为同一图像；
    //第一幅的实例分割图进入时，forw_img必然为空；
    if (forw_img.empty())
    {
        prev_img = cur_img = forw_img = img;
        cout<<"第一副图像进来了！！"<<endl;
    }
    else //如果forw_img不空，说明已经读入有效的图像了；如果是第一次读图；
    {
        forw_img = img;
    }
    cout<<"Opticalflow cost "<<ImageProcess.toc()<<endl;
    forw_pts.clear();

    //cv::waitKeyEx(0);

    //该代码为进入的第三帧才计算光流，因为假劣init_feature标识，ROS版本则为第二帧；cout<<"when do you cal the optic and get cur_pts?"<<endl;
    //一系列的2D特征；如果cur_pts没有特征，说明是第一帧进入（有可能是重定位？）则跳过；
    //此时，因为第一帧图像不提取cur_pts光流点；所以，第二帧图像进入时，才进行计算；
    if (cur_pts.size() > 0)
    {
        //cout<<"here get cur_pts2 ?"<<endl;
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;

        //计算光流、3层金字塔，搜索空间为[21,21];
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;

        reduceVector(prev_pts, status);//会将测量量反退到上一帧；
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);

        //  ids 如何初始化？
        //cur_un_pts 如何初始化？
        //track_cnt如何初始化？

        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
        //ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }

    for (auto &n : track_cnt)
        n++;

    //PUB_THIS_FRAME 由帧率决定；
    if (PUB_THIS_FRAME)
    {
        //rejectWithF成立的同时，cur_pts也成立；
        //cout<<"here get cur_pts3 ?"<<endl;
        rejectWithF();
        //ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();
        //ROS_DEBUG("set mask costs %fms", t_m.toc());
        //cout<<"Mask.type is "<<Mask.type()<<endl;
        //cv::imshow("Mask is ",mask);
        //cv::waitKey(0);
        //cout<<"Mask.type is "<<Mask.type()<<endl;
        //ROS_DEBUG("detect feature begins");
        TicToc t_t;
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                //cout<<"CV_8UC1 is "<<CV_8UC1<<endl;
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            //goodFeaturesToTrack意味着追寻好的可跟踪点；第一个为图像，第二个检测的角点；第三：角点数目最大值；
            //第四：角点的品质因子；MinDistance 对于初选出来的角点，如果周围minDistance范围内存在其他更强角点，则将此角点删除；
            //mask：指定感兴趣区域，不需要在整副图像中寻找；
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
            //mask是需要的,但对于后续帧中提取的goodFeatures需要检验是否在上一帧中的背景区域需要进行判断；
            //上述逻辑决定了，对于当前帧，我们生成新的特征点的时候，要对其在上一分割帧中的位置进行判定
            //而新的分割mask到来时,我们则对其在last-mask与cur_mask两个分割帧中进行位置判定。（不仅在good_feature，也对于上一帧中的跟踪数据）
            //这是设计结果，不是观察结果。如果发现跟踪结果仅跟踪一个mask即有较好的结果,则不需要。

        }
        else
            n_pts.clear();
        //ROS_DEBUG("detect feature costs: %fms", t_t.toc());


        //ROS_DEBUG("add feature begins");
        TicToc t_a;
        addPoints();
        cv::cvtColor(img,img, CV_GRAY2RGB);
        //cout<<"获得了 "<<n_pts.size()<<"个特征"<<endl;

        for(int j=0;j<n_pts.size();j++)
        {
            cv::circle(img, forw_pts[j], 3, cv::Scalar(255, 0, 255),2);

        //inMask(Mask,n_pts);
        //cout<<"经过Mask检验 n_pts的size为"<<n_pts.size()<<endl;
        //cv::namedWindow("IMAGE", CV_WINDOW_AUTOSIZE);
        //cv::imshow("IMAGE",img);
        //cv::waitKey(0);
        //ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }

    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;

    //状态转移；
    cur_img = forw_img;
    cur_pts = forw_pts;
    undistortedPoints();
    prev_time = cur_time;
    cout<<"Opticalflow cost "<<ImageProcess.toc()<<endl;
}*/
/*void FeatureTracker::readDynkImage(pair<const cv::Mat&, double>Img,cv::Mat Mask,int m,int n)
{
    TicToc ImageProcess;
    cv::Mat _img,img;
    cur_time = Img.second;

    //是否对图像进行均衡化处理？

    cvtColor(Img.first,_img,CV_BGR2GRAY);
    cout<<"_Img.type(): "<<_img.type()<<endl;
    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        //TicToc t_c;
        //cout<<"here works 1"<<endl;
        clahe->apply(_img, img);
        //cout<<"here works 2"<<endl;
        //ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;

    //如果是第一帧，则prev_img cur_img forw_img三者为同一图像；
    //第一幅的实例分割图进入时，forw_img必然为空；
    if (forw_img.empty())
    {
        prev_img = cur_img = forw_img = img;
        cout<<"第一副图像进来了！！"<<endl;
    }
    else //如果forw_img不空，说明已经读入有效的图像了；如果是第一次读图；
    {
        forw_img = img;
    }
    cout<<"Opticalflow cost "<<ImageProcess.toc()<<endl;
    forw_pts.clear();

    //cv::waitKeyEx(0);

    //该代码为进入的第三帧才计算光流，因为假劣init_feature标识，ROS版本则为第二帧；cout<<"when do you cal the optic and get cur_pts?"<<endl;
    //一系列的2D特征；如果cur_pts没有特征，说明是第一帧进入（有可能是重定位？）则跳过；
    //此时，因为第一帧图像不提取cur_pts光流点；所以，第二帧图像进入时，才进行计算；
    if (cur_pts.size() > 0)
    {
        //cout<<"here get cur_pts2 ?"<<endl;
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;

        //计算光流、3层金字塔，搜索空间为[21,21];
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;

        reduceVector(prev_pts, status);//会将测量量反退到上一帧；
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);

        //  ids 如何初始化？
        //cur_un_pts 如何初始化？
        //track_cnt如何初始化？

        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
        //ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }

    for (auto &n : track_cnt)
        n++;

    //PUB_THIS_FRAME 由帧率决定；
    if (PUB_THIS_FRAME)
    {
        //rejectWithF成立的同时，cur_pts也成立；
        //cout<<"here get cur_pts3 ?"<<endl;
        rejectWithF();
        //ROS_DEBUG("set mask begins");
        TicToc t_m;
        Mask.convertTo(Mask, CV_8UC1, 255.0);

        int m = Mask.cols;
        int n = Mask.rows;
        for(int i =0;i<n;i++)
        {
            for(int j =0;j<m;j++){
                cout<<" "<<(int)Mask.at<unsigned char>(i,j)<<",";
            }
            cout<<endl;
        }
        bitwise_not(Mask,Mask);

        setMask(Mask);
        //ROS_DEBUG("set mask costs %fms", t_m.toc());
        //cout<<"Mask.type is "<<Mask.type()<<endl;
        //cv::imshow("Mask is ",mask);
        //cv::waitKey(0);
        //cout<<"Mask.type is "<<Mask.type()<<endl;
        //ROS_DEBUG("detect feature begins");
        TicToc t_t;
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                //cout<<"CV_8UC1 is "<<CV_8UC1<<endl;
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            //goodFeaturesToTrack意味着追寻好的可跟踪点；第一个为图像，第二个检测的角点；第三：角点数目最大值；
            //第四：角点的品质因子；MinDistance 对于初选出来的角点，如果周围minDistance范围内存在其他更强角点，则将此角点删除；
            //mask：指定感兴趣区域，不需要在整副图像中寻找；
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
            //mask是需要的,但对于后续帧中提取的goodFeatures需要检验是否在上一帧中的背景区域需要进行判断；
            //上述逻辑决定了，对于当前帧，我们生成新的特征点的时候，要对其在上一分割帧中的位置进行判定
            //而新的分割mask到来时,我们则对其在last-mask与cur_mask两个分割帧中进行位置判定。（不仅在good_feature，也对于上一帧中的跟踪数据）
            //这是设计结果，不是观察结果。如果发现跟踪结果仅跟踪一个mask即有较好的结果,则不需要。

        }
        else
            n_pts.clear();
        //ROS_DEBUG("detect feature costs: %fms", t_t.toc());


        //ROS_DEBUG("add feature begins");
        TicToc t_a;
        addPoints();
        cv::cvtColor(img,img, CV_GRAY2RGB);
        //cout<<"获得了 "<<n_pts.size()<<"个特征"<<endl;

        for(int j=0;j<n_pts.size();j++)
        {
            cv::circle(img, forw_pts[j], 3, cv::Scalar(255, 0, 255),2);
        }
        //inMask(Mask,n_pts);
        //cout<<"经过Mask检验 n_pts的size为"<<n_pts.size()<<endl;
        //cv::namedWindow("IMAGE", CV_WINDOW_AUTOSIZE);
        //cv::imshow("IMAGE",img);
        //cv::waitKey(0);
        //ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;

    //状态转移；
    cur_img = forw_img;
    cur_pts = forw_pts;
    undistortedPoints();
    prev_time = cur_time;
    cout<<"Opticalflow cost "<<ImageProcess.toc()<<endl;
}*/
void FeatureTracker::readMaskImage(pair<const cv::Mat&, double>Img,cv::Mat Mask,int m,int n)
{
    TicToc ImageProcess;
    cv::Mat _img,img;
    cur_time = Img.second;

    //遇到新的分割帧如何处理？
    //如何建立新的分割对象与旧的分割对象之间的关系？
    //是创建新的Object，还是统一到旧的Object中？

    //是否对图像进行均衡化处理？

    cvtColor(Img.first,_img,CV_BGR2GRAY);
    cout<<"_Img.type(): "<<_img.type()<<endl;
    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        //TicToc t_c;
        //cout<<"here works 1"<<endl;
        clahe->apply(_img, img);
        //cout<<"here works 2"<<endl;
        //ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;

    //如果是第一帧，则prev_img cur_img forw_img三者为同一图像；
    //第一幅的实例分割图进入时，forw_img必然为空；
    if (forw_img.empty())
    {
        prev_img = cur_img = forw_img = img;
        cout<<"第一副图像进来了！！"<<endl;
    }
    else //如果forw_img不空，说明已经读入有效的图像了；如果是第一次读图；
    {
        forw_img = img;
    }
    cout<<"Opticalflow cost "<<ImageProcess.toc()<<"n is "<< n <<endl;

    Object cur_object;

    //forw_pts[n].p.clear();
    if(forw_pts.size()>=n+1){
        //已经出现过的物体；已经提取过焦点,需要进行光流跟踪；
        forw_pts[n].p.clear();
    }else{
        //新增加的物体;
        forw_pts.push_back(cur_object);
        cur_pts.push_back(cur_object);
    }

    //cv::waitKeyEx(0);

    //该代码为进入的第三帧才计算光流，因为假劣init_feature标识，ROS版本则为第二帧；cout<<"when do you cal the optic and get cur_pts?"<<endl;
    //一系列的2D特征；如果cur_pts没有特征，说明是第一帧进入（有可能是重定位？）则跳过；
    //此时，因为第一帧图像不提取cur_pts光流点；所以，第二帧图像进入时，才进行计算；
    if (cur_pts[n].p.size() > 0)
    {
        cout<<"here get cur_pts2 ?"<<endl;
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;

        //计算光流、3层金字塔，搜索空间为[21,21];
        cv::calcOpticalFlowPyrLK(cur_img, forw_img,cur_pts[n].p, forw_pts[n].p, status, err, cv::Size(21, 21), 3);

        for (int i = 0; i < int(forw_pts[n].p.size()); i++)
            if (status[i] && !inBorder(forw_pts[n].p[i]))
                status[i] = 0;

        reduceVector(prev_pts[n].p, status);//会将测量量反退到上一帧；
        reduceVector(cur_pts[n].p, status);
        reduceVector(forw_pts[n].p, status);

        //  ids 如何初始化？
        //cur_un_pts 如何初始化？
        //track_cnt如何初始化？

        reduceVector(ids[n], status);
        reduceVector(cur_un_pts[n].p, status);
        reduceVector(track_cnt[n], status);
        //ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }
    //利用其内部变量，也需要进行初始化；
    cout<<"here works 2"<<endl;
    vector<int>track_cnt[n];
    for (auto &in : track_cnt[n])
        in++;
    cout<<"here works 2"<<endl;
    //PUB_THIS_FRAME 由帧率决定；
    if (PUB_THIS_FRAME)
    {
        cout<<"here works 2"<<endl;
        //rejectWithF成立的同时，cur_pts也成立；
        //cout<<"here get cur_pts3 ?"<<endl;
        rejectWithF(n);
        //ROS_DEBUG("set mask begins");
        TicToc t_m;
        Mask.convertTo(Mask, CV_8UC1, 255.0);
        /*
        int m = Mask.cols;
        int n = Mask.rows;
        for(int i =0;i<n;i++)
        {
            for(int j =0;j<m;j++){
                cout<<" "<<(int)Mask.at<unsigned char>(i,j)<<",";
            }
            cout<<endl;
        }*/
        bitwise_not(Mask,Mask);

        setMask(Mask,n);
        //ROS_DEBUG("set mask costs %fms", t_m.toc());
        //cout<<"Mask.type is "<<Mask.type()<<endl;
        //cv::imshow("Mask is ",mask);
        //cv::waitKey(0);
        //cout<<"Mask.type is "<<Mask.type()<<endl;
        //ROS_DEBUG("detect feature begins");
        TicToc t_t;
        //背景和实例提取的最大特征点数量不同；
        //如果ｎ＝＝０,说明是背景；
        int n_max_cnt;
        if(n == 0){
            n_max_cnt = MAX_CNT - static_cast<int>(forw_pts[n].p.size());
        }else{
            //否则说明是Ｍａｓｋ中物体；
            n_max_cnt = 50 - static_cast<int>(forw_pts[n].p.size());
        }

        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                //cout<<"CV_8UC1 is "<<CV_8UC1<<endl;
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            //goodFeaturesToTrack意味着追寻好的可跟踪点；第一个为图像，第二个检测的角点；第三：角点数目最大值；
            //第四：角点的品质因子；MinDistance 对于初选出来的角点，如果周围minDistance范围内存在其他更强角点，则将此角点删除；
            //mask：指定感兴趣区域，不需要在整副图像中寻找；
            cv::goodFeaturesToTrack(forw_img, n_pts,  n_max_cnt, 0.01, MIN_DIST, mask);
            //mask是需要的,但对于后续帧中提取的goodFeatures需要检验是否在上一帧中的背景区域需要进行判断；
            //上述逻辑决定了，对于当前帧，我们生成新的特征点的时候，要对其在上一分割帧中的位置进行判定
            //而新的分割mask到来时,我们则对其在last-mask与cur_mask两个分割帧中进行位置判定。（不仅在good_feature，也对于上一帧中的跟踪数据）
            //这是设计结果，不是观察结果。如果发现跟踪结果仅跟踪一个mask即有较好的结果,则不需要。

        }
        else
            n_pts.clear();
        //ROS_DEBUG("detect feature costs: %fms", t_t.toc());


        //ROS_DEBUG("add feature begins");
        TicToc t_a;

        //因为点各有不同，因此需要添加m,n信息；
        addPoints(m,n);
        //cv::cvtColor(img,img, CV_GRAY2RGB);
        //cout<<"获得了 "<<n_pts.size()<<"个特征"<<endl;
        /*
        for(int j=0;j<n_pts.size();j++)
        {
            cv::circle(img, forw_pts[j], 3, cv::Scalar(255, 0, 255),2);
        }*/
        //inMask(Mask,n_pts);
        //cout<<"经过Mask检验 n_pts的size为"<<n_pts.size()<<endl;
        //cv::namedWindow("IMAGE", CV_WINDOW_AUTOSIZE);
        //cv::imshow("IMAGE",img);
        //cv::waitKey(0);
        //ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }

    prev_pts[n] = cur_pts[n];
    prev_un_pts[n] = cur_un_pts[n];

    //状态转移；

    cur_pts[n] = forw_pts[n];
    undistortedPoints(n);
    //时间是一个大问题；只有时间戳更新当前时间，此时时间戳更新不等于该函数已经进入下一次图像处理；
    if(cur_time!=prev_time){
        prev_img = cur_img;
        cur_img = forw_img;
        prev_time = cur_time;
    }

    cout<<"Opticalflow cost "<<ImageProcess.toc()<<endl;
}

void FeatureTracker::rejectWithF(int n)
{
    //如果是进入的第一幅图像，则直接跳过该函数；
    if (forw_pts[n].p.size() >= 8)
    {
        //第二幅有效图像进入；
        //ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts[n].p.size()), un_forw_pts(forw_pts[n].p.size());

        for (unsigned int i = 0; i < cur_pts[n].p.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            //liftProjective 函数将坐标由图像平面转移到归一化坐标；中间牵扯到中间点的平移；z轴上的深度缩放；
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[n].p[i].x, cur_pts[n].p[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[n].p[i].x, forw_pts[n].p[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        //计算基本矩阵；
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);

        //此处目的是利用status对光流匹配进行筛选剔除；
        //利用findFundamentalMat函数 以及 RANSAC算法保留那些符合F模型的特征；

        int size_a = cur_pts[n].p.size();

        reduceVector(prev_pts[n].p, status);
        reduceVector(cur_pts[n].p, status);
        reduceVector(forw_pts[n].p, status);

        reduceVector(cur_un_pts[n].p, status);

        reduceVector(ids[n], status);
        reduceVector(track_cnt[n], status);
        //ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        //ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}


bool FeatureTracker::updateID(unsigned int i,int n)
{
    if (i < ids[n].size())
    {
        if (ids[n][i] == -1)
            ids[n][i] = n_id++;
        return true;
    }
    else
        return false;
}

//读内参；
void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    cout << "reading paramerter of camera " << calib_file << endl;
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

//展现未产生畸变的点；
void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

//此处计算了光流在x，y方向上的速度； 经过处理了；
void FeatureTracker::undistortedPoints(int n){
    cur_un_pts[n].p.clear();
    cur_un_pts_map[n].clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts[n].p.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[n].p[i].x, cur_pts[n].p[i].y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);
        cur_un_pts[n].p.emplace_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        //cur_un_pts_map中输入的是归一化坐标；
        cur_un_pts_map[n].insert(make_pair(ids[n][i],cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }

    // caculate points velocity
    if (!prev_un_pts_map[n].empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity[n].clear();
        for (unsigned int i = 0; i < cur_un_pts[n].p.size(); i++)
        {
            if (ids[n][i] != -1) {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map[n].find(ids[n][i]);
                if (it != prev_un_pts_map[n].end()) {
                    double v_x = (cur_un_pts[n].p[i].x - it->second.x) / dt;

                    double v_y = (cur_un_pts[n].p[i].y - it->second.y) / dt;

                    pts_velocity[n].push_back(cv::Point2f(v_x, v_y));

                } else {
                    pts_velocity[n].push_back(cv::Point2f(0, 0));
                }
            }
            else
                {
                    pts_velocity[n].push_back(cv::Point2f(0, 0));
                }
        }
    }
    else//如果计算的速度是空的；
    {
        for (unsigned int i = 0; i < cur_pts[n].p.size(); i++)
        {
            pts_velocity[n].push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map[n] = cur_un_pts_map[n];
}

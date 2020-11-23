#include <thread>
#include "feature_tracker.h"
#include "estimator.h"
#include <opencv2/core/eigen.hpp>
#include "toolsForModel.h"

//int FeatureTracker::Object_id = 0;
/*
bool FeatureTracker::updateID(unsigned int i)
{
    if (i < Instance_Objects.size())
    {
        //ids[i] creat new ideas;
        if (Instance_Objects[i] == -1)
            Instance_Objects[i] = Object_id++;
        return true;
    }
    else
        return false;
}*/

double NumberInMask(cv::Mat Mask,vector<cv::Point2f>points){
    int count = 0;
    for(int i=0;i<points.size();i++){
        //if Object.point in the Mask;
        if(Mask.at<unsigned char>(points[i].y,points[i].x)==255){
          count += 1;
        }
    }
    return count/points.size();
    //cout<<"inMask endl"<<endl;
}
bool inBorder(cv::Point2f point,cv::Mat mask)
{
   if(mask.at<unsigned char>(point.y,point.x) == 255)
   {
       return true;
   }else{
       return false;
   }
}
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}
void reduceVector(vector<int> &v, vector<uchar>status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}


void reduceObject(vector<Instance_Object>&Instance_Objects,vector<int>&v){
    int j = 0 ;
    if(Instance_Objects.size() == 1){
        Instance_Objects[j] = Instance_Objects.back();
        v[j] = v.back();
        j++;
        Instance_Objects.resize(j);
        v.resize(j);
    }else{
        for(int i = 0 ;i < int(Instance_Objects.size()-1);i++){
            if(Instance_Objects[i].Tracking){
                v[j]=v[i];
                Instance_Objects[j]=Instance_Objects[i];
                j++;
            }
        }
        Instance_Objects[j] = Instance_Objects.back();
        v[j]=v.back();
        j++;
        Instance_Objects.resize(j);
        v.resize(j);
    }
    cout<<"j is "<<j<<endl;
    cout<<"Instance_Objects.size() is "<<Instance_Objects.size()<<endl;
}

void FeatureTracker::setK(string &calib_file)
{
    cv::FileStorage fs(calib_file.c_str(), cv::FileStorage::READ);
    cv::FileNode n = fs["projection_parameters"];

    float fx = static_cast<double>(n["fx"]);
    float fy = static_cast<double>(n["fy"]);
    float cx = static_cast<double>(n["cx"]);
    float cy = static_cast<double>(n["cy"]);



    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
};

FeatureTracker::FeatureTracker() {};
void setMask(Instance_Object& Object)
{
    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < Object.forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(Object.track_cnt[i], make_pair(Object.forw_pts[i], Object.ids[i])));
    //cout<<"cnt_pts_id.size() is "<<cnt_pts_id.size()<<endl;
    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    Object.forw_pts.clear();
    Object.ids.clear();
    Object.track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (Object.mask.at<uchar>(it.second.first) == 255)
        {
            Object.forw_pts.push_back(it.second.first);
            Object.ids.push_back(it.second.second);
            Object.track_cnt.push_back(it.first);
            cv::circle(Object.mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}
void addPoints(Instance_Object& Object)
{
    for (auto &p : Object.n_pts)
    {
        Object.forw_pts.push_back(p);
        Object.ids.push_back(-1);
        Object.track_cnt.push_back(1);
    }
}
vector<Instance_Object>FeatureTracker::addObjects(vector<Instance_Object>Instance_Objects){
    vector<Instance_Object>addDyna_Object;
    for(auto &Object:Instance_Objects){
        if(Object.NewObject && Object.Tracking && Object.isDyna)
        {
            //Object.NewOBject has noe been set to false yet;
            ids.push_back(-1);
            Object.NewObject = false;
            addDyna_Object.push_back(Object);
        }
    }
    return addDyna_Object;
};
vector<Instance_Object>FeatureTracker::addObjects(Instance_Object Bg){
    vector<Instance_Object>addDyna_Object;
    if(Bg.Tracking){
        ids.push_back(-1);
        Bg.NewObject = false;
        addDyna_Object.push_back(Bg);
    }else{
        cerr<<"no enough  features on BG"<<endl;
    }
    return addDyna_Object;
};

void FeatureGet(pair<const cv::Mat &,double>Img,Instance_Object& Object,camodocal::CameraPtr m_camera)
{
    cv::Mat _img,img;
    Object.cur_time = Img.second;
    cvtColor(Img.first,_img,CV_BGR2GRAY);


    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(_img, img);
    }
    else
        img = _img;

    if (Object.forw_img.empty())
    {
        Object.prev_img = Object.cur_img = Object.forw_img = img;
        cout<<"Object : 第一副图像进来了！！"<<endl;
    }
    else //如果forw_img不空，说明已经读入有效的图像了；如果是第一次读图；
    {
        Object.forw_img = img;
    }
    Object.forw_pts.clear();

    //if there is some
    if (Object.cur_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        cv::calcOpticalFlowPyrLK(Object.cur_img, Object.forw_img, Object.cur_pts, Object.forw_pts, status, err, cv::Size(21, 21), 3);

        for (int i = 0; i < int(Object.forw_pts.size()); i++)
            if (status[i] && !inBorder(Object.forw_pts[i],Object.mask))
                status[i] = 0;
        reduceVector(Object.prev_pts, status);
        reduceVector(Object.cur_pts, status);
        reduceVector(Object.forw_pts, status);
        reduceVector(Object.ids, status);
        reduceVector(Object.cur_un_pts, status);
        reduceVector(Object.track_cnt, status);
    }

    for (auto &n : Object.track_cnt)
        n++;

    Object.mask.convertTo(Object.mask, CV_8UC1, 255.0);


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
    //bitwise_not(Mask,Mask);

    rejectWithF(Object,m_camera);
    if(Object.id_for_class == -1){
        cout<<"bgground already has "<<Object.forw_pts.size()<<endl;
    }

    if(Object.NeedObstract){
        if(Object.id_for_class == -1){
            MAX_CNT = 200;
            MIN_DIST = 20;
            setMask(Object);
        }
        else{
            MAX_CNT = 80;
            MIN_DIST = 0;
        }
        //cout<<"here we can see  Max_CNT is"<<MAX_CNT<<endl;

        int n_max_cnt = MAX_CNT - static_cast<int>(Object.forw_pts.size());
        if (n_max_cnt > 0)
        {
            if( Object.mask.empty())
                cout << "mask is empty " << endl;
            if ( Object.mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if ( Object.mask.size() !=  Object.forw_img.size())
                cout << "wrong size " << endl;
            cv::goodFeaturesToTrack(Object.forw_img,  Object.n_pts, MAX_CNT - Object.forw_pts.size(), 0.01,MIN_DIST, Object.mask);
            cout<<"current img we take new points is "<< Object.n_pts.size()<<endl;
        }
        else
            Object.n_pts.clear();

        //cout<<"Object.n_pts.size() is "<<Object.n_pts.size()<<endl;

        addPoints(Object);
        //for after Seg Mask, we initial the NeedObstract with false;
        Object.NeedObstract = false;
    }


    Object.prev_img = Object.cur_img;
    Object.prev_pts = Object.cur_pts;
    Object.prev_un_pts = Object.cur_un_pts;
    Object.cur_img = Object.forw_img;
    Object.cur_pts = Object.forw_pts;
    undistortedPoints(Object,m_camera);
    Object.prev_time = Object.cur_time;

    if(Object.cur_pts.size() < 10)
    {
        Object.Tracking = false;
        cout<<"here is a tracking false egg";
    }
    //cv::cvtColor(img,img, CV_GRAY2RGB);
    //cout<<"获得了 "<<n_pts.size()<<"个特征"<<endl;

    /*
    for(int j=0;j<Object.cur_pts.size();j++)
    {
        cv::circle(img, Object.cur_pts[j], 3, cv::Scalar(255, 0, 255),2);}
    cv::namedWindow("IMAGE", CV_WINDOW_AUTOSIZE);
    cv::imshow("IMAGE",img);
    cv::waitKey(0);*/
};
void TrackObject(pair<pre_integration_ImuPtr, ORI_MSGPtr>measurement,Instance_Object& Object,camodocal::CameraPtr m_camera,cv::Mat& F)
{
    //Object.id_for_class = Mask_id; keep class id;
    cv::Mat _img,img;
    Object.cur_time = measurement.second->header;
    cvtColor(measurement.second->img,_img,CV_BGR2GRAY);
    //cout<<"_Img.type(): "<<_img.type()<<endl;

    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(_img, img);
    }
    else
        img = _img;

    if (Object.forw_img.empty())
    {
        Object.prev_img = Object.cur_img = Object.forw_img = img;
        cerr<<"Object no created"<<endl;
    }
    else //如果forw_img不空，说明已经读入有效的图像了；如果是第一次读图；
    {
        Object.forw_img = img;
    }
    Object.forw_pts.clear();

    //if there is some
    //cout<<"we can see how many corners in the object"<<Object.cur_pts.size()<<endl;
    if (Object.cur_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        cv::calcOpticalFlowPyrLK(Object.cur_img, Object.forw_img, Object.cur_pts, Object.forw_pts, status, err, cv::Size(21, 21), 3);

        /*
        for (int i = 0; i < int(Object.forw_pts.size()); i++)
            if (status[i] && !inBorder(Object.forw_pts[i],Object.mask))
                status[i] = 0;*/

        if(Object.id_for_class==-1){
            int count = 0;
            for(int i =0;i<43;i++){
                if(status[i]){
                    count++;
                }
            }
            cout<<"we can see number after match is "<<count<<endl;
        }
        reduceVector(Object.prev_pts, status);
        reduceVector(Object.cur_pts, status);
        reduceVector(Object.forw_pts, status);
        reduceVector(Object.ids, status);
        reduceVector(Object.cur_un_pts, status);
        reduceVector(Object.track_cnt, status);
    }

    for (auto &n : Object.track_cnt)
        n++;


    if(Object.forw_pts.size() < 10)
    {
        Object.Tracking = false;
        //cout<<"one object tracking fails"<<endl;
        return;
    }
    cv::Mat F_object;
    if(Object.id_for_class == -1){
        F = rejectWithF(Object,m_camera);
    }else{
        F_object = rejectWithF(Object,m_camera);
    }
    if(Object.forw_pts.size() < 10)
    {
        Object.Tracking = false;
        cout<<"one object tracking fails"<<endl;
        return;
    }
    if(Object.id_for_class == -1){
        cout<<"bgground already has "<<Object.forw_pts.size()<<endl;
    }

    Object.prev_img = Object.cur_img;
    Object.prev_pts = Object.cur_pts;
    Object.prev_un_pts = Object.cur_un_pts;
    Object.cur_img = Object.forw_img;
    Object.cur_pts = Object.forw_pts;
    undistortedPoints(Object,m_camera);
    Object.prev_time = Object.cur_time;

    //cout<<"here we can see how many inliers "<<Object.cur_pts.size()<<endl;

    ///judge whether this is dynamic object;
    //only in the second frame, we do dyna_judge, if it is dynamic, we will focous the state of the Object;
    //if it's first appear;

    if(Object.Dyna_judge){
        //check if this object is dynamic according to the data named: Object.cur_pts and Object.prev_pts;
        Matrix3d F_eigen;
        //cout<<"here works 1"<<endl;
        cv2eigen(F,F_eigen);
        //cout<<"here works 2"<<endl;

        const float th = 1;
        int count = 0;
        //cout<<"here works "<<endl;
        for (int i = 0; i < Object.cur_pts.size(); i++) {
            //let's do check;
            const float f11 = F_eigen(0, 0);
            const float f12 = F_eigen(0, 1);
            const float f13 = F_eigen(0, 2);
            const float f21 = F_eigen(1, 0);
            const float f22 = F_eigen(1, 1);
            const float f23 = F_eigen(1, 2);
            const float f31 = F_eigen(2, 0);
            const float f32 = F_eigen(2, 1);
            const float f33 = F_eigen(2, 2);

            const float u1 = Object.prev_pts[i].x;
            const float v1 = Object.prev_pts[i].y;
            const float u2 = Object.cur_pts[i].x;
            const float v2 = Object.cur_pts[i].y;

            const float a2 = f11 * u1 + f12 * v1 + f13;
            const float b2 = f21 * u1 + f22 * v1 + f23;
            const float c2 = f31 * u1 + f32 * v1 + f33;

            const float num2 = a2 * u2 + b2 * v2 + c2;

            const float squareDist1 = num2 * num2 / (a2 * a2 + b2 * b2);

            const float chiSquare1 = squareDist1;


            const float a1 = f11 * u2 + f21 * v2 + f31;
            const float b1 = f12 * u2 + f22 * v2 + f32;
            const float c1 = f13 * u2 + f23 * v2 + f33;

            const float num1 = a1 * u1 + b1 * v1 + c1;

            const float squareDist2 = num1 * num1 / (a1 * a1 + b1 * b1);

            const float chiSquare2 = squareDist2;
            if (chiSquare2 < th && chiSquare1 < th) {
                count++;
            }
        }

        cout << "  static rate is!!   "<< (float(count)/float(Object.cur_pts.size())) << endl;
        if( (float(count)/float(Object.cur_pts.size())) < 0.3){
            //if only little points in object match camera model, we take it dynamic;
            Object.isDyna = true;
            cout<<"we can see the class of object"<<Object.id_for_class<<endl;
            cout << "   this is new Dyna_Object!!   "<<endl;
            /*.
            if( Object.Tracking ){
                cout<<"Tracking is correct"<<endl;
            }
            if( Object.isDyna ){
                cout<<"isDyna is correct"<<endl;
            }
            if( Object.NewObject ){
                cout<<"NewObject is correct"<<endl;
            }*/
        }else{
            cout<<"one seg_mask is static Object"<<endl;
        }
        Object.Dyna_judge = false;
    }
}
void FeatureTracker::readMaskImage(pair<const cv::Mat &,double>Img,vector<cv::Mat>Mask,vector<int> Mask_id,vector<vector<double>>pred_boxex,cv::Mat& BGMask)
{
    vector<cv::Mat>addMask;
    vector<int>add_id;
    vector<bool>IsOldObject(Mask.size(),false);

    if(Instance_Objects.size()==0){
        //BGMask is BackgroundMask;

        if(Mask.size()>0)
        {
            Dyna_judge = true;
        }
        //vector<Instance_Object>thread_Instance_Objects(Mask.size()+1);

        vector<Instance_Object>thread_Instance_Objects(Mask.size());
        for(int i = 0;i<Mask.size();i++){
            //cout<<"BGMask works"<<"1 round"<<endl;
            BGMask +=Mask[i];
            //cout<<"here works"<<endl;
            thread_Instance_Objects[i].mask=Mask[i];
            //cout<<"here works1"<<endl;
            thread_Instance_Objects[i].id_for_class = Mask_id[i];
            //cout<<"here works2"<<endl;
            thread_Instance_Objects[i].NeedObstract = true;
            //cout<<"here works3"<<endl;
            thread_Instance_Objects[i].Dyna_judge = true;
            thread_Instance_Objects[i].NewObject = true;
            //cout<<"here works4"<<endl;
        }

        //BGground no need to do Dyna_judge,and NewObject Judge;
        BGMask.convertTo(BGMask, CV_8UC1, 255.0);
        bitwise_not(BGMask,BGMask);
        //cout<<"BGMask works"<<endl;
        Instance_Object Cur_New_Object;
        Cur_New_Object.mask = BGMask;
        Cur_New_Object.id_for_class= -1;
        Cur_New_Object.NeedObstract = true;
        Cur_New_Object.NewObject = true;

        //FeatureGet(Img,thread_Instance_Objects[0],m_camera);
        std::thread thread_BG = std::thread(FeatureGet,Img,ref(Cur_New_Object),m_camera);
        std::thread threads[thread_Instance_Objects.size()];
        for(int i=0;i< thread_Instance_Objects.size();i++){
            //threads is for creating multi_objects;
            //Input:Img;addMask[i],addMask_id[i];
            threads[i] = std::thread(FeatureGet,Img,ref(thread_Instance_Objects[i]),m_camera);
        }
        for (auto& t: threads) {
            t.join();
        }
        thread_BG.join();

        Instance_Objects.push_back(Cur_New_Object);
        New_Instance_Objects = thread_Instance_Objects;

        cout<<"first Seg process end "<<endl;

    }else{
        //Three condition;
        //1.Old object no Mask(Mask means new optical_flow);
        //2.Old Object with new Mask;
        //3. New Object;
        cout<<" here works Mask.size() is "<<Mask.size()<<endl;
        cout<<"Instance_Object.size() is "<<Instance_Objects.size()<<endl;
        for(int i=0;i<Mask.size();i++){
            //through all Objects;
            BGMask +=Mask[i];
            //size - 1 for the last one is the BGmask;
            for(int j=0;j<Instance_Objects.size()-1;j++){
                //if 0.7 * Points of Object is in Mask[i], we think they are the same Objects;
                if (Instance_Objects[j].NeedObstract == true){
                    //if this object has new mask, we won't test new mask for this object;
                    //cout<<"here works"<<endl;
                    continue;
                }
                //cout<<"here works"<<endl;
                if(NumberInMask(Mask[i],Instance_Objects[j].cur_pts)>0.5){
                    //Mask is the old Object;
                    //addMask.push_back(Mask[i]);
                    //update mask of old Instance_Object;
                    Instance_Objects[j].mask = Mask[i];
                    Instance_Objects[j].id_for_class = Mask_id[i];
                    //delete this Mask from Mask[i];
                    IsOldObject[i]=true;
                    Instance_Objects[j].NeedObstract = true;
                    break;//if this mask has detectes Old_object,we will not test other Objects for this mask;
                }
                //cout<<"detect works"<<endl;
            }
        }
        //cout<<"here works 1"<<endl;
        //get New Mask or New Object;
         if(IsOldObject.size()>0){
             Dyna_judge = true;
         }
         for(int i =0;i<IsOldObject.size();i++) {
             Instance_Object Cur_New_Object;
             if (IsOldObject[i] == false) {
                 Cur_New_Object.mask = Mask[i];
                 Cur_New_Object.id_for_class = Mask_id[i];
                 Cur_New_Object.NeedObstract = true;
                 Cur_New_Object.Dyna_judge = true;
                 Cur_New_Object.NewObject = true;
             }
             New_Instance_Objects.push_back(Cur_New_Object);
         }
        //cout<<"here works 2"<<endl;
        BGMask.convertTo(BGMask, CV_8UC1, 255.0);
        bitwise_not(BGMask,BGMask);
        Instance_Objects.back().mask = BGMask;
        Instance_Objects.back().id_for_class = -1;
        Instance_Objects.back().NeedObstract = true;
        //std::thread threads[Instance_Objects.size()];
        std::thread threads[Instance_Objects.size()];
        std::thread threads_new_masks[New_Instance_Objects.size()];
        for(int i=0;i<Instance_Objects.size();i++){
            //threads is for creating multi_objects;
            //Input:Img;addMask[i],addMask_id[i];
            threads[i] = std::thread(FeatureGet,Img,ref(Instance_Objects[i]),m_camera);
        }


        for(int i = 0 ;i<New_Instance_Objects.size();i++){
            threads_new_masks[i] = std::thread(FeatureGet,Img,ref(New_Instance_Objects[i]),m_camera);
        }
        for (auto& th: threads_new_masks){
            th.join();
        }
        for (auto& t: threads) {
            t.join();
        }
        //cout<<"here works 3"<<endl;
        //belong to tracking, so we need to judge whether the Object track enough features;
        reduceObject(Instance_Objects,ids);
        cout<<"the 12 frame appears "<<New_Instance_Objects.size()<<" candidate Objects "<<endl;
    }


};
void FeatureTracker::trackImg(pair<pre_integration_ImuPtr, ORI_MSGPtr>measurement){
    //TrackObject;
    //Instane_Objects has locked sizes;
    //cout<<"we can see there are already "<<Instance_Objects.size()<<" body "<<endl;
    //cout<<"we can see there are " <<New_Instance_Objects.size()<<" candidate mask"<<endl;
    cv::Mat F = cv::Mat::zeros(3,3,CV_32F);
    if(Dyna_judge == true){
        //cout<<"Dyna_judge is true"<<endl;
        TrackObject(measurement,Instance_Objects.back(),m_camera,F);
        Dyna_judge = false;
        //cout<<"F.size is "<<F.size()<<endl;
        //cout<<"Instance_Objects is"<< Instance_Objects.size()<<endl;
        //Instance_Objects.insert(Instance_Objects.end()-1,New_Instance_Objects.begin(),New_Instance_Objects.end());
        //cout<<"New_Instancee_Objects.size() is "<<New_Instance_Objects.size()<<endl;
        if(Instance_Objects.size() == 1){
            //cout<< " no old dyna_Object " <<endl;
            std::thread threads_new_masks[New_Instance_Objects.size()];
            for(int i = 0 ;i<New_Instance_Objects.size();i++){
                threads_new_masks[i] = std::thread(TrackObject,measurement,ref(New_Instance_Objects[i]),m_camera,ref(F));
            }
            for(auto & t:threads_new_masks){
                t.join();
            }
        }else{
            std::thread threads_new_masks[New_Instance_Objects.size()];
            std::thread Trc_threads[Instance_Objects.size()-1];
            for(int i = 0 ;i<New_Instance_Objects.size();i++){
                threads_new_masks[i] = std::thread(TrackObject,measurement,ref(New_Instance_Objects[i]),m_camera,ref(F));
            }

            for(int i=0;i<Instance_Objects.size()-1;i++){
                Trc_threads[i] = std::thread(TrackObject,measurement,ref(Instance_Objects[i]),m_camera,ref(F));
            }
            for (auto& t: Trc_threads) {
                t.join();
            }
            for(auto & t:threads_new_masks){
                t.join();
            }
        }

        if(ids.size()==0){
            addObjects(Instance_Objects.back());
            reduceObject(Instance_Objects,ids); //nonsense this sentence;
            vector<Instance_Object>Dyna_Object = addObjects(New_Instance_Objects);
            New_Instance_Objects.clear();
            Instance_Objects.insert(Instance_Objects.end()-1,Dyna_Object.begin(),Dyna_Object.end());
            //correct ids to the back of ids, even here no essential;
            int Bg_id = ids.front();
            vector<int>::iterator k = ids.begin();
            ids.erase(k);
            ids.push_back(Bg_id);
        }else{
            reduceObject(Instance_Objects,ids);

            int Bg_id = ids.back();
            ids.pop_back();

            vector<Instance_Object>Dyna_Object = addObjects(New_Instance_Objects);
            cout<<" Dyna_Object has "<<Dyna_Object.size() << " Objects "<<endl;
            New_Instance_Objects.clear();
            Instance_Objects.insert(Instance_Objects.end()-1,Dyna_Object.begin(),Dyna_Object.end());
            ids.push_back(Bg_id);
        }

    }else{
        cout<<"just track"<<endl;
        cout<<"there are " <<Instance_Objects.size()-1<<"left Dyna_Objects while tracking correctly"<<endl;
        std::thread Trc_threads[Instance_Objects.size()];

        for(int i=0;i<Instance_Objects.size();i++){

            Trc_threads[i] = std::thread(TrackObject,measurement,ref(Instance_Objects[i]),m_camera,ref(F));
        }

        for (auto& t: Trc_threads) {
            t.join();
        }
        reduceObject(Instance_Objects,ids);
    }

    //according to Tracking, we get the dynamic Object while have enough features in cur_img;

    /*
    for(int i = 0;i<Instance_Objects.size();i++){
        cout<<" here we can see there are features in different Instance_Objects "<<endl;
        cout<<"the number is "<< Instance_Objects[i].cur_un_pts.size()<<endl;
    }*/
};

cv::Mat rejectWithF(Instance_Object& Object,camodocal::CameraPtr m_camera)
{

    if ( Object.forw_pts.size() >= 8)
    {
        //ROS_DEBUG("FM ransac begins");.
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(Object.cur_pts.size()), un_forw_pts(Object.forw_pts.size());

        for (unsigned int i = 0; i <  Object.cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera->liftProjective(Eigen::Vector2d( Object.cur_pts[i].x,  Object.cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d( Object.forw_pts[i].x,  Object.forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar>status;
        //cout<<"before we can see how many features::cur_pts.size() is "<<Object.cur_pts.size()<<" class is "<<Object.id_for_class<<endl;
        cv::Mat F_mat = cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);

        if(Object.id_for_class==-1){
            int count = 0;
            for(int i =0;i<87;i++){
                if(status[i]){
                    count++;
                }
            }
            cout<<"we can see number is"<<count<<endl;

        }
        //cv::Mat F_mat = Cal_F(Object.cur_pts,Object.forw_pts,status);
        //cout<<"F_mat is "<<F_mat<<endl;
        int size_a =  Object.cur_pts.size();
        reduceVector( Object.prev_pts, status);
        reduceVector( Object.cur_pts, status);
        reduceVector( Object.forw_pts, status);
        reduceVector( Object.cur_un_pts, status);
        //cout<<"here we can see how many features: cur_pts.size() is "<< Object.cur_pts.size() <<" class is "<<Object.id_for_class<<endl;
        reduceVector( Object.ids, status);

        reduceVector( Object.track_cnt, status);
        return F_mat;
    }else{
        cv::Mat F_mat = cv::Mat::zeros(3,3,CV_32F);
        return F_mat;
    }

}

bool FeatureTracker::update_Object_ID(unsigned int i)
{
    if (i < ids.size())
    {
        //ids[i] creat new ideas;
        if (ids[i] == -1){
            ids[i] = Object_id++;
            cout<<"  new Object has been recorded  "<<endl;
        }
        return true;
    }
    else
        return false;
}

//update feature id in Instance_Objects;
void FeatureTracker::updateID(vector<Instance_Object>& Instance_Objects)
{
    for(int j =0;j<Instance_Objects.size();j++){
        //cout<<"we can see here is "<< j <<" ge object"<<endl;

        for(int i=0;i < Instance_Objects[j].ids.size();i++)
        {
            if ( Instance_Objects[j].ids[i] == -1 )
            {
                Instance_Objects[j].ids[i] = Instance_Objects[j].n_id++;
                //cout<<"ids is " <<Instance_Objects[j].ids[i]<<endl;
            }
        }

        if(Instance_Objects[j].id_for_class==-1){
            int count = 0;
            for(int i =0;i<Instance_Objects[j].ids.size();i++){
                if(Instance_Objects[j].ids[i]<108){
                    count++;
                }
            }
            cout<<"the n_id in Instance_Object is"<<Instance_Objects[j].n_id<<endl;
            cout<<"old features after F is"<<count<<endl;
        }
    }
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    cout << "reading paramerter of camera " << calib_file << endl;
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
    cout<<"camera -> K"<<m_camera->K;
}

void undistortedPoints(Instance_Object& Object,camodocal::CameraPtr m_camera)
{
    Object.cur_un_pts.clear();
    Object.cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < Object.cur_pts.size(); i++)
    {
        Eigen::Vector2d a(Object.cur_pts[i].x, Object.cur_pts[i].y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);
        //cout<<"b.x() is "<<b.x()<<endl;
        //cout<<"b.y() is "<<b.y()<<endl;
        Object.cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        Object.cur_un_pts_map.insert(make_pair(Object.ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity
    if (!Object.prev_un_pts_map.empty())
    {
        double dt = Object.cur_time -Object.prev_time;
        Object.pts_velocity.clear();
        for (unsigned int i = 0; i < Object.cur_un_pts.size(); i++)
        {
            if (Object.ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = Object.prev_un_pts_map.find(Object.ids[i]);
                if (it != Object.prev_un_pts_map.end())
                {
                    double v_x = (Object.cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (Object.cur_un_pts[i].y - it->second.y) / dt;
                    Object.pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    Object.pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
            {
                Object.pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < Object.cur_pts.size(); i++)
        {
            Object.pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    Object.prev_un_pts_map = Object.cur_un_pts_map;
}
/*
       Object.Dyna_judge = false;
       //get pre_integration result;
       Eigen::Vector3d delta_p;
       Eigen::Quaterniond delta_q;
       delta_p  = measurement.first->delta_p;
       delta_q  = measurement.first->delta_q;

       Matrix3d R,K_eigen,K_eigen_inverse;
       R = delta_q.toRotationMatrix();
       //get the systemetric matrix;
       delta_p = delta_p/delta_p.norm();
       Matrix3d Tsymmetric  = Eigen::Matrix3d::Zero();
       delta_p = delta_p/delta_p.norm();
       Tsymmetric << 0,         -delta_p(2), delta_p(1),
               delta_p(2),    0,    -delta_p(0),
               delta_p(1),    delta_p(0), 0;
       cout<<"Tsysmmetric is "<< Tsymmetric <<endl;

       //contruct K
       //E = Tsymmetric*R;F=K(T)*E*K;
       Eigen::Matrix3d F;
       cv2eigen(K,K_eigen);

       cout<<"K is "<<  K_eigen <<endl;
       K_eigen_inverse = K_eigen.inverse();
       F = K_eigen_inverse.transpose()*Tsymmetric*R*K_eigen_inverse;
       F = F/F(2,2);

       cout<<"F is "<<F<<endl;*/
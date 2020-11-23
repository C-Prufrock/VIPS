
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <thread>
#include <iomanip>

#include <cv.h>
#include <opencv2/opencv.hpp>
#include <highgui.h>
#include <eigen3/Eigen/Dense>
#include "System.h"

#include <InstanceSeg.h>

using namespace std;
using namespace cv;
using namespace Eigen;

const int nDelayTimes = 2;
//string sData_path = "/home//lxy/VIDataset/MH_05_difficult/mav0/";
string sConfig_path = "/home/lxy/SLAM/VIO/VINS-Course-master/config/";

std::shared_ptr<System>pSystem;

//下列代码用来获取imu数据；
void PubImuData()
{
    //从路径中读取imu相关数据；
	string sImu_data_file = sConfig_path+"0926_imu/timestamps.txt";
	cout << "1 PubImuData start sImu_data_filea: " << sImu_data_file << endl;
	FILE* file;
	file = std::fopen(sImu_data_file.c_str(),"r");

	if(file == NULL){
	    printf("cannot find file: %simage_00/timestamps.txt \n", sImu_data_file.c_str());
	    return ;
	}
	vector<double>imuTimeList;
	int year,month,day;
	int hour,minute;
	double second;
    while (fscanf(file, "%d-%d-%d %d:%d:%lf", &year, &month, &day, &hour, &minute, &second) != EOF)
    {
        //printf("%lf\n", second);
        //printf("%lf\n",hour * 60 * 60 + minute * 60 + second);
        imuTimeList.push_back(hour * 60 * 60 + minute * 60 + second);
    }
    std::fclose(file);
    pSystem->t_r;
    string ImuFilePath;
    //cout<<"here works"<<endl;
    for(int i =0 ;i<imuTimeList.size();i++){
        //读取KITTI文件名称；
        stringstream ss;
        ss << setfill('0') << setw(10) << i;
        //cout<<"ss : "<< ss.str() <<endl;
        ImuFilePath =  sConfig_path +"0926_imu/data/"+ss.str()+".txt";
        //cout<<"ImuFilePath : "<<ImuFilePath<<endl;
        FILE* ImuFile = std::fopen(ImuFilePath.c_str() , "r");
        if(ImuFile == NULL){
            printf("cannot find file: %s\n", ImuFilePath.c_str());
            return ;
        }
        //cout<<"ImuFile works"<<endl;
        double lat, lon, alt, roll, pitch, yaw;
        double vn, ve, vf, vl, vu;
        double ax, ay, az, af, al, au;
        double wx, wy, wz, wf, wl, wu;
        double pos_accuracy, vel_accuracy;
        double navstat, numsats;
        double velmode, orimode;
        Vector3d vAcc;
        Vector3d vGyr;
        Vector3d vVel;

        fscanf(ImuFile, "%lf %lf %lf %lf %lf %lf ", &lat, &lon, &alt, &roll, &pitch, &yaw);
        //cout<<" we can read some data"<<endl;
        //printf("lat:%lf lon:%lf alt:%lf roll:%lf pitch:%lf yaw:%lf \n",  lat, lon, alt, roll, pitch, yaw);

        fscanf(ImuFile, "%lf %lf %lf %lf %lf ", &vn, &ve, &vVel.x(), &vVel.y(), &vVel.z());
        //printf("vn:%lf ve:%lf vf:%lf vl:%lf vu:%lf \n",  vn, ve, vf, vl, vu);
        fscanf(ImuFile, "%lf %lf %lf %lf %lf %lf ", &vAcc.x(), &vAcc.y(), &vAcc.z(), &af, &al, &au);
        //printf("ax:%lf ay:%lf az:%lf af:%lf al:%lf au:%lf\n",  ax, ay, az, af, al, au);
        fscanf(ImuFile, "%lf %lf %lf %lf %lf %lf ", &vGyr.x(), &vGyr.y(), &vGyr.z(), &wf, &wl, &wu);
        //printf("wx:%lf wy:%lf wz:%lf wf:%lf wl:%lf wu:%lf\n",  wx, wy, wz, wf, wl, wu);
        fscanf(ImuFile, "%lf %lf %lf %lf %lf %lf ", &pos_accuracy, &vel_accuracy, &navstat, &numsats, &velmode, &orimode);
        //printf("pos_accuracy:%lf vel_accuracy:%lf navstat:%lf numsats:%lf velmode:%lf orimode:%lf\n",
        //	    pos_accuracy, vel_accuracy, navstat, numsats, velmode, orimode);
        //cout<<"vAcc.x():"<<vAcc.x()<<endl;
        //cout<<"vAcc.y():"<<vAcc.y()<<endl;
        //cout<<"vAcc.z():"<<vAcc.z()<<endl;
        //cout<<"vGyr.x():"<<vGyr.x()<<endl;
        //cout<<"vGyr.y():"<<vGyr.y()<<endl;
        //cout<<"vGyr.z():"<<vGyr.z()<<endl;

        //cout<<setiosflags(ios::fixed);  //保证setprecision()是设置小数点后的位数。
        //cout<<setprecision(6) << imuTimeList[i] << endl;    //输出3.14
        //cout<<"imuTimeList[i]"<<imuTimeList[i]/1e6<<endl;
        //cout<<"i is "<< i << endl;
        pSystem->PubImuData(imuTimeList[i], vGyr, vAcc,vVel);
        usleep(500*nDelayTimes);
        std::fclose(ImuFile);
    }
}

//下列数据用来获取Image数据；

void PubImageData()
{
	string sImage_data_file = sConfig_path + "image_02/timestamps.txt";
    cout << "1 PubImgData start sImg_data_filea: " << sImage_data_file<< endl;
    FILE* imgTimefile;
    imgTimefile = std::fopen(sImage_data_file.c_str(),"r");
    if(imgTimefile == NULL){
        printf("cannot find file: %simage_00/timestamps.txt \n", sImage_data_file.c_str());
        return ;
    }
    vector<double>imgTimeList;
    int year,month,day;
    int hour,minute;
    double second;
    while (fscanf(imgTimefile, "%d-%d-%d %d:%d:%lf", &year, &month, &day, &hour, &minute, &second) != EOF)
    {
        //printf("%lf\n", second);
        //printf("%lf\n",hour * 60 * 60 + minute * 60 + second);
        imgTimeList.push_back(hour * 60 * 60 + minute * 60 + second);
    }
    string ImgFilePath;
    pSystem->t_r;
    for(int i =55;i < imgTimeList.size();i++) {
        stringstream ss;
        ss << setfill('0') << setw(10) << i;
        //cout<<"ss : "<< ss.str() <<endl;
        ImgFilePath =  sConfig_path +"image_02/data/"+ss.str()+".png";
        //cout<<"ImgFilePath "<<ImgFilePath<<endl;
        Mat img = imread(ImgFilePath.c_str());
        if (img.empty())
        {
            cerr << "image is empty! path: " << ImgFilePath << endl;
            return;
        }
        //std::cout<<"Img size is"<<img.size()<<endl;
        //cv::imshow("SOURCE IMAGE", img);
        //cv::waitKey(3000);
        usleep(5000*nDelayTimes);
        pSystem->PubImageData(imgTimeList[i], img);
    }
    // cv::namedWindow("SOURCE IMAGE", CV_WINDOW_AUTOSIZE);
}

void DoSeg(){
    //初始化分割网络；
    pSystem->t_r;
    pSystem->DynSeg = new InstanceSeg();
    //cout<<"Tic_toc in Seg_th"<<pSystem->t_r.toc()<<endl;
    //根据System的函数进行分割选择,并返回分割结果到System中的变量中；
    pSystem->SegImg();
    //等待要处理的图像；

}

void Track(){
    pSystem->DynaTrack();
}
/*
void Track(){
    //利用
}*/
int main(int argc, char **argv)
{
	/*if(argc != 3)
	{
		cerr << "./run_euroc PATH_TO_FOLDER/MH-05/mav0 PATH_TO_CONFIG/config \n" 
			<< "For example: ./run_euroc /home/stevencui/dataset/EuRoC/MH-05/mav0/ ../config/"<< endl;
		return -1;
	}*/
	//string sData_path = "/home//lxy/VIDataset/MH_05_difficult/mav0/";;
	sConfig_path = "/home/lxy/SLAM/VIO/VINS-Course-master/config/";
    //InstanceSeg* DynSeg = new InstanceSeg();
	pSystem.reset(new System(sConfig_path));
	
	//std::thread thd_BackEnd(&System::ProcessBackEnd, pSystem);

	// sleep(5);
	std::thread thd_PubImuData(PubImuData);
	std::thread thd_PubImageData(PubImageData);
    std::thread thd_DoSeg(DoSeg);
    std::thread thd_Track(Track);

#ifdef __linux__	
     //std::thread thd_Draw(&System::Draw, pSystem);
	//支持Mac系统下运行；
#elif __APPLE__
	DrawIMGandGLinMainThrd();
#endif

	thd_PubImuData.join();
	thd_PubImageData.join();
    thd_DoSeg.join();
    thd_Track.join();

	//thd_BackEnd.join();

#ifdef __linux__	
	//thd_Draw.join();
#endif

	cout << "main end... see you ..." << endl;
	return 0;
}

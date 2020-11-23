//
// Created by lxy on 20-10-15.
//

#include "toolsForModel.h"
#include "utility/Random.h"



using namespace std;
using namespace Eigen;

cv::Mat Cal_F(vector<cv::Point2f>&points1, vector<cv::Point2f>&points2,vector<uchar>& status){

    const int N = points1.size();
    vector<vector<size_t>> mvSets; ///< 二维容器，外层容器的大小为迭代次数，内层容器大小为每次迭代算H或F矩阵需要的点
    int mMaxIterations = 2000; ///< 算Fundamental和Homography矩阵时RANSAC迭代次数

    // Indices for minimum set selection
    vector<size_t> vAllIndices;
    vAllIndices.reserve(N);
    vector<size_t> vAvailableIndices;

    for (int i = 0; i < N; i++) {
        vAllIndices.push_back(i);
    }

    mvSets = vector<vector<size_t> >(mMaxIterations, vector<size_t>(8, 0));

    DUtils::Random::SeedRandOnce(0);

    for (int it = 0; it < mMaxIterations; it++) {
        vAvailableIndices = vAllIndices;

        // Select a minimum set
        for (size_t j = 0; j < 8; j++) {
            int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size() - 1);
            int idx = vAvailableIndices[randi];

            mvSets[it][j] = idx;

            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }
    }
    vector<Vector2f>vPn1, vPn2;
    Matrix3f T1, T2;
    Normalize(points1, vPn1, T1);
    Normalize(points2, vPn2, T2);
    Matrix3f T2t = T2.transpose();

    double score = 0.0;
    vector<uchar>vbMatchesInliers = vector<uchar>(N, false);

    // Iteration variables
    vector<Vector2f> vPn1i(8);
    vector<Vector2f> vPn2i(8);
    Matrix3f F21i,F21;
    vector<uchar>vbCurrentInliers(N, false);
    float currentScore;

    // Perform all RANSAC iterations and save the solution with highest score
    for (int it = 0; it < mMaxIterations; it++) {
        // Select a minimum set
        for (int j = 0; j < 8; j++) {
            int idx = mvSets[it][j];

            vPn1i[j] = vPn1[idx];
            vPn2i[j] = vPn2[idx];
        }

        Matrix3f Fn = ComputeF21(vPn1i, vPn2i);

        F21i = T2t * Fn * T1;
        double mSigma = 1.0;
        currentScore = CheckFundamental(F21i,vbCurrentInliers,points1,points2,mSigma);

        if (currentScore > score) {
            F21 = F21i;
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
    cv::Mat F_mat;
    //check the score distribution;
    double mSigma = 1.0;
    CheckFundamental(F21,vbCurrentInliers,points1,points2,mSigma);

    show(F21,vbMatchesInliers,points1,points2,mSigma);

    cv::eigen2cv(F21,F_mat);
    status = vbMatchesInliers;
    return F_mat;
}
void Normalize(vector<cv::Point2f>&vKeys, vector<Vector2f> &vNormalizedPoints, Matrix3f &T) {
    float meanX = 0;
    float meanY = 0;
    const int N = vKeys.size();

    vNormalizedPoints.resize(N);

    for (int i = 0; i < N; i++) {
        meanX += vKeys[i].x;
        meanY += vKeys[i].y;
    }

    meanX = meanX / N;
    meanY = meanY / N;

    float meanDevX = 0;
    float meanDevY = 0;

    for (int i = 0; i < N; i++) {
        vNormalizedPoints[i][0] = vKeys[i].x - meanX;
        vNormalizedPoints[i][1] = vKeys[i].y - meanY;

        meanDevX += fabs(vNormalizedPoints[i][0]);
        meanDevY += fabs(vNormalizedPoints[i][1]);
    }

    meanDevX = meanDevX / N;
    meanDevY = meanDevY / N;

    float sX = 1.0 / meanDevX;
    float sY = 1.0 / meanDevY;

    for (int i = 0; i < N; i++) {
        vNormalizedPoints[i][0] = vNormalizedPoints[i][0] * sX;
        vNormalizedPoints[i][1] = vNormalizedPoints[i][1] * sY;
    }

    T = Matrix3f::Identity();
    T(0, 0) = sX;
    T(1, 1) = sY;
    T(0, 2) = -meanX * sX;
    T(1, 2) = -meanY * sY;
}
Matrix3f ComputeF21(vector<Vector2f> &vP1, const vector<Vector2f> &vP2) {
    const int N = vP1.size();
    Eigen::MatrixXf A(N, 9);
    for (size_t i = 0; i < 8; i++) {
        const float u1 = vP1[i][0];
        const float v1 = vP1[i][1];
        const float u2 = vP2[i][0];
        const float v2 = vP2[i][1];

        A(i, 0) = u2 * u1;
        A(i, 1) = u2 * v1;
        A(i, 2) = u2;
        A(i, 3) = v2 * u1;
        A(i, 4) = v2 * v1;
        A(i, 5) = v2;
        A(i, 6) = u1;
        A(i, 7) = v1;
        A(i, 8) = 1;
    }

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXf V = svd.matrixV();
    Matrix3f Fpre;
    Fpre << V(0, 8), V(1, 8), V(2, 8),
            V(3, 8), V(4, 8), V(5, 8),
            V(6, 8), V(7, 8), V(8, 8);       // 最后一列转成3x3
    Eigen::JacobiSVD<Eigen::Matrix3f> svd_F(Fpre, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Vector3f sigma = svd_F.singularValues();
    return svd_F.matrixU() * Eigen::DiagonalMatrix<float, 3>(sigma[0], sigma[1], 0) *
           svd_F.matrixV().transpose(); // 第3个奇异值设零
}
float CheckFundamental(Matrix3f &F21, vector<uchar>&vbMatchesInliers,vector<cv::Point2f>points1,vector<cv::Point2f>points2,float sigma) {
    const int N = points1.size();

    const float f11 = F21(0, 0);
    const float f12 = F21(0, 1);
    const float f13 = F21(0, 2);
    const float f21 = F21(1, 0);
    const float f22 = F21(1, 1);
    const float f23 = F21(1, 2);
    const float f31 = F21(2, 0);
    const float f32 = F21(2, 1);
    const float f33 = F21(2, 2);

    vbMatchesInliers.resize(N);

    float score = 0;

    const float th = 0.1;
    const float thScore = 1.5;

    const float invSigmaSquare = 1.0 / (sigma * sigma);

    for (int i = 0; i < N; i++) {
        bool bIn = true;

        const cv::Point2f &kp1 = points1[i];
        const cv::Point2f &kp2 = points2[i];

        const float u1 = kp1.x;
        const float v1 = kp1.y;
        const float u2 = kp2.x;
        const float v2 = kp2.y;

        // Reprojection error in second image
        // l2=F21x1=(a2,b2,c2)

        const float a2 = f11 * u1 + f12 * v1 + f13;
        const float b2 = f21 * u1 + f22 * v1 + f23;
        const float c2 = f31 * u1 + f32 * v1 + f33;

        const float num2 = a2 * u2 + b2 * v2 + c2;

        const float squareDist1 = num2 * num2 / (a2 * a2 + b2 * b2);

        const float chiSquare1 = squareDist1 * invSigmaSquare;

        if (chiSquare1 > th)
            bIn = false;
        else
            score += thScore - chiSquare1;

        // Reprojection error in second image
        // l1 =x2tF21=(a1,b1,c1)

        const float a1 = f11 * u2 + f21 * v2 + f31;
        const float b1 = f12 * u2 + f22 * v2 + f32;
        const float c1 = f13 * u2 + f23 * v2 + f33;

        const float num1 = a1 * u1 + b1 * v1 + c1;

        const float squareDist2 = num1 * num1 / (a1 * a1 + b1 * b1);

        const float chiSquare2 = squareDist2 * invSigmaSquare;

        if (chiSquare2 > th)
            bIn = false;
        else
            score += thScore - chiSquare2;

        if (bIn)
            vbMatchesInliers[i] = true;
        else
            vbMatchesInliers[i] = false;
    }
    return score;
}
void show(Matrix3f &F21, vector<uchar>&vbMatchesInliers,vector<cv::Point2f>points1,vector<cv::Point2f>points2,float sigma){
    const int N = points1.size();

    const float f11 = F21(0, 0);
    const float f12 = F21(0, 1);
    const float f13 = F21(0, 2);
    const float f21 = F21(1, 0);
    const float f22 = F21(1, 1);
    const float f23 = F21(1, 2);
    const float f31 = F21(2, 0);
    const float f32 = F21(2, 1);
    const float f33 = F21(2, 2);

    vbMatchesInliers.resize(N);

    float score = 0;

    const float th = 0.1;
    const float thScore = 5.991;

    const float invSigmaSquare = 1.0 / (sigma * sigma);

    for (int i = 0; i < N; i++) {
        if(vbMatchesInliers[i]){

        const cv::Point2f &kp1 = points1[i];
        const cv::Point2f &kp2 = points2[i];

        const float u1 = kp1.x;
        const float v1 = kp1.y;
        const float u2 = kp2.x;
        const float v2 = kp2.y;

        // Reprojection error in second image
        // l2=F21x1=(a2,b2,c2)

        const float a2 = f11 * u1 + f12 * v1 + f13;
        const float b2 = f21 * u1 + f22 * v1 + f23;
        const float c2 = f31 * u1 + f32 * v1 + f33;

        const float num2 = a2 * u2 + b2 * v2 + c2;

        const float squareDist1 = num2 * num2 / (a2 * a2 + b2 * b2);

        const float chiSquare1 = squareDist1 * invSigmaSquare;



        // Reprojection error in second image
        // l1 =x2tF21=(a1,b1,c1)

        const float a1 = f11 * u2 + f21 * v2 + f31;
        const float b1 = f12 * u2 + f22 * v2 + f32;
        const float c1 = f13 * u2 + f23 * v2 + f33;

        const float num1 = a1 * u1 + b1 * v1 + c1;

        const float squareDist2 = num1 * num1 / (a1 * a1 + b1 * b1);

        const float chiSquare2 = squareDist2 * invSigmaSquare;
        //cout<<"chiSqures1 is "<<chiSquare1<<endl;
        //cout<<"chiSqures2 is "<<chiSquare2<<endl;
        }
    }
};
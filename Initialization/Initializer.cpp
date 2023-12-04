

#include "Initialization/Initializer.hpp"

#include <glog/logging.h>
#include <thread>
#include <opencv2/core/eigen.hpp>

#include "Utils/Converter.hpp"
#include "Utils/UtilsCV.hpp"


namespace ORB_SLAM_Tracking {

Initializer::Initializer(const Frame& referenceFrame, float sigma, int iterations) {
  mK = referenceFrame.mK.clone();
  mvKeys1 = referenceFrame.mvKeysUn;
  mSigma = sigma;
  mSigma2 = sigma * sigma;
  mMaxIterations = iterations;
}

bool Initializer::Initialize(const Frame& currentFrame, const std::vector<int>& matches12,
                             PoseT& Tcw, std::vector<cv::Point3f>& vP3D,
                             std::vector<bool>& vbTriangulated) {
  mvKeys2 = currentFrame.mvKeysUn;

  mvMatches12.clear();
  mvMatches12.reserve(mvKeys2.size());
  mvbMatched1.resize(mvKeys1.size(), false);
  for (size_t i = 0; i < matches12.size(); i++) {
    if (matches12[i] >= 0) {
      mvMatches12.push_back(std::make_pair(i, matches12[i]));
      mvbMatched1[i] = true;
    } else {
      mvbMatched1[i] = false;
    }
  }

  // 匹配点对的数量
  const int N = mvMatches12.size();

  // Step 1 --------------------------------------------------------------
  // 生成200组（maxiterations）八个点的匹配点对，用作H、F矩阵的估计
  mvSets = std::vector<std::vector<size_t>>(mMaxIterations, std::vector<size_t>(8, 0));

  // 使用vAllIndices存储所有的匹配点对的索引，保证8个点不会选到同一个点
  std::vector<size_t> vAllIndices;
  vAllIndices.reserve(N);
  std::vector<size_t> vAvailableIndices;
  for (int i = 0; i < N; i++) {
    vAllIndices.push_back(i);
  }

  for (int it = 0; it < mMaxIterations; it++) {
    // 随机选取8个点
    vAvailableIndices = vAllIndices;

    for (size_t j = 0; j < 8; j++) {
      // 随机选取一个索引
      int randi = rand() % vAvailableIndices.size();
      int idx = vAvailableIndices[randi];
      // 将选取的索引从vAvailableIndices中删除(保证8个点不会选到同一个点)
      vAvailableIndices[randi] = vAvailableIndices.back();
      vAvailableIndices.pop_back();

      // 将选取的索引存储到vSets中
      mvSets[it][j] = idx;
    }
  }

  // Step 2 --------------------------------------------------------------
  // 启动两个线程来估计H、F矩阵

  // 最终估计出的H、F模型的内点
  std::vector<bool> vbMatchesInliersH, vbMatchesInliersF;
  // SH、SF分别表示H、F模型的双向重投影误差（F1->F2 + F2->F1）
  float SH, SF;
  Eigen::Matrix3f H, F;

  // 启动线程来估计H、F矩阵
  std::thread threadH(&Initializer::FindHomography, this, std::ref(vbMatchesInliersH), std::ref(SH),
                      std::ref(H));
  std::thread threadF(&Initializer::FindFundamental, this, std::ref(vbMatchesInliersF),
                      std::ref(SF), std::ref(F));

  // 等待线程结束
  threadH.join();
  threadF.join();

  // Step 3 --------------------------------------------------------------
  // 根据H、F模型的得分来选择最终的模型
  if (SH + SF == 0.f) return false;
  float RH = SH / (SH + SF);
  DLOG(INFO) << "Score of H: " << SH;
  DLOG(INFO) << "Score of F: " << SF;
  mF = Converter::toCvMat3(F);

  // 统计H,F模型inliers的数量
  int nH = 0, nF = 0;
  for (size_t i = 0; i < N; i++) {
    if (vbMatchesInliersH[i]) nH++;
    if (vbMatchesInliersF[i]) nF++;
  }
  DLOG(INFO) << "inliers of H: " << nH;
  DLOG(INFO) << "inliers of F: " << nF;

  int minParallax = 1.0;  // 视差角限制，单位是度

  // Step 4 --------------------------------------------------------------
  // 根据选择的模型来恢复位姿，并进行关键点三角化

  bool isReconstructed;
  Eigen::Matrix3f mKe = Converter::toMatrix3f(mK);
  // ORBSLAM2的RH阈值是0.4,ORBSLAM3的RH阈值是0.5
  if (RH > 0.50) {
    isReconstructed =
        ReconstructHF(vbMatchesInliersH, H, mKe, Tcw, vP3D, vbTriangulated, false, minParallax, 50);
  } else {
    isReconstructed =
        ReconstructHF(vbMatchesInliersF, F, mKe, Tcw, vP3D, vbTriangulated, true, minParallax, 50);
  }

  if (!isReconstructed) {
    std::cerr << "Initialization failed!";
    return false;
  }

  return true;
}  // function Initialize


void Normalize(const std::vector<cv::KeyPoint> &vKeys, std::vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T)
{
    float meanX = 0;
    float meanY = 0;
    const int N = vKeys.size();

    vNormalizedPoints.resize(N);

    for(int i=0; i<N; i++)
    {
        meanX += vKeys[i].pt.x;
        meanY += vKeys[i].pt.y;
    }

    meanX = meanX/N;
    meanY = meanY/N;

    float meanDevX = 0;
    float meanDevY = 0;

    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
        vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

        meanDevX += fabs(vNormalizedPoints[i].x);
        meanDevY += fabs(vNormalizedPoints[i].y);
    }

    meanDevX = meanDevX/N;
    meanDevY = meanDevY/N;

    float sX = 1.0/meanDevX;
    float sY = 1.0/meanDevY;

    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
        vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
    }

    T = cv::Mat::eye(3,3,CV_32F);
    T.at<float>(0,0) = sX;
    T.at<float>(1,1) = sY;
    T.at<float>(0,2) = -meanX*sX;
    T.at<float>(1,2) = -meanY*sY;
} // function Normalize


void Initializer::FindHomographyCV(std::vector<bool>& vbMatchesInliers, float& score,
                                   Eigen::Matrix3f& H21) {
  // 直接通过OpenCV接口估计单应性矩阵
  std::vector<cv::Point2f> srcPoints, dstPoints;
  for (size_t matchIdx = 0; matchIdx < mvMatches12.size(); matchIdx++) {
    const size_t idx1 = mvMatches12[matchIdx].first;
    const size_t idx2 = mvMatches12[matchIdx].second;

    const cv::KeyPoint& kp1 = mvKeys1[idx1];
    const cv::KeyPoint& kp2 = mvKeys2[idx2];

    srcPoints.push_back(kp1.pt);
    dstPoints.push_back(kp2.pt);
  }

  cv::Mat H21Mat;
  cv::Mat maskUnused_;
  H21Mat = cv::findHomography(srcPoints, dstPoints, cv::RANSAC, 2, maskUnused_, 10000, 0.999);
  H21 = Converter::toMatrix3f(H21Mat);
  score = CheckHomography(H21, H21.inverse(), vbMatchesInliers, mSigma);

}

void Initializer::FindFundamentalCV(std::vector<bool> &vbMatchesInliers, float &score, Eigen::Matrix3f &F21) {
  // 直接通过Opencv接口估计基础矩阵
  std::vector<cv::Point2f> _NormSrcPoints, _NormDstPoints;
  std::vector<cv::Point2f> srcPoints, dstPoints;
  cv::Mat T1, T2;
  Normalize(mvKeys1, _NormSrcPoints, T1);
  Normalize(mvKeys2, _NormDstPoints, T2);
  cv::Mat T2t = T2.t();
  for (size_t matchIdx = 0; matchIdx < mvMatches12.size(); matchIdx++) {
    const size_t idx1 = mvMatches12[matchIdx].first;
    const size_t idx2 = mvMatches12[matchIdx].second;

    // const cv::KeyPoint& kp1 = mvKeys1[idx1];
    // const cv::KeyPoint& kp2 = mvKeys2[idx2];

    // srcPoints.push_back(kp1.pt);
    // dstPoints.push_back(kp2.pt);

    // for normalized version
    srcPoints.push_back(_NormSrcPoints[idx1]);
    dstPoints.push_back(_NormDstPoints[idx2]);    
  }
  cv::Mat F21Mat;
  // cv::findFundamentalMat : https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga59b0d57f46f8677fb5904294a23d404a
  F21Mat = cv::findFundamentalMat(srcPoints, dstPoints, cv::FM_RANSAC, 1.95, 0.999);
  F21Mat.convertTo(F21Mat, CV_32F);
  T2t.convertTo(T2t, CV_32F);
  T1.convertTo(T1, CV_32F);
  F21Mat = T2t * F21Mat * T1;
  F21 = Converter::toMatrix3f(F21Mat);
  score = CheckFundamental(F21, vbMatchesInliers, mSigma);
}

void Initializer::FindHomography(std::vector<bool>& vbMatchesInliers, float& score,
                                 Eigen::Matrix3f& H21) {
  // 通过RANSAC来估计单应性矩阵
  cv::Mat H21i;
  Eigen::Matrix3f H21ie;
  float currentScore;
  std::vector<bool> vbCurrentInliers;

  // 找到的最好结果
  score = 0;
  vbMatchesInliers = std::vector<bool>(mvMatches12.size(), false);
  for (int it = 0; it < mMaxIterations; it++) {
    // from mvSets construct srcPoints and dstPoints
    std::vector<cv::Point2f> srcPoints, dstPoints;
    for (size_t j = 0; j < 8; j++) {
      const size_t idx1 = mvMatches12[mvSets[it][j]].first;
      const size_t idx2 = mvMatches12[mvSets[it][j]].second;

      const cv::KeyPoint& kp1 = mvKeys1[idx1];
      const cv::KeyPoint& kp2 = mvKeys2[idx2];

      srcPoints.push_back(kp1.pt);
      dstPoints.push_back(kp2.pt);
    }

    // 使用opencv接口求解单应性矩阵;原始代码这里使用的是自己手写的一个求解单应性矩阵的函数，使用SVD求解
    // method = 0 使用最小二乘法求解
    // api :
    // https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780
    H21i = cv::findHomography(srcPoints, dstPoints, 0);

    H21ie = Converter::toMatrix3f(H21i);

    // 计算单应性矩阵的得分
    currentScore = CheckHomography(H21ie, H21ie.inverse(), vbCurrentInliers, mSigma);

    // 如果当前得分比最好的得分还要高，则更新最好的得分
    if (currentScore > score) {
      score = currentScore;
      vbMatchesInliers = vbCurrentInliers;
      H21 = H21ie;
    }
  }  // for (int it = 0; it < mMaxIterations; it++)
}  // function FindHomography


void Initializer::FindFundamental(std::vector<bool>& vbMatchesInliers, float& score,
                                  Eigen::Matrix3f& F21) {
  // 通过RANSAC来估计基础矩阵
  // 原始代码在计算基础矩阵是自己用SVD求解，这里改成直接调用Opencv的api
  // 但是最外层的RANSAC还是采用原始代码的实现

  // 清空存储结果变量
  score = 0;
  vbMatchesInliers = std::vector<bool>(mvMatches12.size(), false);

  // 归一化点
  std::vector<cv::Point2f> vPn1, vPn2;
  cv::Mat T1, T2;
  Normalize(mvKeys1, vPn1, T1);
  Normalize(mvKeys2, vPn2, T2);
  cv::Mat T2t = T2.t();

  // RANSAC循环变量
  score = 0;
  std::vector<bool> vbCurrentInliers;
  float currentScore;
  cv::Mat F21i;
  Eigen::Matrix3f F21ie;  // F21i的Eigen表示

  // 通过RANSAC来估计基础矩阵
  for (int it = 0; it < mMaxIterations; it++) {
    vbCurrentInliers.clear();
    // from mvSets construct srcPoints and dstPoints
    cv::Mat srcPoints(8, 2, CV_32F);
    cv::Mat dstPoints(8, 2, CV_32F);
    for (size_t j = 0; j < 8; j++) {
      const size_t idx1 = mvMatches12[mvSets[it][j]].first;
      const size_t idx2 = mvMatches12[mvSets[it][j]].second;

      const cv::Point2f& kp1 = vPn1[idx1];
      const cv::Point2f& kp2 = vPn2[idx2];

      srcPoints.at<float>(j, 0) = kp1.x;
      srcPoints.at<float>(j, 1) = kp1.y;
      dstPoints.at<float>(j, 0) = kp2.x;
      dstPoints.at<float>(j, 1) = kp2.y;
    }
    srcPoints = srcPoints.reshape(2);
    dstPoints = dstPoints.reshape(2);

    // 使用opencv接口求解基础矩阵;原始代码这里使用的是自己手写的一个求解基础矩阵的函数，使用SVD求解
    // method = FM_8POINT 使用8点法求解
    // https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga59b0d57f46f8677fb5904294a23d404a
    // F21i = cv::findFundamentalMat(srcPoints, dstPoints, cv::FM_8POINT);
    // Debug
    {
      bool ret = UtilsCV::ComputeF21(srcPoints, dstPoints, F21i);
      if (!ret) {
        continue;
      }
    }
    
    F21i.convertTo(F21i, CV_32F);
    F21i = T2t * F21i * T1;
    F21ie = Converter::toMatrix3f(F21i);

    // 计算基础矩阵的得分
    currentScore = CheckFundamental(F21ie, vbCurrentInliers, mSigma);

    // 如果当前得分比最好的得分还要高，则更新最好的得分
    if (currentScore > score) {
      score = currentScore;
      vbMatchesInliers = vbCurrentInliers;
      F21 = F21ie;
      // Debug
      // DLOG(INFO) << "FindFundamental by Ransac in iter = " << it << "F21 = \n" << F21 / F21(2,2);
    }
  }  // for (int it = 0; it < mMaxIterations)
}  // function FindFundamental

float Initializer::CheckHomography(const Eigen::Matrix3f& H21, const Eigen::Matrix3f& H12,
                                   std::vector<bool>& vbMatchesInliers, float sigma) {
  const int N = mvMatches12.size();

  const float h11 = H21(0, 0);
  const float h12 = H21(0, 1);
  const float h13 = H21(0, 2);
  const float h21 = H21(1, 0);
  const float h22 = H21(1, 1);
  const float h23 = H21(1, 2);
  const float h31 = H21(2, 0);
  const float h32 = H21(2, 1);
  const float h33 = H21(2, 2);

  const float h11inv = H12(0, 0);
  const float h12inv = H12(0, 1);
  const float h13inv = H12(0, 2);
  const float h21inv = H12(1, 0);
  const float h22inv = H12(1, 1);
  const float h23inv = H12(1, 2);
  const float h31inv = H12(2, 0);
  const float h32inv = H12(2, 1);
  const float h33inv = H12(2, 2);

  vbMatchesInliers.resize(N);

  float score = 0;
  // 基于卡方检验计算出的阈值 自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值
  const float th = 5.991;

  const float invSigmaSquare = 1.0 / (sigma * sigma);

  for (int i = 0; i < N; i++) {
    bool bIn = true;

    const cv::KeyPoint& kp1 = mvKeys1[mvMatches12[i].first];
    const cv::KeyPoint& kp2 = mvKeys2[mvMatches12[i].second];

    const float u1 = kp1.pt.x;
    const float v1 = kp1.pt.y;
    const float u2 = kp2.pt.x;
    const float v2 = kp2.pt.y;

    // Reprojection error in first image
    // x2in1 = H12*x2

    // 计算投影误差，2投1
    // 单应矩阵投影就是直接的矩阵相乘（然后归一化）：p1 = H12 * p2, H12 = [h11 h12 h13; h21 h22 h23;
    // h31 h32 h33], p1 = [u1; v1; 1], p2 = [u2; v2; 1]
    const float w2in1inv = 1.0 / (h31inv * u2 + h32inv * v2 + h33inv);
    const float u2in1 = (h11inv * u2 + h12inv * v2 + h13inv) * w2in1inv;
    const float v2in1 = (h21inv * u2 + h22inv * v2 + h23inv) * w2in1inv;

    const float squareDist1 = (u1 - u2in1) * (u1 - u2in1) + (v1 - v2in1) * (v1 - v2in1);

    const float chiSquare1 = squareDist1 * invSigmaSquare;

    if (chiSquare1 > th)
      bIn = false;
    else
      score += th - chiSquare1;

    // Reprojection error in second image
    // x1in2 = H21*x1

    const float w1in2inv = 1.0 / (h31 * u1 + h32 * v1 + h33);
    const float u1in2 = (h11 * u1 + h12 * v1 + h13) * w1in2inv;
    const float v1in2 = (h21 * u1 + h22 * v1 + h23) * w1in2inv;

    const float squareDist2 = (u2 - u1in2) * (u2 - u1in2) + (v2 - v1in2) * (v2 - v1in2);

    const float chiSquare2 = squareDist2 * invSigmaSquare;

    if (chiSquare2 > th)
      bIn = false;
    else
      score += th - chiSquare2;

    if (bIn)
      vbMatchesInliers[i] = true;
    else
      vbMatchesInliers[i] = false;
  }

  return score;
}

float Initializer::CheckFundamental(const Eigen::Matrix3f& F21, std::vector<bool>& vbMatchesInliers,
                                    float sigma) {
  const int N = mvMatches12.size();

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

  // 基于卡方检验计算出的阈值 自由度为1的卡方分布，显著性水平为0.05，对应的临界阈值
  const float th = 3.841;
  // 基于卡方检验计算出的阈值 自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值
  const float thScore = 5.991;

  const float invSigmaSquare = 1.0 / (sigma * sigma);

  for (int i = 0; i < N; i++) {
    bool bIn = true;

    const cv::KeyPoint& kp1 = mvKeys1[mvMatches12[i].first];
    const cv::KeyPoint& kp2 = mvKeys2[mvMatches12[i].second];

    const float u1 = kp1.pt.x;
    const float v1 = kp1.pt.y;
    const float u2 = kp2.pt.x;
    const float v2 = kp2.pt.y;

    // Reprojection error in second image
    // l2 = F21 * p1 = (a2,b2,c2)
    // 计算 img1 上的点在 img2 上投影得到的极线 l2 = F21 * p1 = (a2,b2,c2)
    const float a2 = f11 * u1 + f12 * v1 + f13;
    const float b2 = f21 * u1 + f22 * v1 + f23;
    const float c2 = f31 * u1 + f32 * v1 + f33;

    // 计算误差 e = (a * p2.x + b * p2.y + c) /  sqrt(a * a + b * b)
    const float num2 = a2 * u2 + b2 * v2 + c2;

    const float squareDist1 = num2 * num2 / (a2 * a2 + b2 * b2);

    const float chiSquare1 = squareDist1 * invSigmaSquare;

    // 自由度为1是因为这里的计算是点到线的距离
    if (chiSquare1 > th)
      bIn = false;
    else
      // thScore保持跟计算单应矩阵时一样是为了保证两者分数的可比性
      // 但是实际中还是可能会出现F矩阵的得分要更高，因为这里的th比计算单应矩阵时的更小，导致score的最小值会更大
      score += thScore - chiSquare1;

    // Reprojection error in second image
    // l1 = F21^T * p2 = (a1,b1,c1)
    // 与上面相同只不过反过来了
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
}  // function CheckFundamental

bool Initializer::ReconstructHF(std::vector<bool>& vbMatchesInliers, Eigen::Matrix3f& HF21,
                                Eigen::Matrix3f& K, PoseT& T21, std::vector<cv::Point3f>& vP3D,
                                std::vector<bool>& vbTriangulated, bool isF, float minParallax,
                                int minTriangulated) {
  // ------------------- 1. 从单应矩阵或者基础矩阵恢复R,t的多个解 -------------------
  std::vector<cv::Mat> vRs, vts;  // 恢复出的R,t的可能解
  int nSolution = 0;              // 解的数量

  // 从F、H恢复R,t
  if (isF) {  // 从F恢复R,t
    LOG(INFO) << "Reconstruct from F";
    // F = K^{-T} * E * K^{-1}
    // E = K^{T} * F * K
    Eigen::Matrix3f E21 = K.transpose() * HF21 * K;
    cv::Mat E21cv = Converter::toCvMat3(E21);
    cv::Mat R1, R2, t;
    // 分解本质矩阵E，得到四组R,t
    // api :
    // https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga54a2f5b3f8aeaf6c76d4a31dece85d5d
    cv::decomposeEssentialMat(E21cv, R1, R2, t);
    // UtilsCV::DecomposeE(E21cv, R1, R2, t);

    // 将R1,R2,t构造配对并放入vRs,vts中
    // 四种可能的解[R1,t], [R1,-t], [R2,t], [R2,-t]
    vRs.push_back(R1);
    vts.push_back(t);
    vRs.push_back(R1);
    vts.push_back(-t);
    vRs.push_back(R2);
    vts.push_back(t);
    vRs.push_back(R2);
    vts.push_back(-t);
    nSolution = 4;

  } else {  // 从H矩阵恢复R,t
    LOG(INFO) << "Reconstruct from H";
    cv::Mat H21 = Converter::toCvMat3(HF21);
    cv::Mat Kcv = Converter::toCvMat3(K);
    // cv::Mat Rcvs, tcvs, planeNormals;
    std::vector<cv::Mat> Rcvs, tcvs, planeNormals;
    // 分解单应矩阵H，得到四组R,t
    // api :
    // https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga7f60bdff78833d1e3fd6d9d0fd538d92
    nSolution = cv::decomposeHomographyMat(H21, Kcv, Rcvs, tcvs, planeNormals);
    DLOG(INFO) << "nSolution = " << nSolution << ", Rcvs.size() = " << Rcvs.size()
              << ", tcvs.size() = " << tcvs.size() << ", planeNormals.size() = "
              << planeNormals.size();

    // 将Rcvs,tcvs构造配对并放入vRs,vts中
    for (int i = 0; i < nSolution; i++) {
      cv::Mat _R, _t;
      Rcvs[i].convertTo(_R, CV_32F);
      tcvs[i].convertTo(_t, CV_32F);
      vRs.push_back(_R);
      vts.push_back(_t);

      // Debug
      {
        DLOG(WARNING) << "Recover R/t from H in solution " << i << "; R = \n" << vRs[i];
        DLOG(WARNING) << "Recover R/t from H in solution " << i << "; t = \n" << vts[i];
      }
    }
  }  // if (isF)

  // ------------------- 2. 遍历R,t，记录内点最多的解 -------------------
  cv::Mat Kcv = Converter::toCvMat3(K);

  int bestGood = 0;
  int secondBestGood = 0;
  int bestSolutionIdx = -1;
  float bestParallax = -1;
  std::vector<cv::Point3f> bestP3D;
  std::vector<bool> bestTriangulated;

  for (int i = 0; i < nSolution; ++i) {
    float parallaxi;
    std::vector<cv::Point3f> vP3Di;
    std::vector<bool> vbTriangulatedi;
    // 通过当前R,t三角化关键点，并计算内点数量
    int nGood = CheckRT2(vRs[i], vts[i], mvKeys1, mvKeys2, mvMatches12, vbMatchesInliers, Kcv, vP3Di,
                        4.0 * mSigma2, vbTriangulatedi, parallaxi);
    // Debug
    {
      DLOG(WARNING) << "Debug F ------------------";
      DLOG(WARNING) << "For solution " << i;
      cv::Mat pose = cv::Mat::eye(4, 4, CV_32F);
      vRs[i].copyTo(pose.rowRange(0, 3).colRange(0, 3));
      vts[i].copyTo(pose.rowRange(0, 3).col(3));
      DLOG(WARNING) << "pose = \n" << pose;
      DLOG(WARNING) << "nGood = " << nGood;
      DLOG(WARNING) << "K = \n" << Kcv;
    }

    if (nGood > bestGood) {
      secondBestGood = bestGood;
      bestGood = nGood;
      bestSolutionIdx = i;
      bestParallax = parallaxi;
      bestP3D = vP3Di;
      bestTriangulated = vbTriangulatedi;
    } else if (nGood > secondBestGood) {
      secondBestGood = nGood;
    }
  }

  // ------------------- 3. 选择最佳的R,t -------------------
  // 统计合法的匹配数量（在估计H、F时的内点数量）
  int N = 0;
  for (size_t i = 0; i < vbMatchesInliers.size(); ++i) {
    if (vbMatchesInliers[i]) {
      N++;
    }
  }
  // 在原始代码中，重建F使用阈值0.7,重建H时使用阈值0.75.这里使用F阈值0.7
  bool triRet = true;
  if (secondBestGood > 0.7 * bestGood) {
    LOG(INFO) << "Failed: Second best solution = " << secondBestGood
              << " is too close to best solution = " << bestGood;
    triRet = false;
  }

  if (bestParallax < minParallax) {
    LOG(INFO) << "Failed: Parallax = " << bestParallax << " is below the minimum threshold ("
              << minParallax << ")";
    triRet = false;
  }

  if (bestGood < minTriangulated) {
    LOG(INFO) << "Failed: Number of triangulated points = " << bestGood
              << " is below the minimum threshold (" << minTriangulated << ")";
    triRet = false;
  }

  if (bestGood < 0.9 * N) {
    LOG(INFO) << "Failed: Number of inliers = " << bestGood
              << " is less than 90% of total matches = " << N;
    triRet = false;
  }

  if (!triRet) {
    return false;
  }

  // ------------------- 4. 保存结果 -------------------
  // 保存最佳的R,t
  Eigen::Matrix3f R = Converter::toMatrix3f(vRs[bestSolutionIdx]);
  Eigen::Vector3f t = Converter::toVector3f(vts[bestSolutionIdx]);
  // construct T21 from R,t
  T21 = Eigen::Affine3d::Identity();
  T21.linear() = R.cast<double>();
  T21.translation() = t.cast<double>();
  vbTriangulated = bestTriangulated;
  vP3D = bestP3D;
  return true;
}  // function ReconstructHF

int Initializer::CheckRT(const cv::Mat& R21, const cv::Mat& t21,
                         const std::vector<cv::KeyPoint>& vKeys1,
                         const std::vector<cv::KeyPoint>& vKeys2,
                         const std::vector<Match>& vMatches12, std::vector<bool>& vbMatchesInliers,
                         const cv::Mat& K, std::vector<cv::Point3f>& vP3D, float th2,
                         std::vector<bool>& vbTriGood, float& parallax) {
  // ------------------- 1. 计算相机1和相机2的投影矩阵 -------------------

  cv::Mat P1(3, 4, CV_32F);
  cv::Mat P2(3, 4, CV_32F);
  // 将P1,P2设置为0
  P1.setTo(0);
  P2.setTo(0);

  // 设置相机1的投影矩阵 -- 相机1是世界坐标系，所以投影矩阵就是内参
  K.copyTo(P1(cv::Rect(0, 0, 3, 3)));
  // 设置相机2的投影矩阵 -- 相机2的投影矩阵是K[R|t]
  R21.copyTo(P2(cv::Rect(0, 0, 3, 3)));
  t21.copyTo(P2.col(3));
  P2 = K * P2;

  // 相机1,2的光心位置
  cv::Mat O1, O2;
  O1 = cv::Mat::zeros(3, 1, CV_32F);  // 相机1的光心就是原点（世界坐标系）
  O2 = -R21.t() * t21;

  // 设置输出变量
  vbTriGood = std::vector<bool>(vKeys1.size(), false);
  vP3D.resize(vKeys1.size());
  std::vector<float> vCosParallax;
  int nGood = 0;  // 内点数量

  // ------------------- 2. 三角化点 -------------------
  // 将vKeys1,vKeys2转换为cv::Mat 2xN 类型
  std::vector<cv::Point2f> vP1, vP2;  // N x 2
  cv::Mat vP3Dc1;                     // N x 3
  cv::Mat vp4Dc1;                     // 4 x N

  std::vector<size_t> vIndices;
  vP1.reserve(vMatches12.size());
  vP2.reserve(vMatches12.size());
  for (size_t i = 0; i < vMatches12.size(); ++i) {
    if (!vbMatchesInliers[i]) continue;
    vP1.push_back(vKeys1[vMatches12[i].first].pt);
    vP2.push_back(vKeys2[vMatches12[i].second].pt);
    vIndices.push_back(i);
  }

  // api : https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gad3fc9a0c82b08df034234979960b778c
  cv::triangulatePoints(P1, P2, vP1, vP2, vp4Dc1);
  vp4Dc1.convertTo(vp4Dc1, CV_32F);
  K.convertTo(K, CV_32F);
  // LOG(INFO) << "vp4Dc1 : " << vp4Dc1.t();

  // ------------------- 3. 遍历3D点，过滤非法3D点 -------------------
  for (int i = 0; i < vp4Dc1.cols; ++i) {
    // 3D点的齐次坐标
    cv::Mat x3Dc1 = vp4Dc1.col(i);
    if (x3Dc1.depth() != CV_32F) exit(1);
    // 3D点的非齐次坐标
    cv::Mat x3Dc1n = x3Dc1.rowRange(0, 3) / x3Dc1.at<float>(3);
    // 图1像素值
    const float u1 = vP1[i].x;
    const float v1 = vP1[i].y;
    // 图2像素值
    const float u2 = vP2[i].x;
    const float v2 = vP2[i].y;

    // 过滤数值不合理的3D点
    if (!isfinite(x3Dc1.at<float>(0)) || !isfinite(x3Dc1.at<float>(1)) ||
        !isfinite(x3Dc1.at<float>(2))) {
      vbTriGood[vMatches12[i].first] = false;
      continue;
    }

    // 过滤掉（0,0,0）的点
    if (x3Dc1.at<float>(0) == 0 && x3Dc1.at<float>(1) == 0 && x3Dc1.at<float>(2) == 0) {
      vbTriGood[vMatches12[i].first] = false;
      continue;
    }


    // ------------------- 3.1 计算3D点到两个相机光心的余弦值 -------------------
    // 3D点到相机1,2光心的向量
    cv::Mat OC1 = x3Dc1n - O1;
    cv::Mat OC2 = x3Dc1n - O2;

    // 3D点到相机1,2光心的距离
    const float dist1 = cv::norm(OC1);
    const float dist2 = cv::norm(OC2);
    // 3D点到相机1,2光心的余弦值
    const float cosParallax = OC1.dot(OC2) / (dist1 * dist2);

    // ------------------- 3.2 计算3D点在相机1,2上的投影，并过滤深度值为负的点 -------------------
    // 3D点在相机1,2上的投影
    cv::Mat x3Dc2n = R21 * x3Dc1n + t21;

    // 只当余弦值小于0.99998时才检查深度值
    // 余弦值非常大（也就是角度非常小），意味着该点可能是无穷远点，有可能会出现负的深度值
    if (x3Dc1n.at<float>(2) <= 0 && cosParallax < 0.99998) continue;
    if (x3Dc2n.at<float>(2) <= 0 && cosParallax < 0.99998) continue;

    // ------------------- 3.3 计算3D点在相机1,2上的投影误差，并过滤投影误差大的点
    // -------------------

    // 3D点在相机1,2上的投影
    float im1x, im1y;
    float invZ1 = 1.0 / x3Dc1n.at<float>(2);
    im1x = K.at<float>(0, 0) * x3Dc1n.at<float>(0) * invZ1 + K.at<float>(0, 2);
    im1y = K.at<float>(1, 1) * x3Dc1n.at<float>(1) * invZ1 + K.at<float>(1, 2);
    float squareError1 = (im1x - u1) * (im1x - u1) + (im1y - v1) * (im1y - v1);

    float im2x, im2y;
    float invZ2 = 1.0 / x3Dc2n.at<float>(2);
    im2x = K.at<float>(0, 0) * x3Dc2n.at<float>(0) * invZ2 + K.at<float>(0, 2);
    im2y = K.at<float>(1, 1) * x3Dc2n.at<float>(1) * invZ2 + K.at<float>(1, 2);
    float squareError2 = (im2x - u2) * (im2x - u2) + (im2y - v2) * (im2y - v2);

    // 过滤投影误差大的点
    if (squareError1 > th2 || squareError2 > th2) continue;

    // ------------------- 3.4 接收内点 -------------------
    vCosParallax.push_back(cosParallax);
    vP3D[vMatches12[i].first] = cv::Point3f(x3Dc1n.at<float>(0), x3Dc1n.at<float>(1), x3Dc1n.at<float>(2));
    
    nGood++;
    // arccos(0.99998) = 0.36°
    if (cosParallax < 0.99998) {
      vbTriGood[vMatches12[i].first] = true;
    }

  }  // 遍历重建后的3D点

  // ------------------- 4. 对余弦值排序（小到大），选择第50个 -------------------
  if (nGood > 0) {
    // 对余弦值排序（小到大）
    sort(vCosParallax.begin(), vCosParallax.end());
    size_t idx = std::min(50, int(vCosParallax.size() - 1));
    parallax = acos(vCosParallax[idx]) * 180 / M_PI;
  } else {
    parallax = 0;
  }

  return nGood;

}  // function CheckRT


int Initializer::CheckRT2(const cv::Mat &R, const cv::Mat &t, const std::vector<cv::KeyPoint> &vKeys1, const std::vector<cv::KeyPoint> &vKeys2,
                       const std::vector<Match> &vMatches12, std::vector<bool> &vbMatchesInliers,
                       const cv::Mat &K, std::vector<cv::Point3f> &vP3D, float th2, std::vector<bool> &vbGood, float &parallax) {
  // Calibration parameters
  const float fx = K.at<float>(0, 0);
  const float fy = K.at<float>(1, 1);
  const float cx = K.at<float>(0, 2);
  const float cy = K.at<float>(1, 2);

  vbGood = std::vector<bool>(vKeys1.size(), false);
  vP3D.resize(vKeys1.size());

  std::vector<float> vCosParallax;
  vCosParallax.reserve(vKeys1.size());

  // Camera 1 Projection Matrix K[I|0]
  cv::Mat P1(3, 4, CV_32F, cv::Scalar(0));
  K.copyTo(P1.rowRange(0, 3).colRange(0, 3));

  cv::Mat O1 = cv::Mat::zeros(3, 1, CV_32F);

  // Camera 2 Projection Matrix K[R|t]
  cv::Mat P2(3, 4, CV_32F);
  R.copyTo(P2.rowRange(0, 3).colRange(0, 3));
  t.copyTo(P2.rowRange(0, 3).col(3));
  P2 = K * P2;

  cv::Mat O2 = -R.t() * t;

  int nGood = 0;

  for (size_t i = 0, iend = vMatches12.size(); i < iend; i++) {
    if (!vbMatchesInliers[i]) continue;

    const cv::KeyPoint& kp1 = vKeys1[vMatches12[i].first];
    const cv::KeyPoint& kp2 = vKeys2[vMatches12[i].second];
    cv::Mat p3dC1;
    cv::Mat p4dC1;

    UtilsCV::Triangulate(kp1, kp2, P1, P2, p3dC1);
    cv::Mat kp1Mat, kp2Mat;
    kp1Mat = (cv::Mat_<float>(2, 1) << kp1.pt.x, kp1.pt.y);
    kp2Mat = (cv::Mat_<float>(2, 1) << kp2.pt.x, kp2.pt.y);
    cv::triangulatePoints(P1, P2, kp1Mat, kp2Mat, p4dC1);
    p3dC1 = p4dC1.col(0);
    p3dC1 = p3dC1.rowRange(0, 3) / p3dC1.at<float>(3);


    if (!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) ||
        !isfinite(p3dC1.at<float>(2))) {
      vbGood[vMatches12[i].first] = false;
      continue;
    }

    // Check parallax
    cv::Mat normal1 = p3dC1 - O1;
    float dist1 = cv::norm(normal1);

    cv::Mat normal2 = p3dC1 - O2;
    float dist2 = cv::norm(normal2);

    float cosParallax = normal1.dot(normal2) / (dist1 * dist2);

    // Check depth in front of first camera (only if enough parallax, as "infinite" points can
    // easily go to negative depth)
    if (p3dC1.at<float>(2) <= 0 && cosParallax < 0.99998) continue;

    // Check depth in front of second camera (only if enough parallax, as "infinite" points can
    // easily go to negative depth)
    cv::Mat p3dC2 = R * p3dC1 + t;

    if (p3dC2.at<float>(2) <= 0 && cosParallax < 0.99998) continue;

    // Check reprojection error in first image
    float im1x, im1y;
    float invZ1 = 1.0 / p3dC1.at<float>(2);
    im1x = fx * p3dC1.at<float>(0) * invZ1 + cx;
    im1y = fy * p3dC1.at<float>(1) * invZ1 + cy;

    float squareError1 =
        (im1x - kp1.pt.x) * (im1x - kp1.pt.x) + (im1y - kp1.pt.y) * (im1y - kp1.pt.y);

    if (squareError1 > th2) continue;

    // Check reprojection error in second image
    float im2x, im2y;
    float invZ2 = 1.0 / p3dC2.at<float>(2);
    im2x = fx * p3dC2.at<float>(0) * invZ2 + cx;
    im2y = fy * p3dC2.at<float>(1) * invZ2 + cy;

    float squareError2 =
        (im2x - kp2.pt.x) * (im2x - kp2.pt.x) + (im2y - kp2.pt.y) * (im2y - kp2.pt.y);

    if (squareError2 > th2) continue;

    vCosParallax.push_back(cosParallax);
    vP3D[vMatches12[i].first] =
        cv::Point3f(p3dC1.at<float>(0), p3dC1.at<float>(1), p3dC1.at<float>(2));
    nGood++;

    if (cosParallax < 0.99998) vbGood[vMatches12[i].first] = true;
  }

  if (nGood > 0) {
    sort(vCosParallax.begin(), vCosParallax.end());

    size_t idx = std::min(50, int(vCosParallax.size() - 1));
    parallax = acos(vCosParallax[idx]) * 180 / CV_PI;
  } else
    parallax = 0;

  return nGood;
}

}  // namespace ORB_SLAM_Tracking

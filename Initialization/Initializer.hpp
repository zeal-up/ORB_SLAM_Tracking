#pragma once

#include <opencv2/opencv.hpp>

#include "SlamTypes/Frame.hpp"

namespace ORB_SLAM_Tracking {

class Initializer {
  typedef std::pair<int, int> Match;

 public:
  /**
   * @brief Construct a new Initializer object
   *
   * @param referenceFrame 设置参考帧（第一帧）
   * @param sigma 归一化重投影误差的参数。最终计算出来的双边误差的平方值会除以sigma的平方
   * @param iterations RANSAC迭代次数;使用RANSAC来估计H,F矩阵
   */
  Initializer(const Frame &referenceFrame, float sigma = 1.0, int iterations = 200);

  /**
   * @brief 初始化函数，用于初始化ORB_SLAM的跟踪器
   *
   * @param currentFrame 当前帧
   * @param matches12 匹配点对的索引
   * @param Tcw 相机位姿（根据匹配点计算出currentFrame的位姿）[Output]
   * @param vP3D 三维点云坐标 [Output]
   * @param vbTriangulated 三维点云是否被三角化的标志 [Output]
   * @return bool 初始化是否成功的标志
   */
  bool Initialize(const Frame &currentFrame, const std::vector<int> &matches12, PoseT &Tcw,
                  std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated);

 private:
  /**
   * 寻找单应性矩阵
   * Step 1 : 从mvSets中随机选取8个点，计算出单应性矩阵H21
   * Step 2 : 计算单应性矩阵H21的得分
   * Step 3 : 重复Step 1和Step 2，直到迭代次数达到最大值
   * Step 4 : 选取得分最高的单应性矩阵H21
   * @param vbMatchesInliers 匹配点是否为内点的标志向量[Output]
   * @param score 单应性矩阵的得分[Output]
   * @param H21 输出的单应性矩阵。p2 = H21*p1 ; p1是参考帧中的点，p2是当前帧中的点。 由于F1可以理解为世界坐标系，
   * 所以最终求出来的T是世界坐标系到相机坐标系的变换矩阵Tcw[Output]
   */
  void FindHomography(std::vector<bool> &vbMatchesInliers, float &score, Eigen::Matrix3f &H21);

  /**
   * 寻找基础矩阵
   * @param vbMatchesInliers 匹配点是否为内点的标志向量[Output]
   * @param score 存储基础矩阵的得分[Output]
   * @param F21 存储计算得到的基础矩阵。p2 = F21*p1 ; p1是参考帧中的点，p2是当前帧中的点。 由于F1可以理解为世界坐标系，
   * 所以最终求出来的T是世界坐标系到相机坐标系的变换矩阵Tcw[Output]
   */
  void FindFundamental(std::vector<bool> &vbMatchesInliers, float &score, Eigen::Matrix3f &F21);

  /**
   * @brief 计算单应矩阵的分数。将F1的关键点投影到F2上计算重投影误差+将F2的关键点投影到F1上计算重投影误差
   * 单应矩阵直接可以直接将像素坐标点转到像素坐标点，因此距离的计算可以直接计算两个坐标点的欧式距离（的平方）
   *
   * @param H21 参考帧到当前帧的单应性矩阵
   * @param H12 当前帧到参考帧的单应性矩阵
   * @param vbMatchesInliers 匹配点是否为内点的标志向量[Output]
   * @param sigma 误差归一化参数
   * @return float 单应性矩阵的得分
   */
  float CheckHomography(const Eigen::Matrix3f &H21, const Eigen::Matrix3f &H12, std::vector<bool> &vbMatchesInliers,
                        float sigma);

  /**
   * @brief 计算基础矩阵的分数。将F1的关键点投影到F2上计算重投影误差+将F2的关键点投影到F1上计算重投影误差
   * 基础矩阵可以将图1的点转到图2的极线上，因此距离的计算是点到极线上的距离（的平方）
   *
   * @param F21 参考帧到当前帧的基础矩阵
   * @param vbMatchesInliers 匹配点是否为内点的标志向量[Output]
   * @param sigma 误差归一化参数
   * @return float 基础矩阵的得分
   */
  float CheckFundamental(const Eigen::Matrix3f &F21, std::vector<bool> &vbMatchesInliers, float sigma);

  // /**
  //  * @brief 重建基础矩阵F并三角化特征点
  //  *
  //  * @param[in] vbMatchesInliers 特征点匹配的内点标志
  //  * @param[in] F21 基础矩阵F
  //  * @param[in] K 相机内参矩阵
  //  * @param[out] T21 相机位姿变换矩阵。p2 = T21*p1 ; p1是参考帧中的点，p2是当前帧中的点。 F1可以理解为世界坐标系。
  //  * @param[out] vP3D 三维特征点坐标
  //  * @param[out] vbTriangulated 特征点三角化标志
  //  * @param[in] minParallax
  //  * 最小视差（单位：度）。如果三角化后的特征点视差（实现中是选择第50小的视差）小于该值，则认为三角化失败。默认1度
  //  * @param[in] minTriangulated 最小三角化数量。如果三角化后的特征点数量小于该值，则认为三角化失败。实际中是max(N*0.9,
  //  * minTriangulated). 默认50
  //  * @return bool 是否成功重建基础矩阵F并三角化特征点
  //  */
  // bool ReconstructF(std::vector<bool> &vbMatchesInliers, Eigen::Matrix3f &F21, cv::Mat &K, PoseT &T21,
  //                   std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated, float minParallax,
  //                   int minTriangulated);

  // /**
  //  * @brief 重建基础矩阵F并三角化特征点
  //  *
  //  * @param[in] vbMatchesInliers 特征点匹配的内点标志
  //  * @param[in] H21 基础矩阵F
  //  * @param[in] K 相机内参矩阵
  //  * @param[out] T21 相机位姿变换矩阵。p2 = T21*p1 ; p1是参考帧中的点，p2是当前帧中的点。 F1可以理解为世界坐标系。
  //  * @param[out] vP3D 三维特征点坐标
  //  * @param[out] vbTriangulated 特征点三角化标志
  //  * @param[in] minParallax
  //  * 最小视差（单位：度）。如果三角化后的特征点视差（实现中是选择第50小的视差）小于该值，则认为三角化失败。默认1度
  //  * @param[in] minTriangulated 最小三角化数量。如果三角化后的特征点数量小于该值，则认为三角化失败。实际中是max(N*0.9,
  //  * minTriangulated). 默认50
  //  * @return bool 是否成功重建基础矩阵F并三角化特征点
  //  */
  // bool ReconstructH(std::vector<bool> &vbMatchesInliers, Eigen::Matrix3f &H21, cv::Mat &K, PoseT &T21,
  //                   std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated, float minParallax,
  //                   int minTriangulated);

  /**
   * @brief 1. 从单应矩阵或者基础矩阵恢复R,t; 2. 三角化特征点
   * Step 1 : 从单应矩阵或者基础矩阵恢复R,t
   * 原始代码将单应矩阵和基础矩阵的恢复和三角化分开，并且分解的时候使用自己手写的代码分解。
   * 这里将单应矩阵和基础矩阵分解R,t的过程使用Opencv接口代替
   *
   * Step 2 : 使用恢复的所有R，t解进行特征点三角化，并选择最好的解
   * 这里主要调用CheckRT函数来进行三角化及计算内点数量
   *
   * Step 3 ： 选择三角化点最多的解作为最终的解
   * 这里还会根据最优解的3D点个数必须明显多于次优解（0.7的比例），否则也认为求解失败
   * 
   * Step 4 ： 保存最优结果
   *
   * @param[in] vbMatchesInliers 特征点匹配的内点标志
   * @param[in] HF21 单应矩阵H或者基础矩阵F
   * @param[in] K 相机内参矩阵
   * @param[out] T21 相机位姿变换矩阵。p2 = T21*p1 ; p1是参考帧中的点，p2是当前帧中的点。 F1可以理解为世界坐标系。
   * @param[out] vP3D 三维特征点坐标
   * @param[out] vbTriGood 特征点三角化标志
   * @param[in] isF 是否是基础矩阵
   * @param[in] minParallax
   * 最小视差（单位：度）。如果三角化后的特征点视差（实现中是选择第50小的视差）小于该值，则认为三角化失败。默认1度
   * @param[in] minTriangulated 最小三角化数量。如果三角化后的特征点数量小于该值，则认为三角化失败。实际中是max(N*0.9,
   * minTriangulated). 默认50
   * @return bool 是否成功重建基础矩阵F并三角化特征点
   */
  bool ReconstructHF(std::vector<bool> &vbMatchesInliers, Eigen::Matrix3f &HF21, Eigen::Matrix3f &K, PoseT &T21,
                     std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriGood, bool isF, float minParallax,
                     int minTriangulated);

  /**
   * @brief 通过给入的R,t三角化关键点，并计算内点数量
   * Step 1: 根据R,t;K计算相机1和相机2的投影矩阵（3x4）
   * Step 2：三角化点
   * Step 3：计算3D点到两个相机光心的夹角余弦值。过小的余弦值可能意味着很远的点，有可能会有负的深度值
   * Step 4：计算3D点在两个相机的深度，滤除深度值小于0的点.(只有当余弦值<0.99998才检查深度值)
   * Step 5：计算3D点在两个相机上的投影误差，滤除大于阈值th2的点
   * Step 6：统计所有内点的余弦值，选择第50大的点的余弦值（转换为角度）作为视差parallax返回
   *
   * @param[in] R21 相机的旋转矩阵
   * @param[in] t21 相机的平移向量
   * @param[in] vKeys1 第一帧图像的关键点
   * @param[in] vKeys2 第二帧图像的关键点
   * @param[in] vMatches12 匹配的关键点对
   * @param[in] vbMatchesInliers 关键点对是否有效的标志（从估计F、H的时候得出）
   * @param[in] K 相机的内参矩阵
   * @param[out] vP3D 三维点的容器
   * @param[in] th2 阈值（像素的平方）;重建的3D点投影到两个相机的误差平方必须小于该阈值。默认4.0
   * @param[out] vbTriangulated 有效的重建点标志
   * @param[out] parallax 视差（对重建点按照视差角排序，取第50个点的视差），单位：度
   * @return int 内点数量 可以通过内点数量来选择最优的R,t
   */
  int CheckRT(const cv::Mat &R21, const cv::Mat &t21, const std::vector<cv::KeyPoint> &vKeys1,
              const std::vector<cv::KeyPoint> &vKeys2, const std::vector<Match> &vMatches12,
              std::vector<bool> &vbMatchesInliers, const cv::Mat &K, std::vector<cv::Point3f> &vP3D, float th2,
              std::vector<bool> &vbTriangulated, float &parallax);

 private:
  // 参考帧的特征点
  std::vector<cv::KeyPoint> mvKeys1;
  // 当前帧的特征点
  std::vector<cv::KeyPoint> mvKeys2;

  // 参考帧到当前帧的关键点匹配
  std::vector<Match> mvMatches12;
  std::vector<bool> mvbMatched1;

  // 相机内参
  cv::Mat mK;

  // 误差归一化参数/平方
  float mSigma, mSigma2;

  // RANSAC最大迭代次数
  int mMaxIterations;

  std::vector<std::vector<size_t>> mvSets;
};

}  // namespace ORB_SLAM_Tracking
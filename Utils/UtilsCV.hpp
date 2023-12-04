#pragma once

#include <Eigen/Core>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

namespace ORB_SLAM_Tracking {

namespace UtilsCV {

/**
 * @brief 从两个位姿Tc12,Tc2w计算F12 \n
 * F12 = K1^{-T} * [t12]^R12 * K2^{-1}
 * [t12]^指指的是t12的反对称矩阵
 *
 * @param Tc1w 世界坐标系到Frame1坐标系的坐标变换矩阵
 * @param Tc2w 世界坐标系到Frame2坐标系的坐标变换矩阵
 * @param K1 相机1内参
 * @param K2 相机2内参
 * @return Eigen::Matrix3d
 */
Eigen::Matrix3d ComputeF21(const Eigen::Matrix4d& Tc1w, const Eigen::Matrix4d& Tc2w,
                           const Eigen::Matrix3d& K1, const Eigen::Matrix3d& K2);

// kp1^T * F12 * kp2 = 0
// 计算kp1在图像2下的极线，然后计算kp2到极线的距离
// 如果计算出错，返回-1
float EpipolarDistance12(const cv::KeyPoint& kp1, const cv::KeyPoint& kp2, const Eigen::Matrix3d& F12);

// kp1^T * F12 * kp2 = 0
// 计算kp2在图像1下的极线，然后计算kp1到极线的距离
// 如果计算出错，返回-1
float EpipolarDistance21(const cv::KeyPoint& kp1, const cv::KeyPoint& kp2, const Eigen::Matrix3d& F12);

void DecomposeE(const Eigen::Matrix3d& E, Eigen::Matrix3d& R1, Eigen::Matrix3d& R2, Eigen::Vector3d& t);
bool DecomposeE(const cv::Mat& E, cv::Mat& R1, cv::Mat& R2, cv::Mat& t);

// p2^{T} * F21 * p1 = 0
// Solve by SVD
bool ComputeF21(const std::vector<cv::Point2f>& vP1, const std::vector<cv::Point2f>& vP2, cv::Mat& F21);

void Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);


}  // namespace UtilsCV
}  // namespace ORB_SLAM_Tracking
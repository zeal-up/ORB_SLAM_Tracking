#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <Eigen/Core>
#include "SlamTypes/BasicTypes.hpp"
#include <g2o/types/slam3d/se3quat.h>

namespace ORB_SLAM_Tracking {

class Converter {
  public:
    // ---------------------- cv::Mat convertion -------------------------------
    /**
     * @brief Convert image to grayscale.
     *
     * @param inIm Input image. Can be one channel or three channels.
     * @param outImGray Output grayscale image.
     * @param bRGB If true, convert from RGB to grayscale. Otherwise, convert
     * from BGR to grayscale.
     * @return true if success.
     */
    static bool toGray(const cv::Mat& inIm, cv::Mat& outImGray, bool bRGB = false);
    
    // ---------------------- end cv::Mat convertion ---------------------------

    // ---------------------- to Eigen Matrix convertion------------------------

    /**
     * @brief 将cv::Mat转换为Eigen::Matrix3f
     *
     * @param cvMat3 输入的cv::Mat矩阵
     * @return 转换后的Eigen::Matrix3f矩阵
     */
    static Eigen::Matrix3f toMatrix3f(const cv::Mat& cvMat3);

    /**
     * @brief 将cv::Mat转换为Eigen::Vector3f
     *
     * @param cvMat3 输入的cv::Mat矩阵
     * @return 转换后的Eigen::Vector3f矩阵
     */
    static Eigen::Vector3f toVector3f(const cv::Mat& cvMat3);

    static Point3dT toPoint3dT(const cv::Point3f& cvPoint3f);

    static PoseT toPoseT(const g2o::SE3Quat& se3);

    // ---------------------- end to Eigen Matrix convertion--------------------

    // ---------------------- to cv::Mat ---------------------------------------
    /**
     * @brief 将Eigen::Matrix3f转换为cv::Mat
     *
     * @param eMat3 输入的Eigen::Matrix3f矩阵
     * @return 转换后的cv::Mat矩阵
     */
    static cv::Mat toCvMat3(const Eigen::Matrix3f& eMat3);

    /**
     * @brief 输入的descriptors的每一行是一个特征点的描述子，将其转换为vector<cv::Mat>类型
     *
     * @param[in] descriptors 输入的描述子矩阵
     * @return std::vector<cv::Mat> 转换描述子向量
     */
    static std::vector<cv::Mat> toDescriptorVector(const cv::Mat& descriptors);

    // ---------------------- end to cv::Mat -----------------------------------

    // ---------------------- to g2o types -----------------------------------

    /**
     * @brief 将位姿转换为g2o::SE3Quat
     *
     * @param pose 输入的位姿矩阵
     * @return 转换后的g2o::SE3Quat
     */
    static g2o::SE3Quat toSE3Quat(const PoseT& pose);

    // ---------------------- end to g2o types --------------------------------

};

}  // namespace ORB_SLAM_Tracking
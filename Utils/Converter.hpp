
#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <Eigen/Core>

namespace ORB_SLAM_Tracking {

class Converter {
  public:
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
    
    // to Eigen Matrix ----------------------------------------------------------

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

    // end to Eigen Matrix ------------------------------------------------------

    // to cv::Mat ---------------------------------------------------------------
    /**
     * @brief 将Eigen::Matrix3f转换为cv::Mat
     *
     * @param eMat3 输入的Eigen::Matrix3f矩阵
     * @return 转换后的cv::Mat矩阵
     */
    static cv::Mat toCvMat3(const Eigen::Matrix3f& eMat3);

    // end to cv::Mat -----------------------------------------------------------

};

}  // namespace ORB_SLAM_Tracking
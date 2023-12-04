#include "Utils/Converter.hpp"

namespace ORB_SLAM_Tracking {

bool Converter::toGray(const cv::Mat& inIm, cv::Mat& outImGray, bool bRGB) {
  if (inIm.channels() == 1) {
    inIm.copyTo(outImGray);
    return true;
  } else if (inIm.channels() == 3) {
    if (bRGB) {
      cv::cvtColor(inIm, outImGray, cv::COLOR_RGB2GRAY);
    } else {
      cv::cvtColor(inIm, outImGray, cv::COLOR_BGR2GRAY);
    }
    return true;
  }
  std::cerr << "ERROR: Wrong image format" << std::endl;
  return false;
}  // toGray function

// -------------------------- to Eigen Matrix -----------------------------

template <typename T>
void fillMatrix(const cv::Mat& cvMat3, Eigen::Matrix3f& M) {
  M << cvMat3.at<T>(0, 0), cvMat3.at<T>(0, 1), cvMat3.at<T>(0, 2), cvMat3.at<T>(1, 0),
      cvMat3.at<T>(1, 1), cvMat3.at<T>(1, 2), cvMat3.at<T>(2, 0), cvMat3.at<T>(2, 1),
      cvMat3.at<T>(2, 2);
}

Eigen::Matrix3f Converter::toMatrix3f(const cv::Mat& cvMat3) {
  Eigen::Matrix3f M;

  // Usage:
  if (cvMat3.depth() == CV_32F) {
    fillMatrix<float>(cvMat3, M);
  } else if (cvMat3.depth() == CV_64F) {
    fillMatrix<double>(cvMat3, M);
  }

  return M;
}  // toMatrix3f function

Eigen::Vector3f Converter::toVector3f(const cv::Mat& cvMat3) {
  Eigen::Vector3f v;

  v << cvMat3.at<float>(0, 0), cvMat3.at<float>(1, 0), cvMat3.at<float>(2, 0);

  return v;
} // toVector3f function


// -------------------------- end to Eigen Matrix -------------------------

// -------------------------- to cv::Mat ----------------------------------

cv::Mat Converter::toCvMat3(const Eigen::Matrix3f& eMat3) {
  cv::Mat cvMat3(3, 3, CV_32F);

  cvMat3.at<float>(0, 0) = eMat3(0, 0);
  cvMat3.at<float>(0, 1) = eMat3(0, 1);
  cvMat3.at<float>(0, 2) = eMat3(0, 2);
  cvMat3.at<float>(1, 0) = eMat3(1, 0);
  cvMat3.at<float>(1, 1) = eMat3(1, 1);
  cvMat3.at<float>(1, 2) = eMat3(1, 2);
  cvMat3.at<float>(2, 0) = eMat3(2, 0);
  cvMat3.at<float>(2, 1) = eMat3(2, 1);
  cvMat3.at<float>(2, 2) = eMat3(2, 2);

  return cvMat3;

} // toCvMat3 function

// ------------------------- end to cv::Mat ------------------------------


}  // namespace ORB_SLAM_Tracking
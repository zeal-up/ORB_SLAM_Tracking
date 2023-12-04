#include "Utils/UtilsCV.hpp"

namespace ORB_SLAM_Tracking {
namespace UtilsCV {

Eigen::Matrix3d ComputeF21(const Eigen::Matrix4d& Tc1w, const Eigen::Matrix4d& Tc2w,
                           const Eigen::Matrix3d& K1, const Eigen::Matrix3d& K2) {
  Eigen::Matrix4d T12 = Tc1w * Tc2w.inverse();

  Eigen::Matrix3d R12 = T12.block<3, 3>(0, 0);
  Eigen::Vector3d t12 = T12.block<3, 1>(0, 3);

  // get skew-symmetric matrix of t12
  Eigen::Matrix3d t12x;
  t12x << 0, -t12(2), t12(1), t12(2), 0, -t12(0), -t12(1), t12(0), 0;

  // F12 = K1^{-T} * [t12]^R12 * K2^{-1}
  return K1.transpose() * t12x * R12 * K2.inverse();
}  // ComputeF21

float EpipolarDistance12(const cv::KeyPoint& kp1, const cv::KeyPoint& kp2,
                       const Eigen::Matrix3d& F12) {
  const float a = kp1.pt.x * F12(0, 0) + kp1.pt.y * F12(1, 0) + F12(2, 0);
  const float b = kp1.pt.x * F12(0, 1) + kp1.pt.y * F12(1, 1) + F12(2, 1);
  const float c = kp1.pt.x * F12(0, 2) + kp1.pt.y * F12(1, 2) + F12(2, 2);

  const float num = a * kp2.pt.x + b * kp2.pt.y + c;
  const float den = a * a + b * b;
  if (den == 0) return -1;

  return num * num / den;
}

float EpipolarDistance21(const cv::KeyPoint& kp1, const cv::KeyPoint& kp2, const Eigen::Matrix3d& F12) {
  // kp1^T * F12 * kp2 = 0
  // => kp2^T * F12^T * kp1 = 0
  return EpipolarDistance12(kp2, kp1, F12.transpose());
}


void DecomposeE(const Eigen::Matrix3d& E, Eigen::Matrix3d& R1, Eigen::Matrix3d& R2, Eigen::Vector3d& t) {
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();

  Eigen::Matrix3d W;
  W << 0, -1, 0, 1, 0, 0, 0, 0, 1;

  R1 = U * W * V.transpose();
  if (R1.determinant() < 0) {
    R1 = -R1;
  }
  R2 = U * W.transpose() * V.transpose();
  if (R2.determinant() < 0) {
    R2 = -R2;
  }
  t = U.col(2);
  t = t / t.norm();

} // DecomposeE - eigen

bool DecomposeE(const cv::Mat& E, cv::Mat& R1, cv::Mat& R2, cv::Mat& t) {
  cv::Mat u,w,vt;
  cv::SVD::compute(E,w,u,vt);

  u.col(2).copyTo(t);
  t=t/cv::norm(t);

  cv::Mat W(3,3,CV_32F,cv::Scalar(0));
  W.at<float>(0,1)=-1;
  W.at<float>(1,0)=1;
  W.at<float>(2,2)=1;

  R1 = u*W*vt;
  if(cv::determinant(R1)<0)
      R1=-R1;

  R2 = u*W.t()*vt;
  if(cv::determinant(R2)<0)
      R2=-R2;

  return true;
}

bool ComputeF21(const std::vector<cv::Point2f>& vP1, const std::vector<cv::Point2f>& vP2, cv::Mat& F21) {
  const int N = vP1.size();

  cv::Mat A(N,9,CV_32F);

  for(int i=0; i<N; i++)
  {
      const float u1 = vP1[i].x;
      const float v1 = vP1[i].y;
      const float u2 = vP2[i].x;
      const float v2 = vP2[i].y;

      A.at<float>(i,0) = u2*u1;
      A.at<float>(i,1) = u2*v1;
      A.at<float>(i,2) = u2;
      A.at<float>(i,3) = v2*u1;
      A.at<float>(i,4) = v2*v1;
      A.at<float>(i,5) = v2;
      A.at<float>(i,6) = u1;
      A.at<float>(i,7) = v1;
      A.at<float>(i,8) = 1;
  }

  cv::Mat u,w,vt;

  cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

  cv::Mat Fpre = vt.row(8).reshape(0, 3);

  cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

  w.at<float>(2)=0;
  
  F21 = u*cv::Mat::diag(w)*vt;
  return true;
} // ComputeF21

void Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D) {
  cv::Mat A(4,4,CV_32F);

  A.row(0) = kp1.pt.x*P1.row(2)-P1.row(0);
  A.row(1) = kp1.pt.y*P1.row(2)-P1.row(1);
  A.row(2) = kp2.pt.x*P2.row(2)-P2.row(0);
  A.row(3) = kp2.pt.y*P2.row(2)-P2.row(1);

  cv::Mat u,w,vt;
  cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
  x3D = vt.row(3).t();
  x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
}

}  // namespace UtilsCV
}  // namespace ORB_SLAM_Tracking
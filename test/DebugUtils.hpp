#include <opencv2/opencv.hpp>

namespace DebugUtils {

// Save KeyPoints in such format : x y size angle response octave class_id
bool SaveKeyPoints(const std::string& filename, const std::vector<cv::KeyPoint>& keypoints);
bool ReadKeyPoints(const std::string& filename, std::vector<cv::KeyPoint>& keypoints);

// Save 3D points in such format : x y z
bool SavePoints3D(const std::string& filename, const std::vector<cv::Point3f>& points3d);
bool ReadPoints3D(const std::string& filename, std::vector<cv::Point3f>& points3d);

// Save Pose/Transformation matrix, which is 4x4 matrix
bool SavePoseMat(const std::string& filename, const cv::Mat& pose_mat);
// Read Pose/Transformation matrix, which is 4x4 matrix
bool ReadPoseMat(const std::string& filename, cv::Mat& pose_mat);

// Save camera intrinsic matrix, which is 3x3 matrix
bool SaveIntrinsicMat(const std::string& filename, const cv::Mat& intrinsic_mat);
bool ReadIntrinsicMat(const std::string& filename, cv::Mat& intrinsic_mat);

}	// namespace DebugUtils
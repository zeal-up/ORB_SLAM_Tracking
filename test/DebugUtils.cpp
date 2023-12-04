#include "DebugUtils.hpp"
#include <fstream>
#include <sstream>

namespace DebugUtils {

bool SaveKeyPoints(const std::string& filename, const std::vector<cv::KeyPoint>& keypoints) {
	std::ofstream ofs(filename, std::ios::out | std::ios::trunc);
	if (!ofs.is_open()) {
		return false;
	}
	ofs << "x y size angle response octave class_id" << std::endl;

	for (const auto& keypoint : keypoints) {
		ofs << keypoint.pt.x << " " << keypoint.pt.y << " " << keypoint.size << " " << keypoint.angle << " " << keypoint.response << " " << keypoint.octave << " " << keypoint.class_id << std::endl;
	}

	ofs.close();
	return true;
}	// SaveKeyPoints

bool ReadKeyPoints(const std::string& filename, std::vector<cv::KeyPoint>& keypoints) {
	std::ifstream ifs(filename);
	if (!ifs.is_open()) {
		return false;
	}
	std::stringstream ss;
	std::string line;
	std::getline(ifs, line);	// skip header
	while (std::getline(ifs, line)) {
		ss.clear();
		ss.str(line);
		cv::KeyPoint keypoint;
		ss >> keypoint.pt.x >> keypoint.pt.y >> keypoint.size >> keypoint.angle >> keypoint.response >> keypoint.octave >> keypoint.class_id;
		keypoints.push_back(keypoint);
	}

	ifs.close();
	return true;
}	// ReadKeyPoints

bool SavePoints3D(const std::string& filename, const std::vector<cv::Point3f>& points3d) {
	// open file with rewrite mode
	std::ofstream ofs(filename, std::ios::out | std::ios::trunc);
	if (!ofs.is_open()) {
		return false;
	}

	for (const auto& point3d : points3d) {
		ofs << point3d.x << " " << point3d.y << " " << point3d.z << std::endl;
	}

	ofs.close();
	return true;
}	// SavePoints3D

bool ReadPoints3D(const std::string& filename, std::vector<cv::Point3f>& points3d) {
	std::ifstream ifs(filename);
	if (!ifs.is_open()) {
		return false;
	}
	std::stringstream ss;
	std::string line;
	while (std::getline(ifs, line)) {
		ss.clear();
		ss.str(line);
		cv::Point3f point3d;
		ss >> point3d.x >> point3d.y >> point3d.z;
		points3d.push_back(point3d);
	}

	ifs.close();
	return true;
}	// ReadPoints3D

bool SavePoseMat(const std::string& filename, const cv::Mat& pose_mat) {
	std::ofstream ofs(filename, std::ios::out | std::ios::trunc);
	if (!ofs.is_open()) {
		return false;
	}

	for (int i = 0; i < pose_mat.rows; ++i) {
		for (int j = 0; j < pose_mat.cols; ++j) {
			ofs << pose_mat.at<float>(i, j) << " ";
		}
		ofs << std::endl;
	}

	ofs.close();
	return true;
}	// SavePoseMat

bool ReadPoseMat(const std::string& filename, cv::Mat& pose_mat) {
	std::ifstream ifs(filename);
	if (!ifs.is_open()) {
		return false;
	}
	std::stringstream ss;
	std::string line;
	while (std::getline(ifs, line)) {
		ss.clear();
		ss.str(line);
		std::vector<double> row;
		double value;
		while (ss >> value) {
			row.push_back(value);
		}
		if (row.size() != 4) {
			return false;
		}
		pose_mat.push_back(cv::Mat(row).t());
	}

	if (pose_mat.rows != 4 || pose_mat.cols != 4) {
		return false;
	}
	pose_mat.convertTo(pose_mat, CV_32F);
	ifs.close();
	return true;
}

bool SaveIntrinsicMat(const std::string& filename, const cv::Mat& intrinsic_mat) {
	std::ofstream ofs(filename, std::ios::out | std::ios::trunc);
	if (!ofs.is_open()) {
		return false;
	}
	if (intrinsic_mat.rows != 3 || intrinsic_mat.cols != 3) {
		return false;
	}

	for (int i = 0; i < intrinsic_mat.rows; ++i) {
		for (int j = 0; j < intrinsic_mat.cols; ++j) {
			ofs << intrinsic_mat.at<float>(i, j) << " ";
		}
		ofs << std::endl;
	}
	ofs.close();
	return true;
}
bool ReadIntrinsicMat(const std::string& filename, cv::Mat& intrinsic_mat) {
	std::ifstream ifs(filename);
	if (!ifs.is_open()) {
		return false;
	}
	std::stringstream ss;
	std::string line;
	while (std::getline(ifs, line)) {
		ss.clear();
		ss.str(line);
		std::vector<double> row;
		double value;
		while (ss >> value) {
			row.push_back(value);
		}
		if (row.size() != 3) {
			return false;
		}
		intrinsic_mat.push_back(cv::Mat(row).t());
	}

	if (intrinsic_mat.rows != 3 || intrinsic_mat.cols != 3) {
		return false;
	}
	intrinsic_mat.convertTo(intrinsic_mat, CV_32F);
	ifs.close();
	return true;
}

}	// namespace DebugUtils
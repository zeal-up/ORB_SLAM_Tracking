// Copyright 2023 <zeal-up>

#include <filesystem>
#include <iostream>
#include <vector>

#include "Features/ORBextractor.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

#define DRAW_RESULT

using namespace ORB_SLAM_Tracking;

int main() {
  std::string test_img_file = "./orbtest.png";
  cv::Mat img = cv::imread(test_img_file, cv::IMREAD_GRAYSCALE);
  cv::Size imgSize(img.cols, img.rows);

  ORBextractor orbExtractor(2000, 1.2, 8, 20, 7);

  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  std::vector<int> vLapping{0, 0};

  int ret = orbExtractor(img, cv::Mat(), keypoints, descriptors, vLapping);
  if (ret > 0) {
    std::cout << "Feature extraction successful!" << std::endl;

    std::vector<int> numFeaturesPerLevel = orbExtractor.GetNumFeaturesPerLevel();
    int nlevel = orbExtractor.GetLevels();
    int nfeatures = keypoints.size();

    int sumFeatureAllLevels = 0;
    for (int i = 0; i < nlevel; ++i) {
      sumFeatureAllLevels += numFeaturesPerLevel[i];
    }

    std::cout << "Number of features per level: ";
    for (int i = 0; i < nlevel; ++i) {
      std::cout << numFeaturesPerLevel[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Total number of features: " << nfeatures << std::endl;
    std::cout << "Sum of features in all levels: " << sumFeatureAllLevels << std::endl;

    int featureLength = 32;  // 32 * 8 = 256
    std::cout << "Descriptor rows: " << descriptors.rows << std::endl;
    std::cout << "Descriptor cols: " << descriptors.cols << std::endl;
    std::cout << "Descriptor type: " << descriptors.type() << std::endl;

#ifdef DRAW_RESULT
    cv::Mat imgWithKeypoint;
    cv::drawKeypoints(img, keypoints, imgWithKeypoint);
    // display the image
    cv::imshow("imgWithKeypoint", imgWithKeypoint);
    cv::waitKey(0);
#endif
  } else {
    std::cout << "Feature extraction failed!" << std::endl;
  }

  return 0;
}

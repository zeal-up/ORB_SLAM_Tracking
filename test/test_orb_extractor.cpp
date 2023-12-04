#include "Features/ORBextractor.hpp"
#include "org_orb_extractor_3/ORBextractor.h"

using namespace ORB_SLAM3;

uint64_t ORBdistance(const cv::Mat &a, const cv::Mat &b) {
  // Bit count function got from:
  // http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetKernighan
  // This implementation assumes that a.cols (CV_8U) % sizeof(uint64_t) == 0

  const uint64_t *pa, *pb;
  pa = a.ptr<uint64_t>();  // a & b are actually CV_8U
  pb = b.ptr<uint64_t>();

  uint64_t v, ret = 0;
  for (size_t i = 0; i < a.cols / sizeof(uint64_t); ++i, ++pa, ++pb) {
    v = *pa ^ *pb;
    v = v - ((v >> 1) & (uint64_t) ~(uint64_t)0 / 3);
    v = (v & (uint64_t) ~(uint64_t)0 / 15 * 3) + ((v >> 2) & (uint64_t) ~(uint64_t)0 / 15 * 3);
    v = (v + (v >> 4)) & (uint64_t) ~(uint64_t)0 / 255 * 15;
    ret += (uint64_t)(v * ((uint64_t) ~(uint64_t)0 / 255)) >> (sizeof(uint64_t) - 1) * CHAR_BIT;
  }

  return ret;

  // // If uint64_t is not defined in your system, you can try this
  // // portable approach (requires DUtils from DLib)
  // const unsigned char *pa, *pb;
  // pa = a.ptr<unsigned char>();
  // pb = b.ptr<unsigned char>();
  //
  // int ret = 0;
  // for(int i = 0; i < a.cols; ++i, ++pa, ++pb)
  // {
  //   ret += DUtils::LUT::ones8bits[ *pa ^ *pb ];
  // }
  //
  // return ret;
}

// 如果总体距离小于10, 则认为描述子相同
bool compareDesc(const cv::Mat &d1, const cv::Mat &d2) {
  if (ORBdistance(d1, d2) < 10) {
    return true;
  } else {
    return false;
  }
}

int main() {
  std::string test_img_file = "./init_01.png";
  cv::Mat img = cv::imread(test_img_file, cv::IMREAD_GRAYSCALE);

  ORBextractor orbExtractor(2000, 1.2, 8, 20, 7);

  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  std::vector<int> vLapping{0, 0};
  orbExtractor(img, cv::Mat(), keypoints, descriptors, vLapping);

  // for reimplementation
  std::vector<cv::KeyPoint> keypoints2;
  cv::Mat descriptors2;
  ORB_SLAM_Tracking::ORBextractor orbExtractor2(2000, 1.2, 8, 20, 7);
  std::vector<int> vLapping2{0, 0};
  bool ret = orbExtractor2(img, cv::Mat(), keypoints2, descriptors2, vLapping2);

  // for org implementation
  std::cout << "For org implementation:" << std::endl;
  std::cout << "Number of features: " << keypoints.size() << std::endl;
  std::cout << "Descriptor rows: " << descriptors.rows << std::endl;
  std::cout << "Descriptor cols: " << descriptors.cols << std::endl;
  std::cout << "Descriptor type: " << descriptors.type() << std::endl;

  // for reimplementation
  std::cout << "For reimplementation:" << std::endl;
  std::cout << "Number of features: " << keypoints2.size() << std::endl;
  std::cout << "Descriptor rows: " << descriptors2.rows << std::endl;
  std::cout << "Descriptor cols: " << descriptors2.cols << std::endl;
  std::cout << "Descriptor type: " << descriptors2.type() << std::endl;

  // compute the difference between org and reimplementation
  // for keypoints  ---------------------------------------------
  // 如果x,y,angle的差别都小于0.05pixel,认为是同一个点
  int N_org = keypoints.size();
  int N_re = keypoints2.size();
  // 计算召回率
  int N_recall = 0;
  for (int i = 0; i < N_org; ++i) {
    for (int j = 0; j < N_re; ++j) {
      if (std::abs(keypoints[i].pt.x - keypoints2[j].pt.x) < 0.05 &&
          std::abs(keypoints[i].pt.y - keypoints2[j].pt.y) < 0.05 &&
          std::abs(keypoints[i].angle - keypoints2[j].angle) < 0.05) {
        if (compareDesc(descriptors.row(i), descriptors2.row(j))) {
          N_recall++;
          break;
        }
      }
    }
  }
  // 计算准确率
  int N_correct = 0;
  for (int i = 0; i < N_re; ++i) {
    for (int j = 0; j < N_org; ++j) {
      if (std::abs(keypoints2[i].pt.x - keypoints[j].pt.x) < 0.05 &&
          std::abs(keypoints2[i].pt.y - keypoints[j].pt.y) < 0.05 &&
          std::abs(keypoints2[i].angle - keypoints[j].angle) < 0.05) {
        if (compareDesc(descriptors2.row(i), descriptors.row(j))) {
          N_correct++;
          break;
        }
      }
    }
  }
  std::cout << "Recall: " << N_recall * 1.0 / N_org << std::endl;
  std::cout << "Precision: " << N_correct * 1.0 / N_re << std::endl;

  return 0;
}
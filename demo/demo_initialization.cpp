#include <opencv2/opencv.hpp>

#include "Config/Settings.hpp"
#include "DUtils/DUtils.h"
#include "DUtilsCV/DUtilsCV.h"
#include "Features/ORBextractor.hpp"
#include "Features/ORBmatcher.hpp"
#include "Initialization/Initializer.hpp"
#include "SlamTypes/Frame.hpp"

using namespace ORB_SLAM_Tracking;

bool TryToInitialize(cv::Mat im1, cv::Mat im2, Settings& settings);
bool DISPLAY = false;

int main(int argc, char** argv) {
  // 从argv获取图片1和图片2的路径
  if (argc != 4) {
    std::cerr << "Usage: ./demo_initialization config_path image_dir skip_num" << std::endl;
    return 1;
  }
  std::string configPath = argv[1];
  std::string imageDir = argv[2];
  int skpiNum = std::stoi(argv[3]);

  

  // read all image names in the image directory
  std::vector<std::string> imageNames;
  imageNames = DUtils::FileFunctions::Dir(imageDir.c_str(), ".png", true);

  // 读取配置
  Settings settings(configPath);

  size_t nImages = imageNames.size();
  std::cout << "nImages: " << nImages << std::endl;

  size_t successIm1, successIm2;
  bool bTriSuccess = false;
  for (size_t i = 0; i < nImages - skpiNum; ++i) {
    // set random seed of rand() function
    srand(0);
    cv::Mat im1 = cv::imread(imageNames[i], cv::IMREAD_COLOR);
    cv::Mat im2 = cv::imread(imageNames[i+skpiNum], cv::IMREAD_COLOR);

    std::cout << "Trying to initialize with image: " << imageNames[i] << " and "
              << imageNames[i+skpiNum] << "----------------------------------" << std::endl;
    bTriSuccess = TryToInitialize(im1, im2, settings);
    successIm1 = i;
    successIm2 = i+skpiNum;
    if (bTriSuccess) {
      std::cout << "Initialization success!" << std::endl;
      break;
    }
  }
  
  if (bTriSuccess) DISPLAY = true;
  // set random seed of rand() function
  srand(0);
  cv::Mat im1Success = cv::imread(imageNames[successIm1], cv::IMREAD_COLOR);
  cv::Mat im2Success = cv::imread(imageNames[successIm2], cv::IMREAD_COLOR);
  TryToInitialize(im1Success, im2Success, settings);
}

bool TryToInitialize(cv::Mat im1, cv::Mat im2, Settings& settings) {
  cv::Mat im1Gray, im2Gray;
  cv::cvtColor(im1, im1Gray, cv::COLOR_BGR2GRAY);
  cv::cvtColor(im2, im2Gray, cv::COLOR_BGR2GRAY);

  // 构造ORBextractor 并提取特征点 ---------------------------------------------
  // 初始化时用2倍特征点
  ORBextractor orbExtractor(2 * settings.nFeatures, settings.scaleFactor, settings.nLevels,
                            settings.iniThFAST, settings.minThFAST);

  // 构造Frame
  Frame frame1(im1Gray, 0.0, &orbExtractor, nullptr, settings.mK, settings.mDistCoef);
  Frame frame2(im2Gray, 1.0, &orbExtractor, nullptr, settings.mK, settings.mDistCoef);

  // 打印特征点数量
  std::cout << "frame1.mvKeys.size(): " << frame1.mvKeys.size() << std::endl;
  std::cout << "frame2.mvKeys.size(): " << frame2.mvKeys.size() << std::endl;

  // 统计金字塔底层的特征点数量 —— 初始化时只用底层的特征点
  int n1 = 0, n2 = 0;
  for (auto& keypoint : frame1.mvKeys) {
    if (keypoint.octave == 0) n1++;
  }
  for (auto& keypoint : frame2.mvKeys) {
    if (keypoint.octave == 0) n2++;
  }
  std::cout << "The keypoints number in the finest layer of image1: " << n1 << std::endl;
  std::cout << "The keypoints number in the finest layer of image2: " << n2 << std::endl;

  // 使用DUtilsCV 可视化特征点  -------------------------------------------------
  cv::Mat im1WithFeatures = im1.clone();
  cv::Mat im2WithFeatures = im2.clone();
  DUtilsCV::Drawing::drawKeyPoints(im1WithFeatures, frame1.mvKeys);
  DUtilsCV::Drawing::drawKeyPoints(im2WithFeatures, frame2.mvKeys);
  cv::Mat im1im2;
  cv::hconcat(im1WithFeatures, im2WithFeatures, im1im2);
  DUtilsCV::GUI::tWinHandler win1("ImagesWithKeypoints");
  if (DISPLAY) DUtilsCV::GUI::showImage(im1im2, true, &win1, 0);

  // 构造ORBmatcher
  ORBmatcher orbMatcher(0.9, true);
  // 使用ORBmatcher进行匹配 -----------------------------------------------------
  std::vector<int> mvMatches;
  int nmatches = orbMatcher.SearchForInitialization(frame1, frame2, mvMatches, 100);
  std::cout << "nmatches: " << nmatches << std::endl;
  if (nmatches < 100) {
    std::cerr << "ERROR: Not enough matches." << std::endl;
    return false;
  }

  // 使用DUtilsCV 可视化匹配点  -------------------------------------------------
  std::vector<int> indices1, indices2;
  for (int i = 0; i < mvMatches.size(); i++) {
    if (mvMatches[i] >= 0) {
      indices1.push_back(i);
      indices2.push_back(mvMatches[i]);
    }
  }
  cv::Mat imageWithMatches;
  DUtilsCV::Drawing::drawCorrespondences(imageWithMatches, im1.clone(), im2.clone(), frame1.mvKeys,
                                         frame2.mvKeys, mvMatches);
  DUtilsCV::GUI::tWinHandler win2("ImageWithMatches");
  if (DISPLAY) DUtilsCV::GUI::showImage(imageWithMatches, true, &win2, 0);

  // 获取金字塔底层的特征点并可视化 ---------------------------------------------
  std::vector<cv::KeyPoint> kp1Level0, kp2Level0;
  for (size_t i = 0; i < frame1.mvKeys.size(); i++) {
    if (frame1.mvKeys[i].octave == 0) {
      kp1Level0.push_back(frame1.mvKeys[i]);
    }
  }
  for (size_t i = 0; i < frame2.mvKeys.size(); i++) {
    if (frame2.mvKeys[i].octave == 0) {
      kp2Level0.push_back(frame2.mvKeys[i]);
    }
  }
  cv::Mat im1WithFinestFeatures = im1.clone();
  cv::Mat im2WithFinestFeatures = im2.clone();
  DUtilsCV::Drawing::drawKeyPoints(im1WithFinestFeatures, kp1Level0);
  DUtilsCV::Drawing::drawKeyPoints(im2WithFinestFeatures, kp2Level0);
  cv::Mat im1im2WithFinestFeatures;
  DUtilsCV::Drawing::drawCorrespondences(im1im2WithFinestFeatures, im1WithFinestFeatures,
                                         im2WithFinestFeatures, frame1.mvKeys, frame2.mvKeys,
                                         mvMatches);
  DUtilsCV::GUI::tWinHandler win4("ImagesWithFinestKeypoints");
  if (DISPLAY) DUtilsCV::GUI::showImage(im1im2WithFinestFeatures, true, &win4, 0);

  // 构造Initializer 并进行初始化 ------------------------------------------------
  Initializer initializer(frame1, 1.0, 2000);
  PoseT T21;
  std::vector<bool> vbTriangulated;
  std::vector<cv::Point3f> vP3D;
  DUtils::Profiler profiler(DUtils::Profiler::MS);
  profiler.profile("Triangulation");
  bool isTriangulated = initializer.Initialize(frame2, mvMatches, T21, vP3D, vbTriangulated);
  profiler.stop("Triangulation");
  double tTriangulation = profiler.getMeanTime("Triangulation");
  if (!isTriangulated) {
    std::cerr << "ERROR: Triangulation failed." << std::endl;
    return false;
  } else {
    std::cout << "Triangulation success! Cost time : " << tTriangulation << " ms" << std::endl;
  }

  // 使用DUtilsCV 可视化三角化出来的3D点
  int nTriangulated = 0;
  std::vector<int> mvMatchTriangulated = mvMatches;
  for (size_t i = 0; i < mvMatches.size(); i++) {
    if (mvMatches[i] >= 0 && !vbTriangulated[i]) {
      mvMatchTriangulated[i] = -1;
    }
    if (vbTriangulated[i]) nTriangulated++;
  }
  std::cout << "nTriangulated: " << nTriangulated << std::endl;
  cv::Mat imageWithTriangulatedPoints;
  DUtilsCV::Drawing::drawCorrespondences(imageWithTriangulatedPoints, im1WithFinestFeatures,
                                         im2WithFinestFeatures, frame1.mvKeysUn, frame2.mvKeys, mvMatchTriangulated);

  DUtilsCV::GUI::tWinHandler win3("ImageWithTriangulatedPoints");
  if (DISPLAY) DUtilsCV::GUI::showImage(imageWithTriangulatedPoints, true, &win3, 0);

  return true;
}

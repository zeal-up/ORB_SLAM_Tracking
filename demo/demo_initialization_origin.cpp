#include <Eigen/Core>
#include <Eigen/LU>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <glog/logging.h>
#include "Config/Settings.hpp"
#include "DUtils/DUtils.h"
#include "DUtilsCV/DUtilsCV.h"


#include "Initialization/Initializer2.h"
#include "Features/ORBextractor2.h"
#include "Features/ORBmatcher2.h"
#include "SlamTypes/Frame2.h"

// using namespace ORB_SLAM_Tracking;

const bool USING_ORG_INITIALIZER = true;
const bool USING_ORG_EXTRACTOR = true;
const bool USING_ORG_MATCHER = true;
#define USING_ORG_FRAME 1
#ifdef USING_ORG_FRAME
typedef ORB_SLAM2::Frame Frame;
typedef std::shared_ptr<ORB_SLAM2::Frame> FramePtr;
using ORB_SLAM2::ORBextractor;
#endif

// using ORB_SLAM_Tracking::ORBextractor;
using ORB_SLAM_Tracking::Settings;
using ORB_SLAM_Tracking::PoseT;


enum class FAILED_REASON {
  FIRST_FRAME,
  NOT_ENOUGH_MATCHES,
  TRIANGULATION_FAILED,
  SUCCESS
};
class InitilizerManager {
  public:
    cv::Mat im1, im2;
    // FramePtr frame1Ptr, frame2Ptr;
    Frame frame1, frame2;
    std::vector<cv::Point2f> mvbPrevMatched;
    bool isIm1Set = false;
    ORBextractor* orbExtractor = nullptr;
    Settings settings;

    // drawing 
    cv::Mat im1WithFeatures = cv::Mat(), im2WithFeatures = cv::Mat(), im1im2WithFeatures = cv::Mat();
    cv::Mat imageWithMatches = cv::Mat();
    cv::Mat im1im2WithFinestFeatures = cv::Mat();
    cv::Mat imageWithTriangulatedPoints = cv::Mat();

  public:
    InitilizerManager(Settings _settings):settings(_settings) {
      // 构造ORBextractor 并提取特征点 ---------------------------------------------
      // 初始化时用2倍特征点
      if (USING_ORG_EXTRACTOR) {
        orbExtractor = new ORB_SLAM2::ORBextractor(2*settings.nFeatures, settings.scaleFactor, settings.nLevels,
                        settings.iniThFAST, settings.minThFAST);
      } else {
        // orbExtractor = new ORBextractor(2 * settings.nFeatures, settings.scaleFactor, settings.nLevels,
        //                 settings.iniThFAST, settings.minThFAST);
      }
    } // constructor

    void DrawResult() {
      DUtilsCV::GUI::tWinHandler win1("ImagesWithKeypoints");
      DUtilsCV::GUI::showImage(im1im2WithFeatures, true, &win1, 0);
      DUtilsCV::GUI::tWinHandler win2("ImageWithMatches");
      DUtilsCV::GUI::showImage(imageWithMatches, true, &win2, 0);
      DUtilsCV::GUI::tWinHandler win4("ImagesWithFinestKeypoints");
      DUtilsCV::GUI::showImage(im1im2WithFinestFeatures, true, &win4, 0);
      DUtilsCV::GUI::tWinHandler win3("ImageWithTriangulatedPoints");
      DUtilsCV::GUI::showImage(imageWithTriangulatedPoints, true, &win3, 0);
    }
    FAILED_REASON TryToInitialize(cv::Mat im) {
      
      if (!isIm1Set) {
        cv::Mat im1Gray;
        im1 = im.clone();
        cv::cvtColor(im, im1Gray, cv::COLOR_BGR2GRAY);
        // 构造Frame
        frame1 = Frame(im1Gray, 0.0, orbExtractor, nullptr, settings.mK, settings.mDistCoef, 0, 0);
        // 打印特征点数量
        LOG(INFO) << "frame1.mvKeys.size(): " << frame1.mvKeysUn.size();
        mvbPrevMatched.clear();
        mvbPrevMatched.reserve(frame1.mvKeys.size());
        for (size_t i1 = 0; i1 < frame1.mvKeysUn.size(); i1++) {
          mvbPrevMatched.push_back(frame1.mvKeysUn[i1].pt);
        }
        isIm1Set = true;
        return FAILED_REASON::FIRST_FRAME;
      }
      im2 = im.clone();
      cv::Mat im2Gray;
      cv::cvtColor(im, im2Gray, cv::COLOR_BGR2GRAY);
      frame2 = Frame(im2Gray, 1.0, orbExtractor, nullptr, settings.mK, settings.mDistCoef,0 ,0);
      
      LOG(INFO) << "frame2.mvKeys.size(): " << frame2.mvKeys.size();

      // 统计金字塔底层的特征点数量 —— 初始化时只用底层的特征点
      int n1 = 0, n2 = 0;
      for (auto& keypoint : frame1.mvKeys) {
        if (keypoint.octave == 0) n1++;
      }
      for (auto& keypoint : frame2.mvKeys) {
        if (keypoint.octave == 0) n2++;
      }
      LOG(INFO) << "The keypoints number in the finest layer of image1: " << n1;
      LOG(INFO) << "The keypoints number in the finest layer of image2: " << n2;

      // 使用DUtilsCV 可视化特征点  -------------------------------------------------
      im1WithFeatures = im1.clone();
      im2WithFeatures = im2.clone();
      DUtilsCV::Drawing::drawKeyPoints(im1WithFeatures, frame1.mvKeys);
      DUtilsCV::Drawing::drawKeyPoints(im2WithFeatures, frame2.mvKeys);
      cv::hconcat(im1WithFeatures, im2WithFeatures, im1im2WithFeatures);

      // 使用ORBmatcher进行匹配 -----------------------------------------------------
      std::vector<int> mvMatches;
      int nmatches;
      // 构造ORBmatcher
      if (USING_ORG_MATCHER) {
        LOG(INFO) << "Using org matcher.";
        ORB_SLAM2::ORBmatcher orbMatcher(0.9, true);
        nmatches = orbMatcher.SearchForInitialization(frame1, frame2, mvbPrevMatched, mvMatches, 100);
      } else {
        // ORBmatcher orbMatcher(0.9, true);
        LOG(INFO) << "Using new matcher.";
        // nmatches = orbMatcher.SearchForInitialization(frame1, frame2, mvbPrevMatched, mvMatches, 100);
      }
      
      LOG(INFO) << "nmatches: " << nmatches;
      if (nmatches < 100) {
        LOG(ERROR)<< "ERROR: Not enough matches. Reset Frame1";
        isIm1Set = false;
        return FAILED_REASON::NOT_ENOUGH_MATCHES;
      }

      // 使用DUtilsCV 可视化匹配点  -------------------------------------------------
      std::vector<int> indices1, indices2;
      for (int i = 0; i < mvMatches.size(); i++) {
        if (mvMatches[i] >= 0) {
          indices1.push_back(i);
          indices2.push_back(mvMatches[i]);
        }
      }
      imageWithMatches = im1.clone();
      DUtilsCV::Drawing::drawCorrespondences(imageWithMatches, im1.clone(), im2.clone(), frame1.mvKeys,
                                            frame2.mvKeys, mvMatches);
      

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
      DUtilsCV::Drawing::drawCorrespondences(im1im2WithFinestFeatures, im1WithFinestFeatures,
                                            im2WithFinestFeatures, frame1.mvKeys, frame2.mvKeys,
                                            mvMatches);
      

      // 构造Initializer 并进行初始化 ------------------------------------------------
      std::vector<bool> vbTriangulated;
      std::vector<cv::Point3f> vP3D;
      PoseT T21;
      bool isTriangulated = false;

      DUtils::Profiler profiler(DUtils::Profiler::MS);
      profiler.profile("Triangulation");

      if (USING_ORG_INITIALIZER) {
        LOG(INFO) << "Using org initializer.";
        cv::Mat R21 = cv::Mat::eye(3, 3, CV_32F);
        cv::Mat t21 = cv::Mat::zeros(3, 1, CV_32F);
        ORB_SLAM2::Initializer initializer = ORB_SLAM2::Initializer(frame1, 1.0, 200);
        isTriangulated = initializer.Initialize(frame2, mvMatches, R21, t21, vP3D, vbTriangulated);
        if (isTriangulated) {
          Eigen::MatrixXd R21_eigen, t21_eigen;
          cv::cv2eigen(R21, R21_eigen);
          cv::cv2eigen(t21, t21_eigen);
          T21.matrix().block<3, 3>(0, 0) = R21_eigen;
          T21.matrix().block<3, 1>(0, 3) = t21_eigen;
        }
      } else {
        LOG(INFO) << "Using new initializer.";
        // Initializer initializer(frame1, 1.0, 200);
        // isTriangulated = initializer.Initialize(frame2, mvMatches, T21, vP3D, vbTriangulated);
      }

      profiler.stop("Triangulation");
      double tTriangulation = profiler.getMeanTime("Triangulation");

      if (!isTriangulated) {
        LOG(ERROR) << "Triangulation failed. Keep Frame1";
        return FAILED_REASON::TRIANGULATION_FAILED;
      }
      // cal triangulated points number
      int nTriangulated = 0;
      for (size_t i = 0; i < vbTriangulated.size(); i++) {
        if (vbTriangulated[i]) nTriangulated++;
      }
      if (nTriangulated < 100) {
        LOG(ERROR)<< "ERROR: Triangulation get too few 3d point, get " << nTriangulated << " triangulated points. Keep Frame1";
        return FAILED_REASON::TRIANGULATION_FAILED;
      } else {
        LOG(INFO) << "Triangulation success! Cost time : " << tTriangulation << " ms";
      }

      // 使用DUtilsCV 可视化三角化出来的3D点 -----------------------------------------
      std::vector<int> mvMatchTriangulated = mvMatches;
      for (size_t i = 0; i < mvMatches.size(); i++) {
        if (mvMatches[i] >= 0 && !vbTriangulated[i]) {
          mvMatchTriangulated[i] = -1;
        }
      }
      LOG(INFO) << "nTriangulated: " << nTriangulated;
      DUtilsCV::Drawing::drawCorrespondences(imageWithTriangulatedPoints, im1,
                                            im2, frame1.mvKeysUn, frame2.mvKeysUn, mvMatchTriangulated);

      // save the triangulated points
      std::ofstream triangulatedPointsFile("triangulatedPoints.txt");
      for (size_t i = 0; i < vP3D.size(); i++) {
        if (vbTriangulated[i])
          triangulatedPointsFile << vP3D[i].x << "," << vP3D[i].y << "," << vP3D[i].z;
      }

      return FAILED_REASON::SUCCESS;
    }

};


int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;
  // 从argv获取图片目录
  if (argc != 3) {
    LOG(ERROR)<< "Usage: ./demo_initialization config_path image_dir";
    return 1;
  }
  std::string configPath = argv[1];
  std::string imageDir = argv[2];  

  // read all image names in the image directory
  std::vector<std::string> imageNames;
  imageNames = DUtils::FileFunctions::Dir(imageDir.c_str(), ".png", true);

  // 读取配置
  Settings settings(configPath);
  InitilizerManager initilizerManager(settings);

  size_t nImages = imageNames.size();
  LOG(INFO) << "nImages: " << nImages;

  size_t preImIdx = 0;
  size_t imIdx = 0;
  bool bTriSuccess = false;
  while (imIdx < nImages) {
    if (!initilizerManager.isIm1Set) {
      preImIdx = imIdx;
    }
    cv::Mat im = cv::imread(imageNames[imIdx], cv::IMREAD_COLOR);
    LOG(INFO) << "Trying to initialize with image: " << imageNames[preImIdx] << " and "
              << imageNames[imIdx] << "----------------------------------";
    FAILED_REASON reason = initilizerManager.TryToInitialize(im);
    if (reason == FAILED_REASON::FIRST_FRAME) {
    } else if (reason == FAILED_REASON::NOT_ENOUGH_MATCHES) {
      LOG(ERROR) << "Initialization failed! Reason: NOT_ENOUGH_MATCHES";
    } else if (reason == FAILED_REASON::TRIANGULATION_FAILED) {
      LOG(ERROR) << "Initialization failed! Reason: TRIANGULATION_FAILED";
    } else {
      LOG(INFO) << "Initialization success!";
      bTriSuccess = true;
      break;
    }
    imIdx++;
  }
  
  if (bTriSuccess) {
    initilizerManager.DrawResult();
  } else {
    LOG(INFO) << "Initialization failed!";
  }
}
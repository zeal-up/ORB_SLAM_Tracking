#include "tracking.hpp"

#include "Features/ORBmatcher.hpp"
#include "Utils/Converter.hpp"


namespace ORB_SLAM_Tracking {

Tracking::Tracking(ORBVocabulary* pVoc, Map* pMap, Settings* pSettings,
                   Drawer* pDrawer)
    : mState(NO_IMAGES_YET),
      mpVocabulary(pVoc),
      mpMap(pMap),
      mpSettings(pSettings),
      mpDrawer(pDrawer) {
  // initialize the ORBextractor
  mpORBextractor = new ORBextractor(
      mpSettings->nFeatures, mpSettings->scaleFactor, mpSettings->nLevels,
      mpSettings->iniThFAST, mpSettings->minThFAST);
  // during initialization, double features in the first two frames
  mpIniORBextractor = new ORBextractor(
      2 * mpSettings->nFeatures, mpSettings->scaleFactor, mpSettings->nLevels,
      mpSettings->iniThFAST, mpSettings->minThFAST);

  mVelocity = PoseT::Identity();
}

Tracking::~Tracking() {
  delete mpORBextractor;
  delete mpIniORBextractor;
}

PoseT Tracking::GrabImage(const cv::Mat& im, const double& timestamp) {

  // Convert image to grayscale
  cv::Mat imGray;
  bool bOk = Converter::toGray(im, imGray, mpSettings->bRGB);
  if (!bOk) {
    std::cerr << "ERROR: Cannot convert image to grayscale." << std::endl;
    exit(EXIT_FAILURE);
  }

  // Create Frame object
  if (mState == WORKING || mState == LOST) {
    // normal tracking - use mpORBextractor
    mCurrentFrame = Frame(imGray, timestamp, mpORBextractor, mpVocabulary,
                          mpSettings->mK, mpSettings->mDistCoef);
  } else {
    // initialization - use mpIniORBextractor - double feature points number
    mInitialFrame = Frame(imGray, timestamp, mpIniORBextractor, mpVocabulary,
                          mpSettings->mK, mpSettings->mDistCoef);
  }

  if (mState == NO_IMAGES_YET) {
    mState = NOT_INITIALIZED;
  }

  if (mState == NOT_INITIALIZED) {
    bOk = DealFirstFrame();
    if (bOk) {
      mState = INITIALIZING;
    }
  } else if (mState == INITIALIZING) {
    bOk = Initialize();
    if (bOk) {
      mState = WORKING;
    } else {
      // 当第二帧的特征点太少或者两帧匹配太少时，需要重新初始化
      mState = NOT_INITIALIZED;
    }
  } else {
    // normal tracking
  }


}

bool Tracking::DealFirstFrame() {
  // 初始化时需要确保特征点的数量足够多
  if (mCurrentFrame.mvKeys.size() < 100) {
    return false;
  } else {
    mInitialFrame = Frame(mCurrentFrame);
    mLastFrame = Frame(mCurrentFrame);
    if (mpInitializer) {
      delete mpInitializer;
    }
    mpInitializer = new Initializer(mCurrentFrame, 1.0, 200);
  }
  return true;
}

bool Tracking::Initialize() {
  if (mCurrentFrame.mvKeys.size() < 100) {
    std::cerr << "ERROR: Too few features in new frame." << std::endl;
    return false;
  }

  // Step 1 ------------------------------------------------------------
  // 在第一帧和第二帧之间进行关键点匹配，得到初始的特征点匹配
  ORBmatcher matcher(0.9, true);
  int nmatches = matcher.SearchForInitialization(mInitialFrame, mCurrentFrame, mvIniMatches, 100);

  // 如果匹配点太少，则认为初始化失败 —— 会导致重置第一帧并重新初始化
  if (nmatches < 100) {
    std::cerr << "ERROR: Too few matches in the initialization." << std::endl;
    return false;
  }

  // Step 2 ------------------------------------------------------------
  // 通过H模型或者F模型来估计两帧之间的运动，并使用成功的模型来三角化匹配点

  PoseT Tcw;
  std::vector<bool> vbTriangulated;
  bool isTriangulated = mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Tcw, mvIniP3D, vbTriangulated);



}

}  // namespace ORB_SLAM_Tracking
#include "tracking.hpp"

#include "Features/ORBmatcher.hpp"
#include "Utils/Converter.hpp"
#include "SlamTypes/KeyFrame.hpp"
#include "SlamTypes/MapPoint.hpp"
#include "Optimization/Optimizer.hpp"
#include "SlamTypes/BuildLocalMap.hpp"

#include <glog/logging.h>


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

  mVelocity = std::make_unique<PoseT>(PoseT::Identity());

  // 初始化localmapper
  mpLocalMapper = new LocalMapping(mpMap);
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
    mCurrentFrame = std::make_shared<Frame>(imGray, timestamp, mpORBextractor, mpVocabulary,
                          mpSettings->mK, mpSettings->mDistCoef);
  } else {
    // initialization - use mpIniORBextractor - double feature points number
    mCurrentFrame = std::make_shared<Frame>(imGray, timestamp, mpIniORBextractor, mpVocabulary,
                          mpSettings->mK, mpSettings->mDistCoef);
  }

  if (mState == NO_IMAGES_YET) {
    mState = INITIALIZING;
  }

  if (mState == INITIALIZING) {
    bOk = Initialize();
    // 如果初始化成功，则将mState置为WORKING, 否则会继续尝试初始化
    if (bOk) {
      mState = WORKING;
    }
  } else if (mState == WORKING) {
    // 正常跟踪
    // TrackReferenceKeyFrame()函数会将当前帧与上一帧关键帧进行匹配，得到匹配点对
    // 并根据匹配点对计算当前帧的位姿
    // org : if(!mbMotionModel || mpMap->KeyFramesInMap()<4 || mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2)
    if (mpMap->KeyFramesInMap() < 4 || !mVelocity ) {
      bOk = TrackReferenceKeyFrame();
    } else {
      bOk = TrackWithMotionModel();
      // 无法通过运动模型跟踪到当前帧，则使用TrackPreviousKeyFrame()函数重新尝试跟踪
      if (!bOk) {
        bOk = TrackReferenceKeyFrame();
      }
    }

    // 如果初始跟踪成功，继续用LocalMap进行位姿优化
    if (bOk) {
      bOk = TrackLocalMap();
    }
  } else if (mState == LOST) {
    // 丢失跟踪
  }

  if (bOk) {
    mState = WORKING;
  } else {
    mState = LOST;
  }

  if (mState == LOST) {
    // TODO : 处理丢失跟踪的情况
    LOG(ERROR) << "Tracking lost";
    Reset();
    return PoseT::Identity();
  }

  // 跟踪正常， 插入关键帧
  if (mState == WORKING) {
    // 首先更新恒速模型
    UpdateVelocity();

    // 更新当前帧的跟踪地图点标志位
    for (int i = 0; i < mCurrentFrame->N; ++i) {
      MapPoint* pMP = mCurrentFrame->mvpMapPoints[i];
      if (pMP) {
        if (pMP->Observations() < 1) {
          mCurrentFrame->mvpMapPoints[i] = nullptr;
          mCurrentFrame->mvbOutlier[i] = false;
        }
      }
    } // 更新当前帧的跟踪地图点

    // 检查是否需要插入关键帧然后执行关键帧创建及执行
    if (NeedNewKeyFrame()) {
      CreateNewKeyFrame();
    } else {
      LOG(INFO) << "No need to create new keyframe.";
    }

    // 上面没有执行mCurrentFrame地图点outlier的剔除，这些地图点都会被用来创建关键帧
    // 后续在LocalMapping中会在BA之后根据chi2 test进行outlier剔除
    // 但是Tracking线程需要在这里进行outlier剔除，因为我们不希望下一帧使用这些outlier地图点
    for (int i = 0; i < mCurrentFrame->N; ++i) {
      if (mCurrentFrame->mvpMapPoints[i] && mCurrentFrame->mvbOutlier[i]) {
        mCurrentFrame->mvpMapPoints[i] = nullptr;
      }
    }

    // 更新最后一帧
    mLastFrame = std::make_shared<Frame>(*mCurrentFrame);
  } // 跟踪正常，插入关键帧
  else {
    LOG(ERROR) << "Wrong tracking state";
    exit(EXIT_FAILURE);
  }

}


bool Tracking::Initialize() {

  static bool bHasFirstFrame = false;

  // 第一次接收到图像帧;或者由于初始化失败，需要重置第一帧
  if (!bHasFirstFrame) {
    // 第一帧的特征点太少，直接返回，后续会重置第一帧
    if (mCurrentFrame->N < 100) return false;
    mInitialFrame = std::make_shared<Frame>(mCurrentFrame);
    mLastFrame = std::make_shared<Frame>(mCurrentFrame);
    if (mpInitializer) {
      delete mpInitializer;
    }
    mpInitializer = new Initializer(*mCurrentFrame, 1.0, 200);
    bHasFirstFrame = true;
    return false; // 初始化需要两帧，返回false，外界会继续调用此函数进行初始化
  } else {        // 第二帧来了，可以进行三角化了

    // 第二帧的特征点太少，重置第一帧
    if (mCurrentFrame->mvKeys.size() < 100) {
      LOG(WARNING) << "Initialization Error: Too few features(<100) in second frame.";
      bHasFirstFrame = false;
      return false;
    }

    // Step 1 ------------------------------------------------------------
    // 在第一帧和第二帧之间进行关键点匹配，得到初始的特征点匹配
    ORBmatcher matcher(0.9, true);
    int nmatches = matcher.SearchForInitialization(*mInitialFrame, *mCurrentFrame, mvIniMatches, 100);

    // 如果匹配点太少，则认为初始化失败 —— 会导致重置第一帧并重新初始化
    if (nmatches < 100) {
      LOG(WARNING) << "Initialization Error: Too few matches(<100) in the init && current frame";
      bHasFirstFrame = false;
      return false;
    }

    // Step 2 ------------------------------------------------------------
    // 通过H模型或者F模型来估计两帧之间的运动，并使用成功的模型来三角化匹配点

    PoseT Tcw;
    std::vector<bool> vbTriangulated;
    bool isTriangulated =
        mpInitializer->Initialize(*mCurrentFrame, mvIniMatches, Tcw, mvIniP3D, vbTriangulated);
    // 三角化失败，只重新获取第二帧
    if (!isTriangulated) {
      LOG(WARNING) << "Initialization Error: Triangulation failed. Only re-receive the second frame.";
      return false;
    }

    // Step 3 ------------三角化成功，将三角化出来的3D点添加到地图中----------------

    // 将iniMatches中成功三角化的点置位
    for (size_t i = 0; i < mvIniMatches.size(); i++) {
      if (mvIniMatches[i] >= 0 && !vbTriangulated[i]) {
        mvIniMatches[i] = -1;
        nmatches--;
      }
    }

    // 初始化的第一帧作为世界坐标系，因此其位姿为单位矩阵
    mInitialFrame->SetPose(PoseT::Identity());
    // 当前帧的位姿为Tcw：世界坐标系到当前帧的变换，世界坐标系也就是第一帧，因此这里也就是T21
    mCurrentFrame->SetPose(Tcw);

    // 地图初始化
    bool bOK = CreateInitialMap();
    return true;
  } // end of else : if(!bHasFirstFrame)-else {}
} // end of Initialize()

bool Tracking::CreateInitialMap() {
  // 创建关键帧 初始化的第一二帧均是关键帧
  KeyFrame* pKFini = new KeyFrame(*mInitialFrame, mpMap);
  KeyFrame* pKFcur = new KeyFrame(*mCurrentFrame, mpMap);

  // ------------------------ Step 1 为第一帧和第二帧创建BoW向量 ------------------------
  pKFini->ComputeBoW();
  pKFcur->ComputeBoW();

  // ------------------------ Step 2 将关键帧插入地图 ------------------------
  mpMap->AddKeyFrame(pKFini);
  mpMap->AddKeyFrame(pKFcur);

  // ------------------------ Step 3 将三角化出来的3D点与关键帧关联并插入地图 ------------------------
  for (size_t i = 0; i < mvIniMatches.size(); ++i) {
    if (mvIniMatches[i] < 0) continue;
    // 三角化出来的3D点
    Point3dT worldPos = Converter::toPoint3dT(mvIniP3D[i]);
    // 三角化出来的3D点对应的特征点
    int idx = mvIniMatches[i];
    // 创建地图点
    MapPoint* pMP = new MapPoint(worldPos, pKFcur, mpMap);
    // 将地图点添加到关键帧中
    pKFini->AddMapPoint(pMP, i);
    pKFcur->AddMapPoint(pMP, idx);
    // 为地图点添加观测帧
    pMP->AddObservation(pKFini, i);
    pMP->AddObservation(pKFcur, idx);

    // 更新地图点的描述子和平均观测方向以及最大最小观测深度
    pMP->ComputeDistinctiveDescriptors();
    pMP->UpdateNormalAndDepth();

    // 设置当前普通帧的地图点
    mCurrentFrame->mvpMapPoints[idx] = pMP;

    // 将地图点添加到地图中
    mpMap->AddMapPoint(pMP);
  }

  // ------------------------ Step 4 更新关键帧之间的连接关系 ------------------------
  pKFini->UpdateConnections();
  pKFcur->UpdateConnections();
  LOG(INFO) << "Initial Map created with (before global BA) " << mpMap->MapPointsInMap() << " points";

  // ------------------------ Step 5 全局BA优化 ------------------------
  Optimizer::GlobalBundleAdjustment(mpMap, 20);

  // ------------------------ Step 6 计算地图点平均深度并进行Scale 归一化 ------------------------
  float medianDepth = pKFini->ComputeSceneMedianDepth();  // 初始帧的地图点平均深度
  float invMedianDepth = 1.0f / medianDepth;

  if (medianDepth < 0 || pKFcur->TrackedMapPoints() < 100) {
    LOG(ERROR) << "Wrong initialization, medianDepth = " << medianDepth
               << ", nMapPoints = " << pKFcur->TrackedMapPoints();
    Reset();
    return false;
  }

  PoseT Tc2w = pKFcur->GetPose(); // Tc2w: 世界坐标系到第二帧的变换
  Tc2w.translation() *= invMedianDepth; // 归一化前两帧之间的位移
  pKFcur->SetPose(Tc2w); // 更新第二帧的位姿

  // 同时需要将地图点也进行归一化
  std::vector<MapPoint*> vpAllMapPoints = mpMap->GetAllMapPoints();
  for (auto pMP : vpAllMapPoints) {
    if (!pMP) continue;
    pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
  }

  mCurrentFrame->SetPose(Tc2w); // 更新当前帧的位姿
  mLastFrame = std::make_shared<Frame>(*mCurrentFrame); // 更新最后一帧
  mnLastKFframeId = mCurrentFrame->GetId();
  mpLastKeyFrame.reset(pKFcur); // 更新最后一个关键帧
  mpReferenceKF.reset(pKFcur); // 更新参考关键帧
  // TODO : add keyframe to TrackLocalMap
  // TODO : TrackLocalMap set local map points

  return true;

} // end of CreateInitialMap()

bool Tracking::TrackReferenceKeyFrame() {
  LOG(INFO) << "Enter " << __func__;

  // Step 1 计算BoW向量，计算的过程中会生成DirectIndex,可以用来加速特征匹配 ------
  mCurrentFrame->ComputeBoW();

  // Step 2 通过BoW向量进行特征匹配 --------------------------------------------
  ORBmatcher matcher(0.7, true);
  std::vector<MapPoint*> vpMapPointMatches;
  int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);

  if (nmatches < 15) return false;

  // Step 3 通过3D地图点-2D特征点匹配关系优化位姿 --------------------------------
  mCurrentFrame->mvpMapPoints = vpMapPointMatches;  // 设置匹配上的地图点
  mCurrentFrame->SetPose(mLastFrame->GetPose());    // 设置当前帧的位姿为上一帧的位姿

  Optimizer::PoseOptimization(mCurrentFrame); // 优化当前帧的位姿

  // Step 4 去掉优化期间被判定为outliers的点（chi2 test） ------------------------
  int nmatchesMap = 0;
  for (int i = 0; i < mCurrentFrame->N; i++) {
    if (mCurrentFrame->mvpMapPoints[i]) {
      if (mCurrentFrame->mvbOutlier[i]) {
        MapPoint* pMP = mCurrentFrame->mvpMapPoints[i];
        mCurrentFrame->mvpMapPoints[i] = nullptr;
        mCurrentFrame->mvbOutlier[i] = false;
        pMP->mnLastFrameSeen = mCurrentFrame->GetId();
      } else if (mCurrentFrame->mvpMapPoints[i]->Observations() > 0) {
        nmatchesMap++;
      }
    }
  } // end of for(int i=0; i<mCurrentFrame->N; i++)

  return nmatchesMap >= 10;

} // end of TrackReferenceKeyFrame()

bool Tracking::TrackWithMotionModel() {
  ORBmatcher matcher(0.9, true);

  // Step 1 通过运动模型预测当前帧的位姿 ------------------------------------------
  mCurrentFrame->SetPose(*mVelocity * mLastFrame->GetPose());
  fill(mCurrentFrame->mvpMapPoints.begin(), mCurrentFrame->mvpMapPoints.end(), static_cast<MapPoint*>(nullptr));

  // Step 2 将前一帧的地图点投影到当前帧进行匹配 ------------------------------------
  int th = 15;
  int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th);

  // Step 3 如果匹配数量不够，增大搜索半径再次匹配 ----------------------------------
  if (nmatches < 20) {
    fill(mCurrentFrame->mvpMapPoints.begin(), mCurrentFrame->mvpMapPoints.end(), static_cast<MapPoint*>(nullptr));
    nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * th);
  }

  if (nmatches < 20) return false;

  // Step 4 通过3D地图点-2D特征点匹配关系优化位姿 ------------------------------------
  Optimizer::PoseOptimization(mCurrentFrame);

  // Step 5 去掉优化期间被判定为outliers的点（chi2 test） ----------------------------
  int nmatchesMap = 0;
  for (int i = 0; i < mCurrentFrame->N; i++) {
    if (!mCurrentFrame->mvpMapPoints[i]) continue;
    if (mCurrentFrame->mvbOutlier[i]) {
      MapPoint* pMP = mCurrentFrame->mvpMapPoints[i];
      mCurrentFrame->mvpMapPoints[i] = nullptr;
      mCurrentFrame->mvbOutlier[i] = false;
      pMP->mnLastFrameSeen = mCurrentFrame->GetId();
    } else if (mCurrentFrame->mvpMapPoints[i]->Observations() > 0) {
      nmatchesMap++;
    }
  }

  return nmatchesMap >= 10;
}

bool Tracking::TrackLocalMap() {
  // Step 1 构建局部地图并进行匹配搜索 ----------------------------------------------
  std::vector<MapPoint*> vpLocalMapPoints;
  bool bLocalSearchOK = SearchLocalPoints(vpLocalMapPoints);
  if (!bLocalSearchOK) {
    LOG(ERROR) << "Local map matching failed.";
    return false;
  }

  // Step 1.1 将地图点的Visible数量增加1 ---------------------------------------------
  // TODO 这里做了一次地图点isInFrustum操作，但是在SearchByProjection里面也做了一次，是一种冗余
  // 原代码中在MapPoint中增加投影后的u,v坐标缓存，但是总感觉那样不太优美，这里损失了效率
  for (auto localMP : vpLocalMapPoints) {
    if (!localMP) continue;
    if (localMP->isBad()) continue;
    if (mCurrentFrame->isInFrustum(localMP, 0.5)) {
      localMP->IncreaseVisible();
    }
  }
  for (auto frameMP : mCurrentFrame->mvpMapPoints) {
    if (!frameMP) continue;
    if (frameMP->isBad()) continue;
    if (std::find(vpLocalMapPoints.begin(), vpLocalMapPoints.end(), frameMP) == vpLocalMapPoints.end()) {
      frameMP->IncreaseVisible();
    }
  }

  // Step 2 通过3D地图点-2D特征点匹配关系优化位姿 ------------------------------------
  Optimizer::PoseOptimization(mCurrentFrame);

  // Step 2.1 将BA后仍旧保持在Frame中且不是outlier的地图点的Keep数量增加 1 -------------
  for (int i = 0; i < mCurrentFrame->N; ++i) {
    if (!mCurrentFrame->mvpMapPoints[i]) continue;
    if (!mCurrentFrame->mvbOutlier[i]) {
      mCurrentFrame->mvpMapPoints[i]->IncreaseKeepAfterBA();
    }
  }
  
  // Step 3 确定跟踪状态 -----------------------------------------------------------
  // More restrictive if there was a relocalization recently
  // if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
  //     return false;

  // 有效的跟踪点数量需要大于30（去掉了outliers）
  return mCurrentFrame->GetValidMapPointCount() > 30;
}

bool Tracking::SearchLocalPoints(std::vector<MapPoint*>& vpLocalMapPoints) {

  // Step 1 构建局部地图 ----------------------------------------------------------
  KeyFramePtr bestCovisibleKF;
  BuildLocalMap().BuildLocalMapPoints(mCurrentFrame, vpLocalMapPoints, bestCovisibleKF);

  // Step 2 通过投影进行搜索匹配 ---------------------------------------------------
  ORBmatcher matcher(0.8, true);
  int expandR = 1; // 这里的搜索在代码里被设置为4.0 * expandR
  int nmatches = matcher.SearchByProjection(mCurrentFrame, vpLocalMapPoints, expandR);
  if (nmatches < 10) {
    LOG(ERROR) << "Local map matching less than 10 points. got " << nmatches;
    return false;
  }
  LOG(INFO) << "Local map matching " << nmatches << " points";
  return true;
}

void Tracking::UpdateVelocity() {
  LOG(INFO) << "Enter " << __func__;
  if (!mLastFrame->mbPoseSet || !mCurrentFrame->mbPoseSet) {
    LOG(ERROR) << "Last frame or current frame pose not set. Set velocity to nullptr";
    mVelocity.reset(nullptr);
  }
  mVelocity = std::make_unique<PoseT>(mCurrentFrame->GetPose() * mLastFrame->GetPose().inverse());
}

bool Tracking::NeedNewKeyFrame() {
  // TODO : 完整版本这里还会检查LocalMapping是否被暂停，以及是否刚刚重定位完
  // 由于这个项目只是做Tracking，所以不考虑localmapping和重定位

  // Step 1 计算参考关键帧的跟踪地图点数量 ------------------------------------------
  const int nKFs = mpMap->KeyFramesInMap();
  int nMinObs = 3; // 只统计大于nMinObs观测数的地图点
  if (nKFs <= 2) nMinObs = 2;
  int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

  // Step 2 计算需要关键帧的多个指标 ------------------------------------------------
  const float thRefRatio = 0.9;

  // 条件c1是关于帧数的判断，只要帧数过去太多就进行创建
  // 条件 1a : 距离上一次关键帧创建过去了minFrames帧，默认为0
  const bool c1 = mCurrentFrame->GetId() >= mnLastKFframeId + mpSettings->minFrames;
  // 条件c2是关于跟踪点数的判断，跟踪点数过少则说明跟踪较弱，需要创建关键帧，但又必须满足点数大于阈值
  const int curValidTrackPoints = mCurrentFrame->GetValidMapPointCount();
  const bool c2 = ((curValidTrackPoints < thRefRatio * nRefMatches) && curValidTrackPoints > 15);

  // Step 3 判断是否需要创建关键帧 ---------------------------------------------------
  if (c1 && c2) {
    LOG(INFO) << "Need New KeyFrame, both frame count straint and valid track points straint are satisfied.";
    return true;
  } else {
    LOG(INFO) << "No Need New KeyFrame. Frame cout straint : " << c1 << ", valid track points straint : " << c2;
    return false;
  } 

}

void Tracking::CreateNewKeyFrame() {

  // Step 1 创建关键帧 --------------------------------------------------------------
  KeyFramePtr pKF = std::make_shared<KeyFrame>(*mCurrentFrame, mpMap);
  mpReferenceKF = pKF; // 更新参考关键帧
  mnLastKFframeId = mCurrentFrame->GetId(); // 更新最后一个关键帧的帧ID
  mpLastKeyFrame = pKF; // 更新最后一个关键帧
  LOG(INFO) << "Create new keyframe, id = " << pKF->GetId();

  // Step 2 添加新的关键帧到LocalMapping中 --------------------------------------------
  LOG(INFO) << "Add new keyframe to local mapping";
  mpLocalMapper->AddKeyFrame(pKF);
  return;
}

void Tracking::Reset() {
  LOG(INFO) << "Resetting tracking system";
  {
    std::scoped_lock<std::mutex> lock(mMutexReset);
    mbReseting = true;
  }
  mpMap->Clear();
  KeyFrame::Reset();
  Frame::Reset();
  mState = NO_IMAGES_YET;
  {
    std::scoped_lock<std::mutex> lock(mMutexReset);
    mbReseting = false;
  }
}

}  // namespace ORB_SLAM_Tracking
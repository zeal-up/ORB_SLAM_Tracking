/**
 * This file is part of ORB_SLAM_Tracking
 *
 * Copyright (C) 2023- zeal-up
 *
 * A project for illustrating the tracking thread of ORB-SLAM series.
 * This project remove all the unnecessary code and remain only the tracking
 * part. Most contents are copied from ORB-SLAM/ORB-SLAM2/ORB-SLAM3, which is
 * available under GPLv3 license.
 *
 * ORB_SLAM_Tracking is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * You should have received a copy of the GNU General Public License along with
 * ORB_SLAM_Tracking. If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <opencv2/opencv.hpp>

#include "Config/Settings.hpp"
#include "Features/ORBVocabulary.hpp"
#include "Features/ORBextractor.hpp"
#include "SlamTypes/BasicTypes.hpp"
#include "SlamTypes/Frame.hpp"
#include "SlamTypes/Map.hpp"
#include "Initialization/Initializer.hpp"
#include "Modules/LocalMapping.hpp"

namespace ORB_SLAM_Tracking {

class Drawer;

/**
 * @class Tracking
 * @brief Class for tracking the camera pose in ORB-SLAM.
 */
class Tracking {
 public:
  /**
   * @brief Construct a new Tracking object
   *
   * @param pVoc ORBVocabulary pointer. Used for BoW vector generation.
   * @param pMap Pointer to the Map object. Map store keyframe and map points.
   * @param pSettings Pointer to the Settings object. Config file is loaded
   * before to Settings object.
   * @param pDrawer Pointer to the Drawer object.
   */
  Tracking(ORBVocabulary* pVoc, Map* pMap, Settings* pSettings,
           Drawer* pDrawer);

  /**
   * @enum eTrackingState
   * @brief Enumeration for tracking states.
   */
  enum eTrackingState {
    SYSTEM_NOT_READY = -1,
    NO_IMAGES_YET = 0,
    INITIALIZING = 1,     // initialization (At least two frames is needed)
    WORKING = 2,          // after initialization and tracking well
    LOST = 3
  };

  /**
   * @brief Destructor of the Tracking class.
   */
  ~Tracking();

 public:
  /**
   * @brief Main tracking procedure.
   *
   * @param im Input image.
   * @param timestamp Timestamp of the image.
   * @return PoseT Estimated camera pose.
   */
  PoseT GrabImage(const cv::Mat& im, const double& timestamp);

  private:

    /**
     * @brief 处理到达的图像帧，并尝试进行初始化，如果初始化成功，则返回true，
     * 返回true时，外界需要将mState设置为WORKING，否则外界持续调用此函数进行初始化尝试
     * 当前帧被存储到mCurrentFrame中
     * 初始化需要两帧，第一帧为mInitialFrame，第二帧为mCurrentFrame
     * 当第二帧的特征点数量小于100或者第一帧与第二帧的匹配数量小于100时，直接重新获取第一帧
     * 如果三角化失败，则只重新获取第二帧 
     * 
     * @return true 如果初始化成功返回true, 否则返回false
     */
    bool Initialize();

    // 重置跟踪线程：清空地图、重置ID
    void Reset();

    /**
     * @brief 将成功进行初始化的第一帧mInitialFrame与第二帧mCurrentFrame所三角化的3D点
     * 添加到地图中，并设置一些变量。mInitialFrame与mCurrentFrame的位姿都已经提前设置
     * 1. 将mInitialFrame与mCurrentFrame所三角化的3D点添加到地图中
     * 2. 执行GlobalBundleAdjustment()，优化地图点和关键帧的位姿
     * 3. 进行尺度归一化
     * 
     * @return true 如果成功进行地图初始化，则返回true
     */
    bool CreateInitialMap();

    // 从参考关键帧通过词袋模型寻找3D-2D匹配点
    // 当前帧位姿初始化为上一帧位姿
    // 通过匹配关系优化当前帧位姿
    bool TrackReferenceKeyFrame();
    bool TrackWithMotionModel();
    bool TrackLocalMap();

 private:
  // 构建局部地图点，然后进行匹配搜索. 搜索得到大于10个额外的匹配点，返回true
  bool SearchLocalPoints(std::vector<MapPoint*>& vpLocalMapPoints);

  // 通过mLastFrame与mCurrentFrame的位姿，更新恒速模型参数
  void UpdateVelocity();

  // Keyframe相关 ---------------------------------
  // 计算是否需要新的关键帧
  bool NeedNewKeyFrame();
  // 创建新的关键帧，添加到LocalMapping中
  void CreateNewKeyFrame();
  // 从当前关键帧和上一关键帧创建新的地图点
  bool CreateNewMapPoints();

 private:
  eTrackingState mState;       // Current tracking state
  ORBVocabulary* mpVocabulary; // ORB Vocabulary
  Map* mpMap;                  // Map object
  Settings* mpSettings;        // Settings object
  Drawer* mpDrawer;            // Drawer object
  LocalMapping* mpLocalMapper; // Local Mapping object
  

  // ORBextractor
  ORBextractor* mpORBextractor;  // ORB特征提取器 -- 正常跟踪时使用
  ORBextractor* mpIniORBextractor;  // ORB特征提取器 -- 初始化时使用

  std::unique_ptr<PoseT> mVelocity; // relative pose between t-1 and t-2 frame, used for
                   // TrackWithMotionModel

  // 帧缓存 -----------------------------
  FramePtr mCurrentFrame; // 当前帧
  FramePtr mInitialFrame; // 第一帧，也叫初始帧
  FramePtr mLastFrame;    // 上一帧
  KeyFramePtr mpLastKeyFrame; // 上一关键帧
  KeyFramePtr mpReferenceKF;  // 参考关键帧
  uint64_t mnLastKFframeId = 0; // 上一关键帧的帧ID,不是关键帧ID,是其对应的帧的ID


  // 初始化 -----------------------------
  Initializer* mpInitializer; // 初始化器
  // 初始帧与当前帧的匹配关系
  // mvIniMatches[i] = j 表示初始帧的第i个特征点与当前帧的第j个特征点匹配
  // 如果没有匹配，则mvIniMatches[i] = -1
  std::vector<int> mvIniMatches;
  // 初始化三角化出来的3D点
  std::vector<cv::Point3f> mvIniP3D;

  // 锁 ---------------------------------
  std::mutex mMutexReset; // 重置锁
  bool mbReseting = false; // 是否正在重置

};
}  // namespace ORB_SLAM_Tracking

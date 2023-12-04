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
#include "Initialization/Initializer.hpp"

namespace ORB_SLAM_Tracking {

class Map;
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
    NOT_INITIALIZED = 1,  // before receiving the first image
    INITIALIZING = 2,     // after receiving the first image
    WORKING = 3,          // after initialization (from the third frame)
    LOST = 4
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
     * @brief 处理到达的第一帧，需要有两帧才能进行初始化
     * 第一帧基本上只是存储起来
     * 
     * @return true 如果处理成功返回true, 否则返回false
     */
    bool DealFirstFrame();

    /**
     * @brief 处理到达的第二帧，进行初始化
     * 
     * @return true 如果初始化成功返回true, 否则返回false
     */
    bool Initialize();

 private:
  eTrackingState mState;       // Current tracking state
  ORBVocabulary* mpVocabulary; // ORB Vocabulary
  Map* mpMap;                  // Map object
  Settings* mpSettings;        // Settings object
  Drawer* mpDrawer;            // Drawer object

  // ORBextractor
  ORBextractor* mpORBextractor;  // ORB特征提取器 -- 正常跟踪时使用
  ORBextractor* mpIniORBextractor;  // ORB特征提取器 -- 初始化时使用

  PoseT mVelocity; // relative pose between t-1 and t-2 frame, used for
                   // TrackWithMotionModel

  // 帧缓存 -----------------------------
  Frame mCurrentFrame; // 当前帧
  Frame mInitialFrame; // 第一帧，也叫初始帧
  Frame mLastFrame;    // 上一帧

  // 初始化 -----------------------------
  Initializer* mpInitializer; // 初始化器
  // 初始帧与当前帧的匹配关系
  // mvIniMatches[i] = j 表示初始帧的第i个特征点与当前帧的第j个特征点匹配
  // 如果没有匹配，则mvIniMatches[i] = -1
  std::vector<int> mvIniMatches;
  // 初始化三角化出来的3D点
  std::vector<cv::Point3f> mvIniP3D;

};
}  // namespace ORB_SLAM_Tracking

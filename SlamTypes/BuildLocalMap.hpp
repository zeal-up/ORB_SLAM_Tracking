#pragma once
/**
 * @brief 将Tracking线程中局部地图的构建单独拿出来作为静态函数
 * Tracking中的局部地图构建只与当前帧有关，所有的共视关系和共视帧都可以通过当前帧的mvpMapPoints来获取 
 */

#include "SlamTypes/Frame.hpp"
#include "SlamTypes/KeyFrame.hpp"
#include "SlamTypes/MapPoint.hpp"

namespace ORB_SLAM_Tracking {

class BuildLocalMap {
 public:
  BuildLocalMap();
  BuildLocalMap(const BuildLocalMap&) = delete;
  BuildLocalMap& operator=(const BuildLocalMap&) = delete;
  

  // 通过当前帧，及其中包含的地图点，根据共视关系构建局部地图
  bool BuildLocalMapPoints(const FramePtr curF, std::vector<MapPoint*>& vpLocalMapPoints, KeyFramePtr& bestCovisibleKF);

 private:
  /**
   * @brief 根据当前帧获取共视关系，从共视帧和相邻帧构建局部地图的KeyFrames
   * 
   * @param curF 
   * @param vpLocalKeyFrames 
   * @param pKFmax 与curF共视点数量最多的关键帧
   * @return true 
   * @return false 
   */
  bool UpdateLocalKeyFrames(const FramePtr curF, std::vector<KeyFramePtr>& vpLocalKeyFrames, KeyFramePtr& pKFmax);
};
} // namespace ORB_SLAM_Tracking
#pragma once

#include "SlamTypes/Map.hpp"
#include "SlamTypes/MapPoint.hpp"

namespace ORB_SLAM_Tracking {

class Optimizer {
 public:
  /**
   * @brief 全局BA;Map中所有地图点和所有关键帧做Bundle Adjustment
   * 这个全局BA在下面的情况会被用到
   * 1. 单目初始化时：第一帧和第二帧创建地图后进行两帧的全局优化
   *
   * @param pMap 地图指针
   * @param nIterations 迭代次数
   * @param pbStopFlag 停止标志
   */
  void static GlobalBundleAdjustment(Map* pMap, const int& nIterations = 5,
                                     bool* pbStopFlag = nullptr);

  /**
   * @brief add all keyframes and map points as vertices; only pose-landmark edges
   * 对送入的所有关键帧和地图点进行优化，没有Fixed的帧和地图点。只有“位姿-地图点”的边，没有“位姿-位姿”的边。
   * 因为这里没有回环检测的帧。
   * 
   * ！！！这里构建完优化图之后直接执行Optimize进行优化。而在LocalBundleAdjustment中有更复杂的优化策略
   *
   * @param vpKFs
   * @param vpMPs
   * @param nIterations
   * @param pbStopFlag
   */
  void static BundleAdjustment(const std::vector<KeyFrame*>& vpKFs,
                               const std::vector<MapPoint*>& vpMPs, const int& nIterations = 5,
                               bool* pbStopFlag = nullptr);

  /**
   * @brief 在LocalMapping中对新加入的帧进行局部BA
   * 帧节点：当前帧以及与当前帧相连的关键帧
   * 地图点节点：当前帧能观测到的地图点
   * 固定帧节点：可以观测到地图点节点，但是不是当前帧的相连关键帧
   * 
   * 优化策略：
   * 1. 先执行5次迭代优化，然后删除chi2>5.991的边以及地图点在Camera下depth小于0的边
   * 2. 继续执行10次迭代优化，然后记录chi2>5.991的边所连接的关键帧和地图点，将两者的观测关系删除
   * 
   * @param pK 
   * @param pMap 
   */
  void static LocalBundleAdjustment(KeyFramePtr pK, Map* pMap);

  /**
   * @brief 使用帧pFrame中3D地图点-2D特征点的对应关系进行位姿优化，类似与PnP求解。
   *
   * @param pFrame
   * @return int 将地图点投影到当前帧的inlier数量
   */
  int static PoseOptimization(FramePtr pFrame);
};
}  // namespace ORB_SLAM_Tracking
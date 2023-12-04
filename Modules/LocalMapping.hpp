/**
 * @file LocalMapping.hpp
 * @author your name (you@domain.com)
 * @brief 与原始的LocalMapping不同，这里只是提供一个类来进行新的地图点生成。这里的LocalMapping
 * 会被Tracking线程当作一个工具类调用，不会作为一个线程来运行
 * @version 0.1
 * @date 2024-01-10
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#include <list>

#include "SlamTypes/Map.hpp"
#include "SlamTypes/KeyFrame.hpp"


namespace ORB_SLAM_Tracking {

class LocalMapping {
 public:
  LocalMapping(Map* pMap);

  /**
   * @brief 进行关键帧的插入、新建地图点、BA优化
   * 这里直接进行的是阻塞操作，只有当操作完成才返回结果
   * 
   * @param pKF 
   * @return true 
   * @return false 
   */
  bool AddKeyFrame(KeyFramePtr pKF);

 private:
  /**
   * @brief Step 1 进行新关键帧的各种预处理 \n 
   * 1.1 计算BoW向量 \n
   * 1.2 将当前关键帧加入到地图点的观测帧中 \n
   * 1.3 更新地图点的观测方向和深度 \n
   * 1.4 计算地图点的描述子 \n
   * 1.5 更新当前关键帧的共视图关系 \n
   * 1.6 将当前关键帧加入到地图中 \n 
   * 1.7 如果是第一帧的关键帧，会将地图点加入mlpRecentAddedMapPoints中 \n
   */
  void ProcessNewKeyFrame();
  /**
   * @brief Step 2 对近期添加进地图的地图点进行筛选，需要满足下面的条件才能被保留 \n
   * 1. 地图点被连续3个关键帧观测到 \n
   * 2. 地图点被投影搜索+BA后仍旧保留的数量 / 地图点被观测到的总次数 > 0.25 \n
   * 
   */
  void MapPointCulling();
  
  /**
   * @brief Step 3 使用对极几何创建新的地图点 \n
   * 不单单使用上一关键帧来生成地图点，而是根据当前关键帧的共视图关系，找出前20个共视关键帧，然后
   * 使用这些关键帧来进行地图点的生成
   * 
   */
  void CreateNewMapPoints();

  /**
   * @brief Step 4 地图点冗余去除
   * 在当前帧及第一第二共视帧中，进行地图点互相搜索，去除冗余地图点
   * 
   */
  void SearchInNeighbors();

  // Step 5 局部BA优化，这里主要是调用Optimizer::LocalBundleAdjustment()函数进行BA优化
  void OptimizeLocalMap();


  /**
   * @brief Step 6 删除多余关键帧（只对当前帧的共视帧进行检查）
   * 如果一个关键帧的90%地图点都能够被至少3个关键帧（在相同或者更低层级的特征金字塔）观测到，
   * 则认为这个关键帧是多余的
   * 
   * 实际实现时，就是遍历KF的所有地图点，逐一统计这些地图点的观测数量（跳过当前帧的观测），如果大于
   * 3，则累计加1,如果累计数量大于KF地图点的90%，则认为这个关键帧是多余的
   *  
   */
  void KeyFrameCulling();


 private:
  Map* mpMap;
  KeyFramePtr mpCurrentKF;

  // 用于存储最近添加的地图点
  // 经过筛选后，这些新增的地图点从这个list中移除。因为有频繁的随机插入删除操作，所以使用list
  std::list<MapPoint*> mlpRecentAddedMapPoints;

};

} // namespace ORB_SLAM_Tracking
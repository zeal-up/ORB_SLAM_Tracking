#pragma once

#include "SlamTypes/KeyFrame.hpp"

namespace ORB_SLAM_Tracking {

class Map {
  public:
    Map();
    void Clear();

    /**
     * @brief 添加地图点
     * 1. 检查mnMaxKfId
     * 2. 插入关键帧
     * 3. 设置地图更新标记位mbMapUpdated
     *
     * @param pMP 地图点指针
     */
    void AddKeyFrame(KeyFrame* pKF);
    // 获取所有关键帧
    std::vector<KeyFrame*> GetAllKeyFrames();
    // 从地图中删除关键帧
    void EraseKeyFrame(KeyFrame* pKF);

    /**
     * @brief 添加地图点
     * 1. 插入地图点
     * 2. 设置地图更新标记位mbMapUpdated
     *
     * @param pMP 地图点指针
     */
    void AddMapPoint(MapPoint* pMP);
    // 从地图中删除地图点
    void EraseMapPoint(MapPoint* pMP);
    std::vector<MapPoint*> GetAllMapPoints();

    // 获取地图点数量
    int MapPointsInMap();
    // 获取关键帧数量
    int KeyFramesInMap();

  private:
    // 地图中的地图点
    std::set<MapPoint*> mspMapPoints;
    // 地图中的关键帧
    std::set<KeyFrame*> mspKeyFrames;

    // 最大的关键帧ID
    long unsigned int mnMaxKfId;

    // 互斥锁
    std::mutex mMutexMap;

    // 更新标记 —— 设置为true说明地图被更新
    bool mbMapUpdated;
};
} // namespace ORB_SLAM_Tracking
#pragma once

#include <map>

#include "SlamTypes/BasicTypes.hpp"
#include "SlamTypes/KeyFrame.hpp"

namespace ORB_SLAM_Tracking {

class KeyFrame;
class Map;

class MapPoint {
 public:
  MapPoint() = delete;

  /**
   * @brief 构造函数
   *
   * @param Pos 三维坐标
   * @param pRefKF 参考关键帧。地图点会将一个关键帧设置为参考帧
   * @param pMap 地图
   */
  MapPoint(const Point3dT& Pos, KeyFrame* pRefKF, Map* pMap);

  // 获取地图点的三维坐标
  Point3dT GetWorldPos();

  // 设置地图点的三维坐标
  void SetWorldPos(const Point3dT& Pos);

  // 获取地图点的参考关键帧
  KeyFrame* GetReferenceKeyFrame();

  // 删除观测帧记录，从观测帧中删除此地图点，从地图中删除此地图点
  void SetBadFlag();
  bool isBad();

  // 替换地图点
  void Replace(MapPoint* pMP);
  MapPoint* GetReplaced();

  // -------------- 观测相关 --------------

  // 为地图点添加观测帧
  void AddObservation(KeyFrame* pKF, const size_t& idx);
  // 当前地图点的观测中删除关键帧pKF
  void EraseObservation(KeyFrame* pKF);
  /**
   * @brief 获取地图点的观测关键帧
   *
   * @return 一个map, key是观测关键帧，value是这个地图点在该关键帧的关键点索引
   */
  std::map<KeyFrame*, size_t> GetObservations();
  // 获取观测的数量
  int Observations();
  // 此地图点是否已经添加到关键帧pKF中
  bool IsKeyFrameInObservations(KeyFramePtr pKF);

  // 此地图点在新的一帧（普通帧）中可见（在视野范围内、观测角度限制内），则新增1
  void IncreaseVisible(int n = 1);
  // 此地图点在位姿优化之后，还在当前帧中，且没有在优化期间没有被置为Outlier，则新增1
  void IncreaseKeepAfterBA(int n = 1);
  // 获取此地图点 KeepAfterBA / Visible 的比值
  float GetFoundKeepRatio();

  

  // 更新观测向量和观测的深度值
  // 将所有观测帧的观测向量求平均，然后更新观测的深度值
  void UpdateNormalAndDepth();

  // 获取平均观测方向
  Point3dT GetNormal();

  // 计算此地图点的观测层级
  int PredictScale(const float& dist, const float& scaleFactor, const int& nLevels);

  // 获取地图点的最近可能观测距离
  float GetMinDistanceInvariance();
  // 获取地图点的最远可能观测距离
  float GetMaxDistanceInvariance();

  // -------------- 描述子相关 --------------

  // 计算并设置地图点的描述子
  // 1. 从观测帧中收集该地图点的描述子
  // 2. 计算描述子之间的距离
  // 3. 选择一个到其他描述子距离的中位数最小的作为描述子
  void ComputeDistinctiveDescriptors();
  cv::Mat GetDescriptor();

 public:
  long unsigned int mnId;            // 地图点的ID
  static long unsigned int nNextId;  // 下一个地图点的ID
  long int mnFirstKfId;              // 第一次观测到该地图点的关键帧ID
  // 最近一次观测到该地图点的帧ID, 用在LocalMapOptimization中，
  // SearchByProjection会去掉那些最近一次观测帧就是当前帧的地图点
  // 这些地图点在TrackReferenceKeyFrame中被使用
  long unsigned int mnLastFrameSeen;

 private:
  // 地图点的三维坐标
  Point3dT mPos;
  // 地图点的参考关键帧
  KeyFrame* mpRefKF;
  // 地图指针
  Map* mpMap;

  // 坏点标记 —— 地图点被判定为坏点不会立马被删除
  bool mbBad;
  // 被替换的地图点
  MapPoint* mpReplaced;

  // -------------- 观测相关 --------------
  // 观测关键帧
  std::map<KeyFrame*, size_t> mObservations;
  // 平均观测方向 -- 对所有观测帧的方向向量求平均
  Normal3dT mNormalVector;

  // 在TrackLocalMap中，所有能够被当前帧观测（在观测范围、观测角度限制内）的地图点、以及在位姿初始化阶段被添加到当前帧的地图点都会新增一个观测
  int mnVisible = 0;
  // 在TrackLocalMap中，位姿优化之后，那些还在当前帧里面，且没有在优化期间没有被置为Outlier的地图点，会新增一个匹配
  int mnKeepAfterBA = 0;

  float mfMaxDistance;
  float mfMinDistance;

  // -------------- 描述子相关 --------------
  // 描述子
  cv::Mat mDescriptor;

  // 互斥锁
  std::mutex mMutexPos;     // 操作坐标的互斥锁
  std::mutex mMutexObsers;  // 操作观测关键帧的互斥锁
};
}  // namespace ORB_SLAM_Tracking
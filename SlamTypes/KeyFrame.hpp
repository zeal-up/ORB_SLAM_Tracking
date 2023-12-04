#pragma once

#include "SlamTypes/BasicTypes.hpp"
#include "SlamTypes/Frame.hpp"

namespace ORB_SLAM_Tracking {

class Map;
class MapPoint;

/**
 * @class KeyFrame
 * @brief Class for keyframe.
 * 相比于ORBSLAM的实现，这里直接继承自Frame类
 */
class KeyFrame : public Frame {
 public:
  KeyFrame() = delete;
  KeyFrame(const KeyFrame& kf) = delete;
  KeyFrame& operator=(const KeyFrame& kf) = delete;
  KeyFrame(Frame& F, Map* pMap);

  unsigned long GetId() const { return mnKfId; }
  static void Reset() { nNextKfId = 0; }
  

  // 相比与Frame子类，需要扩展的函数
 public:
  
  // deprecated ： defined in Frame
  // void ComputeBoW();

  // ---------------- 元数据 ----------------

  // 是否是坏帧
  bool isBad();

  // 设置为坏帧：1）从所有连接关系删除当前关键帧;2）从所有地图点删除当前关键帧的观测;3）将所有children的父节点更新为当前关键帧的父节点
  void SetBadFlag();


  // ---------------- 地图点相关 ----------------

  /**
   * @brief 添加地图点
   * 内部会维护一个长度为特征点数量的地图点指针数组
   * 后续可以查看这个数组，确定某个特征点是否有对应的地图点
   *
   * @param pMP 地图点指针
   * @param idx 地图点在这个关键帧的关键点索引
   */
  void AddMapPoint(MapPoint* pMP, const size_t& idx);

  // 计算关键帧中包含的地图点的深度值（Frame坐标系）的中位数
  float ComputeSceneMedianDepth();

  // 此关键帧跟踪的地图点数量, 只统计观测数量大于minObs的地图点
  int TrackedMapPoints(const int& minObs = 0);

  // -------------- 共视图相关 --------------

  // 更新共视图
  // 遍历关键帧中的所有地图点，找到该地图点的所有观测
  // 遍历这些观测，找到这些观测对应的关键帧
  // 如果这些关键帧与当前关键帧的共视点大于阈值，那么就认为这些关键帧与当前关键帧是共视关系
  void UpdateConnections();

  // 获取共视关键帧 （按照共视点数量从大到小排序）
  // N : 返回的共视帧数量。如果N小于0,则返回所有的共视帧
  std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(const int& N);

  // 获取共视权重
  int GetWeight(KeyFrame* pKF);


  // 添加子节点
  void AddChild(KeyFrame* pKF);
  // 删除儿子节点
  void EraseChild(KeyFrame* pKF);
  // 获取儿子节点
  std::set<KeyFrame*> GetChilds();

  // 更新父节点
  void ChangeParent(KeyFrame* pKF);
  // 获取父节点
  KeyFrame* GetParent();


  /**
   * @brief 添加连接关系
   * 如果连接关系有更新，会对内部的共视关键帧进行重新排序
   * 在这里并不会更新父节点，父节点只会在新插入关键帧的时候更新
   * 
   * 
   * @param pKF 关键帧指针
   * @param weight 连接权重，也就是共视点的个数
   */
  void AddConnection(KeyFrame* pKF, const int& weight);

  void EraseConnection(KeyFrame* pKF);

 private:

  long unsigned int mnKfId;  // 当前关键帧的ID, mnId是帧的ID
  static long unsigned int nNextKfId;  // 下一个关键帧的ID
  

  // 标记位 ----------------------------------------------
  bool mbBad = false;  // 是否是坏帧
  bool mbFirstConnection = false;  // 是否是第一次添加连接关系，如果是，那么需要更新父节点

  // 地图指针
  Map* mpMap;

  // 共视图相关 ----------------------------------------------
  // 共视关键帧-》共视点个数
  std::map<KeyFrame*, int> mConnectedKeyFrameWeights;
  // 共视关键帧，按照共视点个数从大到小排序
  std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames;
  // 共视关键帧的权重（也就是共视点个数）
  std::vector<int> mvOrderedWeights;
  // 父节点
  KeyFrame* mpParent = nullptr;
  // 子节点集合（删除此关键帧时，需要将子节点的父节点更新为此节点的父节点）
  std::set<KeyFrame*> mspChildrens;

  // BoW相关 (deprecated, defined in Frame) -----------------
  // DBoW2::FeatureVector mFeatVec;  // Direct index, 可以加速匹配
  // DBoW2::BowVector mBowVec;       // BoW向量

  // 互斥锁 --------------------------------------------------
  std::mutex mMutexFeatures;
  std::mutex mMutexConnections;
  std::mutex mMutexPose;
};

typedef std::shared_ptr<KeyFrame> KeyFramePtr;
}  // namespace ORB_SLAM_Tracking
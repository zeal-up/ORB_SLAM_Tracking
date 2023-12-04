#include "SlamTypes/Map.hpp"
#include "SlamTypes/MapPoint.hpp"

namespace ORB_SLAM_Tracking {

Map::Map() : mbMapUpdated(false), mnMaxKfId(0) {}

void Map::Clear() {
  std::lock_guard<std::mutex> lock(mMutexMap);
  for (auto ptIter : mspMapPoints) {
    delete ptIter;
  }
  for (auto kfIter : mspKeyFrames) {
    delete kfIter;
  }
  mspMapPoints.clear();
  mspKeyFrames.clear();
  mnMaxKfId = 0;
}  // Clear

void Map::EraseKeyFrame(KeyFrame* pKF) {
  std::scoped_lock lock(mMutexMap);
  pKF->SetBadFlag();
  mspKeyFrames.erase(pKF);
  mbMapUpdated = true;
} // EraseKeyFrame

void Map::AddKeyFrame(KeyFrame* pKF) {
  // 互斥锁 —— 跟踪线程和局部地图线程可能同时往地图中增加关键帧
  std::lock_guard<std::mutex> lock(mMutexMap);
  mspKeyFrames.insert(pKF);
  if (pKF->GetId() > mnMaxKfId) {
    mnMaxKfId = pKF->GetId();
  }
  mbMapUpdated = true;
}  // AddKeyFrame

void Map::AddMapPoint(MapPoint* pMP) {
  // 互斥锁 —— 跟踪线程和局部地图线程可能同时往地图中增加地图点
  std::lock_guard<std::mutex> lock(mMutexMap);
  mspMapPoints.insert(pMP);
  mbMapUpdated = true;
}  // AddMapPoint

void Map::EraseMapPoint(MapPoint* pMP) {
  std::scoped_lock<std::mutex> lock(mMutexMap);
  mspMapPoints.erase(pMP);
  mbMapUpdated = true;
}

// 获取地图点数量
int Map::MapPointsInMap() {
  std::lock_guard<std::mutex> lock(mMutexMap);
  return mspMapPoints.size();
}  // MapPointsInMap

int Map::KeyFramesInMap() {
  std::lock_guard<std::mutex> lock(mMutexMap);
  return mspKeyFrames.size();
}  // KeyFramesInMap

}  // namespace ORB_SLAM_Tracking
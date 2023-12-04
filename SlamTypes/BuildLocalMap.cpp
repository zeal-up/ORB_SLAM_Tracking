#include "SlamTypes/BuildLocalMap.hpp"

namespace ORB_SLAM_Tracking {

BuildLocalMap::BuildLocalMap() {}

bool BuildLocalMap::BuildLocalMapPoints(const FramePtr curF, std::vector<MapPoint*>& vpLocalMapPoints, KeyFramePtr& bestCovisibleKF) {
  vpLocalMapPoints.clear(); 
  bool bOK = false;

  std::vector<KeyFramePtr> vpLocalKeyFrames;
  bOK = UpdateLocalKeyFrames(curF, vpLocalKeyFrames, bestCovisibleKF);
  if (!bOK) {
    return false;
  }
  std::unordered_map<MapPoint*, int> localMapPointsSeen;  // 用于去重加速查找

  for (auto kf : vpLocalKeyFrames) {
    const std::vector<MapPoint*> vpMPs = kf->mvpMapPoints;
    for (auto mp : vpMPs) {
      if (!mp) continue;
      if (mp->isBad()) continue;
      
      if (localMapPointsSeen[mp] == 0) {
        vpLocalMapPoints.push_back(mp);
        localMapPointsSeen[mp] = 1;
      }
    }
  } // loop for vpLocalKeyFrames

  return true;
}

bool BuildLocalMap::UpdateLocalKeyFrames(const FramePtr curF, std::vector<KeyFramePtr>& vpLocalKeyFrames, KeyFramePtr& pKFmax) {
  vpLocalKeyFrames.clear();

  // Step 1 获取当前帧的共视关系 ----------------------------------------------------
  std::map<KeyFrame*, int> keyframeCounter; // 统计与当前帧的共视点数量
  for (int i = 0; i < curF->N; ++i) {
    if (!curF->mvpMapPoints[i]) continue;
    MapPoint* pMP = curF->mvpMapPoints[i];
    if (pMP->isBad()) {
      curF->mvpMapPoints[i] = nullptr;
      continue;
    }
    const std::map<KeyFrame*, size_t> observations = pMP->GetObservations();
    for (auto iter = observations.begin(); iter != observations.end(); ++iter) {
      keyframeCounter[iter->first]++;
    }
  } // loop for curF->mvpMapPoints

  if (keyframeCounter.empty()) {
    return false;
  }

  // Step 2 将共视帧添加到局部地图（同时记录共视点最多的关键帧） ---------------------
  int max = 0;
  pKFmax = nullptr;  // 与当前帧共视点最多的关键帧
  vpLocalKeyFrames.reserve(3*keyframeCounter.size());
  for (auto iter = keyframeCounter.begin(); iter != keyframeCounter.end(); ++iter) {
    KeyFrame* pKF = iter->first;
    if (pKF->isBad()) continue;
    if (iter->second > max) {
      max = iter->second;
      pKFmax = std::shared_ptr<KeyFrame>(pKF);
    }
    vpLocalKeyFrames.push_back(std::shared_ptr<KeyFrame>(pKF));
  }

  // Step 3 寻找共视帧的共视帧，将其添加到局部地图 ----------------------------------
  for (auto iter = vpLocalKeyFrames.begin(), iterEnd = vpLocalKeyFrames.end(); iter != iterEnd; ++iter) {
    if (vpLocalKeyFrames.size() > 80) break; // 限制局部地图关键帧数量
    KeyFramePtr pKF = *iter;
    const std::vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);
    for (auto neigh : vNeighs) {
      if (neigh->isBad()) continue;
      if (std::find(vpLocalKeyFrames.begin(), vpLocalKeyFrames.end(), KeyFramePtr(neigh)) != vpLocalKeyFrames.end()) continue;
      vpLocalKeyFrames.push_back(std::shared_ptr<KeyFrame>(neigh));
    }

    // 把相邻帧的子关键帧也添加到局部地图
    const std::set<KeyFrame*> spChilds = pKF->GetChilds();
    for (auto child : spChilds) {
      if (child->isBad()) continue;
      if (std::find(vpLocalKeyFrames.begin(), vpLocalKeyFrames.end(), KeyFramePtr(child)) != vpLocalKeyFrames.end()) continue;
      vpLocalKeyFrames.push_back(std::shared_ptr<KeyFrame>(child));
      break; // 只添加共视关系最强的子关键帧
    }

    // 把相邻帧的父关键帧也添加到局部地图
    KeyFrame* pParent = pKF->GetParent();
    if (pParent && !pParent->isBad()) {
      if (std::find(vpLocalKeyFrames.begin(), vpLocalKeyFrames.end(), KeyFramePtr(pParent)) != vpLocalKeyFrames.end()) continue;
      vpLocalKeyFrames.push_back(std::shared_ptr<KeyFrame>(pParent));
    }
  } // loop for vpLocalKeyFrames


  return true;
}

} // namespace ORB_SLAM_Tracking
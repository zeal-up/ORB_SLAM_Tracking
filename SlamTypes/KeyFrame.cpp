#include "SlamTypes/KeyFrame.hpp"

#include "SlamTypes/MapPoint.hpp"
#include "SlamTypes/Map.hpp"
#include "Utils/Converter.hpp"

namespace ORB_SLAM_Tracking {

long unsigned int KeyFrame::nNextKfId = 0;

KeyFrame::KeyFrame(Frame& F, Map* pMap) : Frame(F) {
  // 设置关键帧ID
  mnKfId = nNextKfId++;
  // 设置地图指针
  mpMap = pMap;

  mbBad = false;
}

void KeyFrame::AddMapPoint(MapPoint* pMP, const size_t& idx) {
  std::lock_guard<std::mutex> lock(mMutexFeatures);
  // 添加地图点
  mvpMapPoints[idx] = pMP;  // mvpMapPoints是Frame类中的成员, 长度是特征点数量
}  // AddMapPoint

bool KeyFrame::isBad() {
  std::lock_guard<std::mutex> lock(mMutexConnections);
  return mbBad;
}  // isBad

void KeyFrame::SetBadFlag() {
  // Step 1 删除与当前帧连接的关键帧的连接关系 ------------------------------------
  for (const auto& pairKfWeight : mConnectedKeyFrameWeights) {
    KeyFrame* pKF = pairKfWeight.first;
    pKF->EraseConnection(this);
  }

  // Step 2 删除当前帧的所有地图点的观测关系 --------------------------------------
  {
    std::scoped_lock lock(mMutexFeatures);
    for (const auto& pMP : mvpMapPoints) {
      if (pMP) {
        pMP->EraseObservation(this);
      }
    }
  }

  // Step 3 更新Spanning Tree关系  -----------------------------------------------
  // 这部分的实现会比较巧妙
  // 这里用的实际是最小生成树构建的Kruskal算法，又叫加边法
  // 这里所有代连接的节点是当前节点的父节点和子节点
  // 将所有节点sFull分成两个集合：sParentCandidates和sChildCandidates
  // sParentCandidates中的节点是已经加到生成树上的节点，sChildCandidates中的节点是还没有加到生成树上的节点
  // 每次循环，找出sChildCandidates中与sParentCandidates中权重最大的边，并将这条边在sChildCandidates中的顶点
  // 加入到sParentCandidates中，直到sChildCandidates为空

  // 这里最后还加入了一个判断，如果sChildCandidates最后不为空，说明这里面的节点与sFull里面的节点都不相连，因此
  // 直接将其父节点设置为当前节点的父节点

  // 为什么不能直接将子节点的父节点设置为其共视关系最强的节点？
  // 因为这样可能会形成回环，违反树结构。
  {
    std::scoped_lock lock(mMutexConnections, mMutexFeatures);
    mConnectedKeyFrameWeights.clear();
    mvpOrderedConnectedKeyFrames.clear();

    // 候选父节点集合——已经加入到生成树上的节点
    // std::set<KeyFrame*> sParentCandidates;
    std::unordered_map<unsigned long, KeyFrame*> mIdParentCandidates;
    // 一开始只有当前节点的父节点
    mIdParentCandidates.at(mpParent->GetId()) = mpParent;

    // 每次循环，找到sChildCandidates中与sParentCandidates中权重最大的边，并将这条边在sChildCandidates中的顶点
    // 加入到sParentCandidates中，直到sChildCandidates为空
    while (!mspChildrens.empty()) {
      // 是否能够在sChildrens与sParentCandidates之间找到一条边
      bool bContinue = false;
      int max = -1;            // 缓存此次循环边的最大权重
      KeyFrame* pC = nullptr;  // 缓存此次循环边的子节点
      KeyFrame* pP = nullptr;  // 缓存此次循环边的父节点

      for (auto& child : mspChildrens) {
        if (child->isBad()) continue;

        // 寻找这个child与sParentCandidates中的节点的最大权重边
        // 这里如果共视帧的缓存关系是一个{ID:KeyFrame*}的map,则查找起来会简洁很多
        auto vpChildConnections = child->GetBestCovisibilityKeyFrames(-1);
        // 查找这些连接关系是否有连接到sParentCandidates中的节点
        for (auto& childConnect : vpChildConnections) {
          if (childConnect->isBad()) continue;
          if (mIdParentCandidates.count(childConnect->GetId())) {
            // 如果找到了，那么就更新max和pC,pP
            int w = child->GetWeight(childConnect);
            if (w > max) {
              max = w;
              pC = child;
              pP = childConnect;  // 这里的childConnect就是sParentCandidates中的节点
              bContinue =
                  true;  // 找到了一条边，可以将这条边加入生成树，从sChildCandidates中删除节点，加入到sParentCandidates中，然后继续执行
            }
          }
        }
      } // 从当前的mspChildrens集合中找到与sParentCandidates中的节点连接的最大权重边
      
      if (bContinue) {
        pC->ChangeParent(pP);  // 将pC的父节点设置为pP
        mIdParentCandidates.at(pC->GetId()) = pC;  // 将pC加入到sParentCandidates中
        mspChildrens.erase(pC);  // 从sChildCandidates中删除pC
      } else {  // 没有找到与sParentCandidates中的节点连接的边
        break;
      }
    } // while (!mspChildrens.empty())

    // 如果sChildCandidates最后不为空，说明这里面的节点与sFull里面的节点都不相连，因此
    // 直接将其父节点设置为当前节点的父节点
    if (!mspChildrens.empty()) {
      for (auto& child : mspChildrens) {
        child->ChangeParent(mpParent);
      }
    }

    // 最后，将当前节点的父节点中删除当前节点
    mpParent->EraseChild(this);
    mbBad = true;

  } // lock - mMutexConnections, mMutexFeatures

  // 原有的调用关系在这里调用Map::EraseKeyFrame(this)
  // 但是我觉得更合理的方式是在Map::EraseKeyFrame中调用KeyFrame::SetBadFlag
  // 这里更改为我认为合理的调用方式。如果后面实现得好，KeyFrame可以不依赖Map
  // Map应该作为管理KeyFrame的入口
  // mpMap->EraseKeyFrame(this);  // source code
}  // SetBadFlag

void KeyFrame::UpdateConnections() {
  std::map<KeyFrame*, int> KFcounter;  // 共视帧-》共视点个数
  std::vector<MapPoint*> vpMP;
  {
    std::lock_guard<std::mutex> lockMPs(mMutexFeatures);
    vpMP = mvpMapPoints;
  }

  // ------------------ Step 1 统计共视帧的共视点个数 ------------------
  // 遍历关键帧中的所有地图点，找到该地图点的所有观测
  // 遍历这些观测，找到这些观测对应的关键帧,并将这些关键帧的共视点个数加1
  for (auto& pMP : vpMP) {
    if (!pMP) {
      continue;
    }
    if (pMP->isBad()) {
      continue;
    }
    std::map<KeyFrame*, size_t> observations;
    observations = pMP->GetObservations();
    for (auto& obs : observations) {
      KeyFrame* pKF = obs.first;
      if (pKF->mnKfId == mnKfId) {
        continue;
      }
      KFcounter[pKF]++;
    }
  }

  if (KFcounter.empty()) {
    return;
  }

  // ------------------ Step 2 添加共视帧（双边联系） ------------------
  // 添加共视点个数大于阈值的共视帧，如果没有，则添加共视个数最大的关键帧
  int nmax = 0;                // 最大的共视个数
  KeyFrame* pKFmax = nullptr;  // 共视最多的关键帧
  int th = 15;                 // 共视阈值
  std::vector<std::pair<KeyFrame*, int>> vPairs(KFcounter.begin(), KFcounter.end());
  // 将共视帧按照共视点数量从大到小排序
  std::sort(vPairs.begin(), vPairs.end(),
            [](const std::pair<KeyFrame*, int>& a, const std::pair<KeyFrame*, int>& b) {
              return a.second > b.second;
            });

  {
    std::lock_guard<std::mutex> lockCon(mMutexConnections);
    mvpOrderedConnectedKeyFrames.clear();
    mvOrderedWeights.clear();
    mConnectedKeyFrameWeights.clear();

    // 原始代码这里可能有一个bug
    // 原始代码在这个函数会做th阈值判断，但是最后把整个KFcounter都添加到了mConnectedKeyFrameWeights中
    // 在AddConnection函数又会对整个mConnectedKeyFrameWeights进行排序
    // 所以在这里的th阈值并没有起到作用，跟论文里的描述有点不符
    // github上有人提出了一个相同的问题：https://github.com/raulmur/ORB_SLAM2/issues/897
    // 这里修改为符合论文的描述！！！（只添加共视点个数大于15的共视帧）

    // 添加共视点个数大于阈值的共视帧
    for (auto& p : vPairs) {
      if (p.second >= th) {
        mvpOrderedConnectedKeyFrames.push_back(p.first);
        mvOrderedWeights.push_back(p.second);
        mConnectedKeyFrameWeights[p.first] = p.second;
        p.first->AddConnection(this, p.second);  // 同时添加双边联系
      } else {
        break;
      }
    }
    // 如果没有共视点个数大于阈值的共视帧，那么添加共视个数最大的关键帧
    if (mvpOrderedConnectedKeyFrames.empty()) {
      mvpOrderedConnectedKeyFrames.push_back(vPairs[0].first);
      mvOrderedWeights.push_back(vPairs[0].second);
      mConnectedKeyFrameWeights[vPairs[0].first] = vPairs[0].second;
      vPairs[0].first->AddConnection(this, vPairs[0].second);  // 同时添加双边联系
    }

    if (mbFirstConnection && mnKfId != 0) {
      mpParent = mvpOrderedConnectedKeyFrames[0];
      mpParent->AddChild(this);
      mbFirstConnection = false;
    }
  }  // lock - mMutexConnections

}  // UpdateConnections

std::vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int& N) {
  std::scoped_lock<std::mutex> lock(mMutexConnections);
  // 如果N小于0,返回所有关键帧
  if (N < 0) return mvpOrderedConnectedKeyFrames;

  if (mvpOrderedConnectedKeyFrames.size() < N) {
    return mvpOrderedConnectedKeyFrames;
  } else {
    return std::vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(),
                                  mvpOrderedConnectedKeyFrames.begin() + N);
  }
}  // GetBestCovisibilityKeyFrames

int KeyFrame::GetWeight(KeyFrame* pKF) {
  std::scoped_lock lock(mMutexConnections);
  if (mConnectedKeyFrameWeights.count(pKF)) {
    return mConnectedKeyFrameWeights[pKF];
  } else {
    return 0;
  }
}  // GetWeight

void KeyFrame::AddChild(KeyFrame* pKF) {
  std::scoped_lock lock(mMutexConnections);
  mspChildrens.insert(pKF);
}  // AddChild

void KeyFrame::EraseChild(KeyFrame* pKF) {
  std::scoped_lock lock(mMutexConnections);
  mspChildrens.erase(pKF);
} // EraseChild

std::set<KeyFrame*> KeyFrame::GetChilds() {
  std::scoped_lock<std::mutex> lock(mMutexConnections);
  return mspChildrens;
}  // GetChilds

void KeyFrame::ChangeParent(KeyFrame* pKF) {
  std::scoped_lock<std::mutex> lock(mMutexConnections);
  mpParent = pKF;
  pKF->AddChild(this);
}  // ChangeParent

KeyFrame* KeyFrame::GetParent() {
  std::scoped_lock<std::mutex> lock(mMutexConnections);
  return mpParent;
}  // GetParent

void KeyFrame::AddConnection(KeyFrame* pKF, const int& weight) {
  std::lock_guard<std::mutex> lock(mMutexConnections);
  // 当此连接关系已经存在且不需要更新权重，直接返回
  if (mConnectedKeyFrameWeights[pKF] == weight) return;

  // 添加共视帧
  mConnectedKeyFrameWeights[pKF] = weight;

  // 由于更新了连接关系，需要重新排序
  std::vector<std::pair<KeyFrame*, int>> vPairs(mConnectedKeyFrameWeights.begin(),
                                                mConnectedKeyFrameWeights.end());
  // 从大到小排序
  sort(vPairs.begin(), vPairs.end(),
       [](const std::pair<KeyFrame*, int>& a, const std::pair<KeyFrame*, int>& b) {
         return a.second > b.second;
       });

  mvpOrderedConnectedKeyFrames.clear();
  mvOrderedWeights.clear();
  for (auto& p : vPairs) {
    mvpOrderedConnectedKeyFrames.push_back(p.first);
    mvOrderedWeights.push_back(p.second);
  }
}  // AddConnection

void KeyFrame::EraseConnection(KeyFrame* pKF) {
  bool bUpdate = false;
  {
    std::scoped_lock lock(mMutexConnections);
    if (mConnectedKeyFrameWeights.count(pKF)) {
      bUpdate = true;
      mConnectedKeyFrameWeights.erase(pKF);
    }
  }

  if (bUpdate) {
    UpdateConnections();
  }
}  // EraseConnection

float KeyFrame::ComputeSceneMedianDepth() {
  std::vector<MapPoint*> vpMapPoints;
  PoseT Tcw_;
  {
    std::scoped_lock lock(mMutexFeatures);
    std::scoped_lock lock2(mMutexPose);
    vpMapPoints = mvpMapPoints;
    Tcw_ = mTcw;
  }

  std::vector<float> vDepths;
  vDepths.reserve(vpMapPoints.size());
  Eigen::Vector3d Rcw2 = Tcw_.rotation().row(2);
  double zcw = Tcw_.translation().z();
  for (auto& pMP : vpMapPoints) {
    if (pMP) {
      Eigen::Vector3d pos = pMP->GetWorldPos();
      float z = Rcw2.dot(pos) + zcw;
      vDepths.push_back(z);
    }
  }
  sort(vDepths.begin(), vDepths.end());
  return vDepths[vDepths.size() - 1 / 2];
}  // ComputeSceneMedianDepth

int KeyFrame::TrackedMapPoints(const int& minObs) {
  std::lock_guard<std::mutex> lock(mMutexFeatures);
  int nPoints = 0;
  for (size_t i = 0, iend = mvpMapPoints.size(); i < iend; ++i) {
    if (mvpMapPoints[i]) {
      if (!mvpMapPoints[i]->isBad()) {
        if (mvpMapPoints[i]->Observations() > minObs) {
          nPoints++;
        }
      }
    }
  }
  return nPoints;
}  // TrackedMapPoints

}  // namespace ORB_SLAM_Tracking
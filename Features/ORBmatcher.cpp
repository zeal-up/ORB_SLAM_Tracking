#include "Features/ORBmatcher.hpp"

#include <glog/logging.h>
#include <unordered_set>
#include <bitset>

#include "SlamTypes/MapPoint.hpp"
#include "Utils/UtilsCV.hpp"

namespace ORB_SLAM_Tracking {

const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_LOW = 50;
const int ORBmatcher::HISTO_LENGTH = 30;

ORBmatcher::ORBmatcher(float nnratio, bool checkOri)
    : mfNNratio(nnratio), mbCheckOrientation(checkOri) {}

int ORBmatcher::SearchForInitialization(Frame& F1, Frame& F2,
                                        std::vector<cv::Point2f>& vbPreMatched,
                                        std::vector<int>& vnMatches12, int windowSize) {
  int nmatches = 0;
  vnMatches12 = std::vector<int>(F1.mvKeysUn.size(), -1);

  // 初始化角度直方图
  std::vector<int> rotHist[HISTO_LENGTH];
  for (int i = 0; i < HISTO_LENGTH; i++) {
    rotHist[i].reserve(500);
  }
  //! 原作者代码是 const float factor = 1.0f/HISTO_LENGTH; 是错误的，更改为下面代码
  // const float factor = 1.0f/HISTO_LENGTH;
  const float factor = HISTO_LENGTH / 360.0f;

  // 开始进行匹配 —— 对于第一帧中的每个特征点，寻找第二帧中的匹配点
  // 记录F2中每个点被匹配的最近距离
  std::vector<int> vMatchedDistance(F2.N, INT_MAX);
  // 记录F2中每个点被匹配到的initFrame中的点的索引
  std::vector<int> vnMatches21(F2.N, -1);

  // 统计各种不合格匹配点的数量
  int invalidMatchByDistance = 0, invalidMatchByRatio = 0, invalidMatchByOrientation = 0;
  int emptyFeatInArea = 0;
  int kpNumLevel0 = 0;

  // 从第一帧中的每个特征点开始寻找匹配点
  for (size_t i1 = 0, iend1 = F1.N; i1 < iend1; i1++) {
    cv::KeyPoint kp1 = F1.mvKeysUn[i1];
    int level1 = kp1.octave;
    if (level1 > 0) continue;  // 只搜索最精细层的特征点,其他层的特征点直接跳过
    kpNumLevel0++;

    // Step 1 --------------------------------------------------------------
    // 从F2中找出点kp1所在半径范围内的特征点
    // 这里minLevel和maxLevel都是0，所以只会搜索最精细层的特征点
    const cv::Point2f& kpSearchPosition = vbPreMatched[i1];
    std::vector<size_t> vIndices2 =
        F2.GetFeaturesInArea(kpSearchPosition.x, kpSearchPosition.y, windowSize, level1, level1);
    if (vIndices2.empty()) {
      emptyFeatInArea++;
      continue;
    } 

    // Step 2 --------------------------------------------------------------
    // 遍历F2中kp1所在区域内的特征点，找出最近的点

    // 取出点kp1的描述子
    cv::Mat d1 = F1.mDescriptors.row(i1);

    int bestDist = INT_MAX;   // 最近距离
    int bestDist2 = INT_MAX;  // 次近距离
    int bestIdx2 = -1;        // 最近距离对应的特征点索引(在F2中)

    for (auto idx2 : vIndices2) {
      // 取出F2中的特征点描述子
      cv::Mat d2 = F2.mDescriptors.row(idx2);

      // Step 2.1 --------------------------------------------------------------
      // 计算描述子距离，如果距离比最近距离还要近，则更新最近距离
      // int dist = ORB_SLAM2::ORBmatcher::DescriptorDistance(d1, d2);
      int dist = DBoW2::FORB::distance(d1, d2);

      // 如果该点已经被匹配过，并且距离比当前点近，则跳过
      if (vMatchedDistance[idx2] <= dist) continue;

      // 更新最近距离
      if (dist < bestDist) {
        bestDist2 = bestDist;
        bestDist = dist;
        bestIdx2 = idx2;
      } else if (dist < bestDist2) {
        bestDist2 = dist;
      }
    }

    // Step 2.2 --------------------------------------------------------------
    // 对找出的匹配点进行检查：满足阈值、最优/次优比例
    // 必须小于TH_LOW阈值（严格的阈值）
    if (bestDist > TH_LOW) {
      invalidMatchByDistance++;
      continue;
    }
    // 最优/次优比例必须小于mfNNratio，默认0.9
    if (static_cast<float>(bestDist) > mfNNratio * static_cast<float>(bestDist2)) {
      invalidMatchByRatio++;
      continue;
    }

    // Step 2.3 --------------------------------------------------------------
    // 删除重复匹配点
    // 如果匹配点已经被匹配过了，需要把原来的匹配删除（当前的匹配的距离一定会更近）
    if (vnMatches21[bestIdx2] >= 0) {
      vnMatches12[vnMatches21[bestIdx2]] = -1;
      nmatches--;
    }

    // 更新匹配点
    vnMatches12[i1] = bestIdx2;
    vnMatches21[bestIdx2] = i1;
    vMatchedDistance[bestIdx2] = bestDist;
    nmatches++;

    // Step 2.4 --------------------------------------------------------------
    // 计算匹配点的旋转角度差所在的直方图位置
    if (mbCheckOrientation) {
      float rot = F1.mvKeysUn[i1].angle - F2.mvKeysUn[bestIdx2].angle;
      if (rot < 0.0) rot += 360.0f;
      // rot * factor = rot / 360 * HISTO_LENGTH
      int bin = round(rot * factor);
      if (bin == HISTO_LENGTH) bin = 0;
      assert(bin >= 0 && bin < HISTO_LENGTH);
      rotHist[bin].push_back(i1);
    }

  }  // 遍历F1中的每个特征点

  // Step 3 --------------------------------------------------------------
  // 根据直方图来删除角度不一致的匹配点
  // 只保留直方图前三bin的匹配点
  if (mbCheckOrientation) {
    int ind1 = -1;
    int ind2 = -1;
    int ind3 = -1;

    ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

    for (int i = 0; i < HISTO_LENGTH; i++) {
      // 如果不是前三个bin，则删除该bin中的匹配点
      if (i != ind1 && i != ind2 && i != ind3) {
        for (auto it = rotHist[i].begin(), itend = rotHist[i].end(); it != itend; it++) {
          int idx1 = *it;
          vnMatches12[idx1] = -1;
          nmatches--;
          invalidMatchByOrientation++;
        }
      }
    }
  }  // mbCheckOrientation

  // Step 4 --------------------------------------------------------------
  // 更新vbPrevMatched
  for (size_t i1 = 0, iend1 = vnMatches12.size(); i1 < iend1; i1++) {
    if (vnMatches12[i1] >= 0) {
      vbPreMatched[i1] = F2.mvKeysUn[vnMatches12[i1]].pt;
    }
  }

  // print statistics
  LOG(INFO) << "SearchForInitialization done ----------------------";
  LOG(INFO) << "\tkpNumLevel0: " << kpNumLevel0;
  LOG(INFO) << "\temptyFeatInArea: " << emptyFeatInArea;
  LOG(INFO) << "\tinvalidMatchByDistance: " << invalidMatchByDistance;
  LOG(INFO) << "\tinvalidMatchByRatio: " << invalidMatchByRatio;
  LOG(INFO) << "\tinvalidMatchByOrientation: " << invalidMatchByOrientation;

  return nmatches;
}  // SearchForInitialization

int ORBmatcher::SearchByBoW(KeyFramePtr refF, FramePtr curF, std::vector<MapPoint*>& vpMatches21) {
  const std::vector<cv::KeyPoint>& vKeys1 = refF->mvKeysUn;
  const DBoW2::FeatureVector& vFeatVec1 = curF->mFeatVec;
  const std::vector<MapPoint*> vpMapPoints1 = refF->mvpMapPoints;
  const cv::Mat& descriptors1 = refF->mDescriptors;

  const std::vector<cv::KeyPoint>& vKeys2 = curF->mvKeysUn;
  const DBoW2::FeatureVector& vFeatVec2 = curF->mFeatVec;
  const std::vector<MapPoint*> vpMapPoints2 = curF->mvpMapPoints;
  const cv::Mat& descriptors2 = curF->mDescriptors;

  // refF -> curF 的匹配索引
  vpMatches21 = std::vector<MapPoint*>(vpMapPoints2.size(), static_cast<MapPoint*>(nullptr));
  // 对于curF中的每个特征点，记录其是否找到匹配点
  std::vector<bool> vbMatched2(vpMapPoints2.size(), false);

  std::vector<int> rotHist[HISTO_LENGTH];  // 旋转角直方图
  //! 原作者代码是 const float factor = 1.0f/HISTO_LENGTH; 是错误的，更改为下面代码
  // const float factor = 1.0f/HISTO_LENGTH;
  const float factor = HISTO_LENGTH / 360.0f;

  int nmatches = 0;  // 匹配点的数量

  // 通过FeatureVector进行特征匹配，FeatureVector[i]存储的是BoW树中第i个节点的特征点索引
  // FeatureVector 继承 std::map<NodeId, std::vector<unsigned int> >
  auto f1it = vFeatVec1.begin();
  auto f2it = vFeatVec2.begin();
  auto f1end = vFeatVec1.end();
  auto f2end = vFeatVec2.end();

  while (f1it != f1end && f2it != f2end) {
    // 如果两个iter的NodeId相等，进行匹配查找
    if (f1it->first == f2it->first) {
      // 遍历Feat1中该Node里面的所有特征，从Feat2中对应的Node找到最近的特征
      for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++) {
        const size_t idx1 = f1it->second[i1];
        MapPoint* pMP1 = vpMapPoints1[idx1];
        if (!pMP1) continue;          // 如果该特征点没有对应的地图点，则跳过
        if (pMP1->isBad()) continue;  // 如果该地图点已经被标记为坏点，则跳过

        const cv::Mat& d1 = descriptors1.row(idx1);  // 取出该特征点的描述子

        int bestDist1 = 256;
        int bestIdx2 = -1;
        int bestDist2 = 256;  // 次近距离
        for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2; i2++) {
          const size_t idx2 = f2it->second[i2];
          if (vbMatched2[idx2]) continue;  // 如果该特征点已经被匹配过，则跳过
          const cv::Mat& d2 = descriptors2.row(idx2);  // 取出该特征点的描述子

          const int dist = DBoW2::FORB::distance(d1, d2);  // 计算描述子距离
          if (dist < bestDist1) {
            bestDist2 = bestDist1;
            bestDist1 = dist;
            bestIdx2 = idx2;
          } else if (dist < bestDist2) {
            bestDist2 = dist;
          }
        }

        if (bestDist1 > TH_LOW) continue;  // 如果最近距离大于阈值，则跳过
        if (static_cast<float>(bestDist1) > mfNNratio * static_cast<float>(bestDist2))
          continue;                    // 如果最优/次优比例大于阈值，则跳过
        vpMatches21[bestIdx2] = pMP1;  // 更新匹配点
        const cv::KeyPoint& kp1 = vKeys1[idx1];
        if (mbCheckOrientation) {
          float rot = kp1.angle - vKeys2[bestIdx2].angle;
          if (rot < 0.0) rot += 360.0f;
          int bin = round(rot * factor);
          if (bin == HISTO_LENGTH) bin = 0;
          assert(bin >= 0 && bin < HISTO_LENGTH);
          rotHist[bin].push_back(idx1);
        }
        nmatches++;
      }  // iter for idx1 in Node
      // 下面的均为对齐NodeId
      f1it++;
      f2it++;
    } else if (f1it->first < f2it->first) {
      f1it = vFeatVec1.lower_bound(f2it->first);
    } else {
      f2it = vFeatVec2.lower_bound(f1it->first);
    }
  }  // end of while

  if (mbCheckOrientation) {
    int ind1 = -1;
    int ind2 = -1;
    int ind3 = -1;

    ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

    for (int i = 0; i < HISTO_LENGTH; i++) {
      // 只保留旋转角在前三个bin的匹配点
      if (i != ind1 && i != ind2 && i != ind3) {
        for (auto it = rotHist[i].begin(), itend = rotHist[i].end(); it != itend; it++) {
          vpMatches21[*it] = static_cast<MapPoint*>(nullptr);
          nmatches--;
        }
      }
    }
  }  // mbCheckOrientation

  return nmatches;
}

int ORBmatcher::SearchByProjection(FramePtr curF, const FramePtr lastF, const float radius) {
  int nmatches = 0;

  // 旋转角直方图
  std::vector<int> rotHist[HISTO_LENGTH];
  //! 原作者代码是 const float factor = 1.0f/HISTO_LENGTH; 是错误的，更改为下面代码
  // const float factor = 1.0f/HISTO_LENGTH;
  const float factor = HISTO_LENGTH / 360.0f;

  const PoseT Tcw = curF->GetPose();

  for (size_t i = 0; i < lastF->N; ++i) {
    MapPoint* pMP = lastF->mvpMapPoints[i];
    if (!pMP) continue;
    if (lastF->mvbOutlier[i]) continue;

    Point3dT x3Dw = pMP->GetWorldPos();
    Point3dT x3Dc = Tcw * x3Dw;
    if (x3Dc[2] <= 0) continue;

    const float invzc = 1.0 / x3Dc[2];
    float u = curF->fx * x3Dc[0] * invzc + curF->cx;
    float v = curF->fy * x3Dc[1] * invzc + curF->cy;
    if (u < curF->mnMinX || u > curF->mnMaxX) continue;
    if (v < curF->mnMinY || v > curF->mnMaxY) continue;

    // 计算搜索半径 ———— 通过特征的的层级放缩半径
    int nLastOctave = lastF->mvKeys[i].octave;
    float searchRadius = radius * curF->mpORBextractor->GetScaleFactors()[nLastOctave];

    // 找到CurF的对应半径中的所有特征点
    std::vector<size_t> vIndices2 =
        curF->GetFeaturesInArea(u, v, searchRadius, nLastOctave - 1, nLastOctave + 1);
    if (vIndices2.empty()) continue;

    // 计算描述子距离，找到最近的一个匹配点
    const cv::Mat& dMP = pMP->GetDescriptor();
    int bestDist = INT_MAX;
    int bestIdx2 = -1;

    for (auto idx2 : vIndices2) {
      if (curF->mvpMapPoints[idx2] && curF->mvpMapPoints[idx2]->Observations() > 0) continue;

      const cv::Mat& dCF = curF->mDescriptors.row(idx2);
      const int dist = DBoW2::FORB::distance(dMP, dCF);
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx2 = idx2;
      }
    }

    // 如果最近距离大于阈值，则跳过
    if (bestDist > TH_HIGH) continue;

    curF->mvpMapPoints[bestIdx2] = pMP;
    nmatches++;

    if (mbCheckOrientation) {
      float rot = lastF->mvKeys[i].angle - curF->mvKeys[bestIdx2].angle;
      if (rot < 0.0) rot += 360.0f;
      int bin = round(rot * factor);
      if (bin == HISTO_LENGTH) bin = 0;
      assert(bin >= 0 && bin < HISTO_LENGTH);
      rotHist[bin].push_back(bestIdx2);
    }  // mbCheckOrientation

  }  // loop for lastF->N

  // 根据旋转角直方图来删除角度不一致的匹配点
  if (mbCheckOrientation) {
    int ind1 = -1;
    int ind2 = -1;
    int ind3 = -1;

    ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);
    for (int i = 0; i < HISTO_LENGTH; i++) {
      if (i != ind1 && i != ind2 && i != ind3) {
        for (auto it = rotHist[i].begin(), itend = rotHist[i].end(); it != itend; it++) {
          curF->mvpMapPoints[*it] = static_cast<MapPoint*>(nullptr);
          nmatches--;
        }
      }
    }  // for(int i=0; i<HISTO_LENGTH; i++)
  }    // mbCheckOrientation

  return nmatches;
}  // SearchByProjection -- TrackWithMotionModel

int ORBmatcher::SearchByProjection(FramePtr curF, const std::vector<MapPoint*>& vpMapPoints,
                                   const float expand) {
  int nmatches = 0;

  std::vector<std::pair<int, int>> vBestDistIdx;
  std::vector<std::pair<int, int>> vSecondBestDistIdx;
  std::vector<std::pair<int, int>> vBestLevel12;
  float th = 4.0; // 检索范围会根据特征点的层级被放大
  // Step 1 : 进行投影搜索可能的匹配点 ---------------------------------------------------------------
  SearchByProjection_(curF.get(), vpMapPoints, th, vBestDistIdx, vSecondBestDistIdx, vBestLevel12);
  const int nMP = vpMapPoints.size();
  // Step 2 ：对候选匹配点进行筛选 ---------------------------------------------------------------------
  for (int idxMP = 0; idxMP < nMP; idxMP++) {
    if (!vpMapPoints[idxMP] || vpMapPoints[idxMP]->isBad()) continue; // 如果该地图点已经被标记为坏点，则跳过
    const auto bestIdx = vBestDistIdx[idxMP].second;
    if (bestIdx < 0) continue;
    MapPoint* pMPinF = curF->mvpMapPoints[bestIdx];
    if (pMPinF && pMPinF->Observations() > 0) continue; // 如果该特征点已经有对应的地图点，则跳过
    const auto secondBestDist = vSecondBestDistIdx[idxMP].first;
    if (secondBestDist > TH_HIGH) continue; // 如果次近距离大于阈值，则跳过
    if (vBestLevel12[idxMP].first == vBestLevel12[idxMP].second) {
      const auto bestDist = vBestDistIdx[idxMP].first;
      if (static_cast<float>(bestDist) > mfNNratio * static_cast<float>(secondBestDist)) continue; // 如果最优/次优比例大于阈值，则跳过
    }

    // 正式接纳，将地图点更新进帧
    curF->mvpMapPoints[bestIdx] = vpMapPoints[idxMP];
    nmatches++;
  } // 筛选候选匹配点

  return nmatches;

  { // deprecated. 原始实现。这里将投影搜索抽出来为单独的函数
    // // Step 1 记录curF中的地图点，不要对已经加入Frame的地图点重新做匹配搜索 -------------------------------
    // std::unordered_set<MapPoint*> spMapPointTrackInFrame;
    // for (auto pMP : curF->mvpMapPoints) {
    //   if (pMP && pMP->Observations() > 0) spMapPointTrackInFrame.insert(pMP);
    // }

    // for (auto pMP : vpMapPoints) {
    //   if (!pMP || pMP->isBad() || spMapPointTrackInFrame.count(pMP)) continue;

    //   // Step 2 计算地图点在当前帧中的投影 -------------------------------------------------
    //   float projU, projV, viewCos;  // 与地图点平均观测方向的夹角
    //   int predictLevel;             // 预测的特征点层级
    //   bool isInFrame = curF->isInFrustum(pMP, 0.5, projU, projV, viewCos, predictLevel);
    //   if (!isInFrame) continue;

    //   // Step 3 获取投影点附近的特征点 -------------------------------------------------------
    //   float r = 4.0;
    //   if (expand > 1.0) r *= expand;
    //   r *= curF->mpORBextractor->GetScaleFactors()[predictLevel];
    //   // 从前一层到后一层的范围内查找特征点
    //   const std::vector<size_t> vIndices2 =
    //       curF->GetFeaturesInArea(projU, projV, r, predictLevel - 1, predictLevel + 1);

    //   if (vIndices2.empty()) continue;

    //   // Step 4 计算描述子距离，找到最近的一个匹配点 -------------------------------------------
    //   const cv::Mat descMP = pMP->GetDescriptor();

    //   int bestDist = INT_MAX;
    //   int bestIdx = -1;
    //   int bestLevel = -1;
    //   int bestDist2 = INT_MAX;  // 次近距离
    //   int bestLevel2 = -1;

    //   for (auto idx2 : vIndices2) {
    //     // 如果这个特征点已经有对应的有效地图点，则跳过
    //     if (curF->mvpMapPoints[idx2] && curF->mvpMapPoints[idx2]->Observations() > 0) continue;

    //     // 获取特征点的描述子
    //     const cv::Mat& desc2 = curF->mDescriptors.row(idx2);
    //     // 计算描述子距离
    //     const int dist = DBoW2::FORB::distance(descMP, desc2);

    //     if (dist < bestDist) {
    //       bestDist2 = bestDist;
    //       bestDist = dist;
    //       bestIdx = idx2;
    //       bestLevel2 = bestLevel;
    //       bestLevel = curF->mvKeys[idx2].octave;
    //     } else if (dist < bestDist2) {
    //       bestDist2 = dist;
    //       bestLevel2 = curF->mvKeys[idx2].octave;
    //     }
    //   }  // for(auto idx2 : vIndices2)

    //   // Step 5 对找出的匹配点进行检查 -------------------------------------------------------
    //   if (bestDist2 > TH_HIGH) continue;
    //   if (bestLevel == bestLevel2 &&
    //       static_cast<float>(bestDist) > mfNNratio * static_cast<float>(bestDist2))
    //     continue;

    //   // Step 6 更新匹配点 --------------------------------------------------------------------
    //   curF->mvpMapPoints[bestIdx] = pMP;
    //   nmatches++;
    // }  // for(auto pMP : vpMapPoints)

    // return nmatches;
  } // deprecated
  
}  // SearchByProjection -- Search local MapPoints

int ORBmatcher::SearchForTriangulation(KeyFramePtr pKF1, KeyFramePtr pKF2, Eigen::Matrix3d F12,
                                       std::vector<cv::KeyPoint>& vMatchedKeysUn1,
                                       std::vector<cv::KeyPoint>& vMatchedKeysUn2,
                                       std::vector<std::pair<size_t, size_t>>& vMatchedIndices) {
  const auto& vpMapPoints1 = pKF1->mvpMapPoints;
  const auto& vpMapPoints2 = pKF2->mvpMapPoints;
  const auto& vKeysUn1 = pKF1->mvKeysUn;
  const auto& vKeysUn2 = pKF2->mvKeysUn;
  const cv::Mat& descriptors1 = pKF1->mDescriptors;
  const cv::Mat& descriptors2 = pKF2->mDescriptors;
  const DBoW2::FeatureVector& vFeatVec1 = pKF1->mFeatVec;
  const DBoW2::FeatureVector& vFeatVec2 = pKF2->mFeatVec;

  // 结果变量
  int nmatches = 0;
  std::vector<bool> vbMatched2(vKeysUn2.size(), false);
  std::vector<int> vMatches12(vKeysUn1.size(), -1);  // pKF1 -> pKF2 的匹配索引

  // 旋转角直方图
  std::vector<int> rotHist[HISTO_LENGTH];
  for (int i = 0; i < HISTO_LENGTH; i++) rotHist[i].reserve(500);
  const float factor = HISTO_LENGTH / 360.0f;

  // 遍历KF1和KF2中的词袋模型节点（FeatureVector），在同一节点进行特征点搜索匹配
  auto f1it = vFeatVec1.begin();
  auto f2it = vFeatVec2.begin();
  auto f1end = vFeatVec1.end();
  auto f2end = vFeatVec2.end();

  while (f1it != f1end && f2it != f2end) {
    // 如果两个iter的NodeId相等，进行匹配查找
    if (f1it->first == f2it->first) {
      // 遍历Feat1该Node里面的所有特征，从Feat2该node里面找到最近的特征
      for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++) {
        const size_t idx1 = f1it->second[i1];
        MapPoint* pMP1 = vpMapPoints1[idx1];
        if (pMP1) continue;  // 如果该特征点已经有对应的地图点，则跳过
        const cv::KeyPoint& kp1 = vKeysUn1[idx1];
        const cv::Mat& d1 = descriptors1.row(idx1);

        // Step 1 对kp1这个点，计算其到Node2中所有特征点的距离，找到最近的一个匹配点
        // ------------------
        std::vector<std::pair<int, size_t>> vDistIdx;  // <距离，特征点索引>，后续可以直接排序
        for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2; i2++) {
          const size_t idx2 = f2it->second[i2];
          MapPoint* pMP2 = vpMapPoints2[idx2];
          // 如果这个特征点已经被匹配过，或者已经有存在的地图点，则跳过
          if (vbMatched2[idx2] || pMP2) continue;

          const cv::Mat& d2 = descriptors2.row(idx2);
          const int dist = DBoW2::FORB::distance(d1, d2);
          if (dist > TH_LOW) continue;  // 如果距离大于阈值，则跳过

          // 将这个候选匹配加入到vDistIdx中
          vDistIdx.push_back(std::make_pair(dist, idx2));
        }  // iter for Node2

        // Step 2 计算完kp1到Node2中所有点的距离之后，开始筛选匹配点
        // ----------------------------------
        if (vDistIdx.empty()) continue;
        sort(vDistIdx.begin(), vDistIdx.end());  // 按距离从小到达排序
        int BesDist = vDistIdx[0].first;
        int DistTh = round(2 * BesDist);  // 2倍最近距离作为阈值
        // 遍历所有候选匹配点
        for (const auto& distIdx : vDistIdx) {
          if (distIdx.first > DistTh) break;  // 如果距离大于阈值，则直接break,后续的也会大于阈值
          const cv::KeyPoint& kp2 = vKeysUn2[distIdx.second];  // 拿出kp2
          // 根据基础矩阵，计算kp1和kp2的极线约束距离
          // 挺奇怪的，这里并没有做对称计算，只计算kp2到投影极线的距离
          const float distEpipolar1to2 = UtilsCV::EpipolarDistance12(kp1, kp2, F12);
          if (distEpipolar1to2 < 0) continue;  // 如果距离小于0，说明计算出错
          // 距离阈值，kp2的层级越高容忍度越高
          if (distEpipolar1to2 > 3.84 * pKF2->mpORBextractor->GetScaleSigmaSquares()[kp2.octave]) {
            continue;
          }

          // 接收此匹配点
          vbMatched2[distIdx.second] = true;
          vMatches12[idx1] = distIdx.second;
          nmatches++;
          // 进行旋转直方图检查
          if (mbCheckOrientation) {
            float rot = kp1.angle - kp2.angle;
            if (rot < 0.0) rot += 360.0f;
            int bin = round(rot * factor);
            if (bin == HISTO_LENGTH) bin = 0;
            assert(bin >= 0 && bin < HISTO_LENGTH);
            rotHist[bin].push_back(idx1);
          }  // mbCheckOrientation
        }    // 遍历候选点
      }      // iter for Node1
      f1it++;
      f2it++;
    }  // if (f1it->first == f2it->first)
    else if (f1it->first < f2it->first) {
      f1it = vFeatVec1.lower_bound(f2it->first);
    } else {
      f2it = vFeatVec2.lower_bound(f1it->first);
    }
  }  // while (f1it != f1end && f2it != f2end)

  if (mbCheckOrientation) {
    int ind1 = -1, ind2 = -1, ind3 = -1;
    ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);
    for (int i = 0; i < HISTO_LENGTH; i++) {
      if (i != ind1 && i != ind2 && i != ind3) {
        for (auto it = rotHist[i].begin(), itend = rotHist[i].end(); it != itend; it++) {
          vMatches12[*it] = -1;
          nmatches--;
        }
      }
    }
  }  // mbCheckOrientation

  // Step 3 记录匹配结果 ---------------------------------------------------------------
  for (size_t i = 0, iend = vMatches12.size(); i < iend; i++) {
    if (vMatches12[i] < 0) continue;
    vMatchedKeysUn1.push_back(vKeysUn1[i]);
    vMatchedKeysUn2.push_back(vKeysUn2[vMatches12[i]]);
    vMatchedIndices.push_back(std::make_pair(i, vMatches12[i]));
  }

  return nmatches;
}  // SearchForTriangulation

int ORBmatcher::Fuse(KeyFramePtr pKF, const std::vector<MapPoint*>& vpMapPoints, const float th) {
  int nFused = 0;
  std::vector<std::pair<int, int>> vBestDistIdx;
  std::vector<std::pair<int, int>> vSecondBestDistIdx;
  std::vector<std::pair<int, int>> vBestLevel12;
  // Step 1 : 进行投影搜索可能的匹配点 ---------------------------------------------------------------
  SearchByProjection_(pKF.get(), vpMapPoints, th, vBestDistIdx, vSecondBestDistIdx, vBestLevel12);
  const int NMapPoint = vpMapPoints.size();
  // Step 2 ：对候选匹配点进行筛选 ---------------------------------------------------------------------
  for (int idxMP = 0; idxMP < NMapPoint; ++idxMP) {
    if (!vpMapPoints[idxMP] || vpMapPoints[idxMP]->isBad()) continue; // 如果该地图点已经被标记为坏点，则跳过
    const auto bestDist = vBestDistIdx[idxMP].first;
    const auto bestIdx = vBestDistIdx[idxMP].second;
    if (bestDist > TH_LOW) continue; // 如果最近距离大于阈值，则跳过
    MapPoint* pMPinKF = pKF->mvpMapPoints[bestIdx];
    MapPoint* pMP = vpMapPoints[idxMP];
    if (pMPinKF) {
      // 该特征点已经有对应的地图点，对比两个地图点的观测数，用观测数多的替换观测数少的
      if (!pMPinKF->isBad() && pMPinKF->Observations() > pMP->Observations()) {
        pMP->Replace(pMPinKF);
      } else {
        pMPinKF->Replace(pMP);
      }
    } // 如果特征点已经有对应的地图点
    else {
      pMP->AddObservation(pKF.get(), bestIdx);
      pKF->AddMapPoint(pMP, bestIdx);
    }
    nFused++;
  } // 筛选候选匹配点
  return nFused;
}

void ORBmatcher::ComputeThreeMaxima(std::vector<int>* histo, const int L, int& ind1, int& ind2,
                                    int& ind3) {
  int max1 = 0;
  int max2 = 0;
  int max3 = 0;

  for (int i = 0; i < L; i++) {
    const int s = histo[i].size();
    if (s > max1) {
      max3 = max2;
      max2 = max1;
      max1 = s;
      ind3 = ind2;
      ind2 = ind1;
      ind1 = i;
    } else if (s > max2) {
      max3 = max2;
      max2 = s;
      ind3 = ind2;
      ind2 = i;
    } else if (s > max3) {
      max3 = s;
      ind3 = i;
    }
  }

  if (max2 < 0.1f * static_cast<float>(max1)) {
    ind2 = -1;
    ind3 = -1;
  } else if (max3 < 0.1f * static_cast<float>(max1)) {
    ind3 = -1;
  }
}  // ComputeThreeMaxima

void ORBmatcher::SearchByProjection_(Frame* tgtF, const std::vector<MapPoint*>& vpMapPoints,
                                     const float th, std::vector<std::pair<int, int>>& bestDistIdx,
                                     std::vector<std::pair<int, int>>& secondBestDistIdx,
                                     std::vector<std::pair<int, int>>& bestLevel12) {
  bestDistIdx.resize(vpMapPoints.size());
  std::fill(bestDistIdx.begin(), bestDistIdx.end(), std::make_pair(-1, -1));
  secondBestDistIdx.resize(vpMapPoints.size());
  std::fill(secondBestDistIdx.begin(), secondBestDistIdx.end(), std::make_pair(-1, -1));
  bestLevel12.resize(vpMapPoints.size());
  std::fill(bestLevel12.begin(), bestLevel12.end(), std::make_pair(-1, -1));

  // Step 1 记录curF中的地图点，不要对已经加入Frame的地图点重新做匹配搜索 -------------------------------
  std::unordered_set<MapPoint*> spMapPointTrackInFrame;
  for (auto pMP : tgtF->mvpMapPoints) {
    if (pMP && pMP->Observations() > 0) spMapPointTrackInFrame.insert(pMP);
  }

  for (size_t mpIdx = 0; mpIdx < vpMapPoints.size(); ++mpIdx) {
    MapPoint* pMP = vpMapPoints[mpIdx];
    if (!pMP || pMP->isBad() || spMapPointTrackInFrame.count(pMP)) continue;

    // Step 2 计算地图点在当前帧中的投影 -------------------------------------------------
    // 这里isInFrustum会检查：1. 地图点是否在帧图像范围点; 2. 观测夹角是否小于60度; 3. 观测距离是否在[dmin,dmax]范围
    float projU, projV, viewCos;  // 与地图点平均观测方向的夹角
    int predictLevel;             // 预测的特征点层级
    // cos(60) = 0.5
    bool isInFrame = tgtF->isInFrustum(pMP, 0.5, projU, projV, viewCos, predictLevel);
    if (!isInFrame) continue;

    // Step 3 获取投影点附近的特征点 -------------------------------------------------------
    float r = 4.0 * tgtF->mpORBextractor->GetScaleFactors()[predictLevel];
    // 从前一层到后一层的范围内查找特征点
    const std::vector<size_t> vIndices2 =
        tgtF->GetFeaturesInArea(projU, projV, r, predictLevel - 1, predictLevel + 1);

    if (vIndices2.empty()) continue;

    // Step 4 计算描述子距离，找到最近的一个匹配点 -------------------------------------------
    const cv::Mat descMP = pMP->GetDescriptor();
    
    int bestDist = INT_MAX;
    int bestIdx = -1;
    int bestLevel = -1;
    int bestDist2 = INT_MAX;  // 次近距离
    int bestIdx2 = -1;
    int bestLevel2 = -1;

    for (auto idxInF : vIndices2) {
      // 去除关键点的描述子
      const cv::Mat descF = tgtF->mDescriptors.row(idxInF);
      // 计算描述子距离
      const int dist = DBoW2::FORB::distance(descMP, descF);

      if (dist < bestDist) {
        bestDist2 = bestDist;
        bestDist = dist;
        bestIdx2 = bestIdx;
        bestIdx = idxInF;
        bestLevel2 = bestLevel;
        bestLevel = tgtF->mvKeys[idxInF].octave;
      } else if (dist < bestDist2) {
        bestDist2 = dist;
        bestIdx2 = idxInF;
        bestLevel2 = tgtF->mvKeys[idxInF].octave;
      }
    } // for(auto idx2 : vIndices2)

    // Step 5 将找出的匹配点记录下来  -------------------------------------------------------
    bestDistIdx[mpIdx] = std::make_pair(bestDist, bestIdx);
    secondBestDistIdx[mpIdx] = std::make_pair(bestDist2, bestIdx2);
    bestLevel12[mpIdx] = std::make_pair(bestLevel, bestLevel2);
  }

}

}  // namespace ORB_SLAM_Tracking
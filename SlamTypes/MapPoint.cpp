#include "SlamTypes/MapPoint.hpp"
#include "SlamTypes/KeyFrame.hpp"
#include "SlamTypes/Map.hpp"

namespace ORB_SLAM_Tracking {

long unsigned int MapPoint::nNextId = 0;

MapPoint::MapPoint(const Point3dT& Pos, KeyFrame* pRefKF, Map* pMap)
: mPos(Pos), mpRefKF(pRefKF), mpMap(pMap), mnVisible(1), mnKeepAfterBA(1),
 mbBad(false) {
    mnId = nNextId++;
    mnFirstKfId = pRefKF->GetId();

    mNormalVector = Normal3dT(0, 0, 0);
} // MapPoint constructor

Point3dT MapPoint::GetWorldPos() {
    std::unique_lock<std::mutex> lock(mMutexPos);
    return mPos;
} // GetWorldPos

void MapPoint::SetWorldPos(const Point3dT& Pos) {
    std::unique_lock<std::mutex> lock(mMutexPos);
    mPos = Pos;
} // SetWorldPos

KeyFrame* MapPoint::GetReferenceKeyFrame() {
    std::unique_lock<std::mutex> lock(mMutexPos);
    return mpRefKF;
} // GetReferenceKeyFrame

void MapPoint::SetBadFlag() {
  std::map<KeyFrame*, size_t> observations;
  {
    std::scoped_lock lock(mMutexObsers, mMutexPos);
    mbBad = true;
    observations = mObservations;
    mObservations.clear();
  }
  
  for (auto& obs : observations) {
    KeyFrame* pKF = obs.first;
    pKF->EraseMapPoint(obs.second);
  }
  mpMap->EraseMapPoint(this);
} // SetBadFlag

bool MapPoint::isBad() {
    std::scoped_lock lock(mMutexObsers, mMutexPos);
    return mbBad;
} // isBad

void MapPoint::Replace(MapPoint* pMP) {
  if (pMP->mnId == this->mnId) {
    return;
  }

  // 替换visible数量和KeepAfterBA数量
  int nvisible, nkeepAfterBA;
  // 并获取观测帧
  std::map<KeyFrame*, size_t> observations;
  {
    std::scoped_lock lock(mMutexObsers, mMutexPos);
    observations = mObservations;
    nvisible = mnVisible;
    nkeepAfterBA = mnKeepAfterBA;
    mbBad = true;
    mObservations.clear();
    mpReplaced = pMP;
  }

  // 将pMP重新加入到观测帧关系中
  for (auto& obs : observations) {
    KeyFrame* pKF = obs.first;
    // 如果pKF还没有加入此地图点的观测，则加入
    if (!pMP->IsKeyFrameInObservations(KeyFramePtr(pKF))) {
      pKF->mvpMapPoints[obs.second] = pMP;
      pMP->AddObservation(pKF, obs.second);
    } 
    // 如果pKF已经加入了此地图点的观测，则删除当前索引对应的地图点
    else {
      pKF->EraseMapPoint(obs.second);
    }
  }

  // 更新pMP的visible数量和KeepAfterBA数量
  pMP->IncreaseVisible(nvisible);
  pMP->IncreaseKeepAfterBA(nkeepAfterBA);
  // 更新地图点
  pMP->UpdateNormalAndDepth();
  pMP->ComputeDistinctiveDescriptors();

  // 从地图中删除当前地图点
  mpMap->EraseMapPoint(this);

} // Replace

// --------------------- 观测相关 ---------------------

void MapPoint::AddObservation(KeyFrame* pKF, const size_t& idx) {
    // 互斥锁
    std::unique_lock<std::mutex> lock(mMutexObsers);
    mObservations[pKF] = idx;
} // AddObservation

void MapPoint::EraseObservation(KeyFrame* pKF) {
  bool bBad = false;
  {
    std::scoped_lock lock(mMutexObsers);
    if (mObservations.count(pKF)) {
      mObservations.erase(pKF);
      if (mpRefKF == pKF) {
        mpRefKF = mObservations.begin()->first;
      }
      if (mObservations.size() <=2 ) {
        bBad = true;
      }
    } // mObservations.count(pKF) > 0
  } // scoped_lock
  if (bBad) {
    SetBadFlag();
  }
} // EraseObservation

std::map<KeyFrame*, size_t> MapPoint::GetObservations() {
  std::scoped_lock lock(mMutexObsers);
  return mObservations;
} // GetObservations

int MapPoint::Observations() {
  std::scoped_lock lock(mMutexObsers);
  return mObservations.size();
} // Observations

bool MapPoint::IsKeyFrameInObservations(KeyFramePtr pKF) {
  std::scoped_lock<std::mutex> lock(mMutexObsers);
  return (mObservations.count(pKF.get()) > 0);
}

void MapPoint::IncreaseVisible(int n) {
  std::scoped_lock<std::mutex> lock(mMutexObsers);
  mnVisible += n;
} // IncreaseVisible

void MapPoint::IncreaseKeepAfterBA(int n) {
  std::scoped_lock<std::mutex> lock(mMutexObsers);
  mnKeepAfterBA += n;
} // IncreaseKeepAfterBA

float MapPoint::GetFoundKeepRatio() {
  std::scoped_lock<std::mutex> lock(mMutexObsers);
  return static_cast<float>(mnKeepAfterBA) / mnVisible;
} // GetFoundKeepRatio

void MapPoint::UpdateNormalAndDepth() {

  // -------------------- Step 1 计算观测帧到此地图点的向量（观测方向）之和 --------------------
  std::map<KeyFrame*, size_t> observations;
  KeyFrame* pRefKF;
  Point3dT Pos;
  {
    std::unique_lock<std::mutex> lock1(mMutexObsers);
    std::unique_lock<std::mutex> lock2(mMutexPos);
    if (mbBad) {
      return;
    }
    observations = mObservations;
    pRefKF = mpRefKF;
    Pos = mPos;
  }

  Normal3dT normal(0, 0, 0);
  int n = 0;
  for (auto& obs : observations) {
    KeyFrame* pKF = obs.first;
    if (!pKF->isBad()) {
      Point3dT Owi = pKF->GetCameraCenter();
      Point3dT normali = Pos - Owi;
      normal += normali.normalized();;
      n++;
    }
  }

  {
    std::unique_lock<std::mutex> lock(mMutexPos);
    mNormalVector = normal / n;
  }

  // -------------------- Step 2 根据参考关键帧，计算此地图点的最大最小可观测深度 --------------------
  // ORB—SLAM2及之后计算最大最小深度的方法与ORB-SLAM1不同，具体可以参考这篇文章
  // https://zhuanlan.zhihu.com/p/618342022
  // 最大最小深度的计算是根据关键点所处图像金字塔的层级计算得到的
  // predefine： 金字塔一共8层，从0-7，0层是最大的，7层是最小的。金字塔scale 为1.2
  // minDistance
  // 假设此地图点到参考关键帧的距离为d，位于金姿态第0层（level=0）
  // 则，当相机往前推进 level + 1 - nlevel = 7层时，此地图点刚好会被放大到金字塔第7层（最高层）
  // 此时的距离为：d * scale^(level + 1 - nlevel)
  // maxDistance
  // 假设此地图点到参考关键帧的距离为d，位于金姿态第7层（level=7）
  // 则，当相机往后移动 level 层时，此地图点刚好会被金字塔第0层（最低层）捕捉
  // 此时的距离为：d * scale^level

  Point3dT PC = Pos - pRefKF->GetCameraCenter();
  const float dist = PC.norm();
  int level; // 此地图点所处的金字塔层级
  level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
  const float levelScaleFactor = pRefKF->mpORBextractor->GetScaleFactors()[level];
  const int nLevels = pRefKF->mpORBextractor->GetLevels();
  {
    std::unique_lock<std::mutex> lock(mMutexPos);
    mfMaxDistance = dist * levelScaleFactor;
    mfMinDistance = mfMaxDistance / pRefKF->mpORBextractor->GetScaleFactors()[nLevels - 1];
  }
  
} // UpdateNormalAndDepth

Point3dT MapPoint::GetNormal() {
  std::unique_lock<std::mutex> lock(mMutexPos);
  return mNormalVector;
} // GetNormal

int MapPoint::PredictScale(const float& dist, const float& scaleFactor, const int& nLevels) {
  // 最大的观测距离对应该地图点在金字塔第0层
  // dist * scale ^{i} = maxDistance
  // i = log(dist / maxDistance) / log(scale)

  float ratio;
  {
    std::scoped_lock<std::mutex> lock(mMutexPos);
    ratio = mfMaxDistance / dist;
  }

  int nScale = ceil(log(ratio) / log(scaleFactor));
  if (nScale < 0) {
    nScale = 0;
  } else if (nScale >= nLevels) {
    nScale = nLevels - 1;
  }
  return nScale;
} // PredictScale

float MapPoint::GetMinDistanceInvariance() {
  std::scoped_lock<std::mutex> lock(mMutexPos);
  return 0.8f * mfMinDistance;
} // GetMinDistanceInvariance
float MapPoint::GetMaxDistanceInvariance() {
  std::scoped_lock<std::mutex> lock(mMutexPos);
  return 1.2f * mfMaxDistance;
} // GetMaxDistanceInvariance


// --------------------- 描述子相关 ---------------------

void MapPoint::ComputeDistinctiveDescriptors(){
  // -------------------- Step 1 从观测帧中收集描述子 --------------------
  std::vector<cv::Mat> vDescriptors;
  std::map<KeyFrame*, size_t> observations;
  {
    std::unique_lock<std::mutex> lock1(mMutexObsers);
    if (mbBad) {
      return;
    }
    // 拷贝观测帧，避免锁的时间过长
    observations = mObservations;
  }

  if (observations.empty()) {
    return;
  }

  vDescriptors.reserve(observations.size());

  for (auto& obs : observations) {
    KeyFrame* pKF = obs.first;
    if (!pKF->isBad()) {
      vDescriptors.push_back(pKF->mDescriptors.row(obs.second));
    }
  }
  if(vDescriptors.empty()) {
    return;
  }

  // -------------------- Step 2 计算描述子之间的距离-Brute Force --------------------
  const size_t N = vDescriptors.size();
  float Distances[N][N];
  for (size_t i = 0; i < N; i++) {
    Distances[i][i] = 0;
    for (size_t j = i + 1; j < N; j++) {
      int distij = DBoW2::FORB::distance(vDescriptors[i], vDescriptors[j]);
      Distances[i][j] = distij;
      Distances[j][i] = distij;
    }
  }

  // -------------------- Step 3 选择到其他点距离中位数最小的描述子作为地图点的描述子 --------------------
  int BestMedian = INT_MAX;
  int BestIdx = 0;
  for (size_t i = 0; i < N; i++) {
    std::vector<int> vDists(Distances[i], Distances[i] + N);
    sort(vDists.begin(), vDists.end());
    int median = vDists[0.5 * (N - 1)];

    if (median < BestMedian) {
      BestMedian = median;
      BestIdx = i;
    }
  }

  {
    std::unique_lock<std::mutex> lock(mMutexObsers);
    mDescriptor = vDescriptors[BestIdx].clone();
  }

} // ComputeDistinctiveDescriptors

cv::Mat MapPoint::GetDescriptor() {
  std::unique_lock<std::mutex> lock(mMutexObsers);
  return mDescriptor.clone();
} // GetDescriptor


}  // namespace ORB_SLAM_Tracking
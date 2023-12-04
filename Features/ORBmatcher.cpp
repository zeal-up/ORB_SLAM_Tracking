#include "Features/ORBmatcher.hpp"

namespace ORB_SLAM_Tracking {

const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_LOW = 50;
const int ORBmatcher::HISTO_LENGTH = 30;

ORBmatcher::ORBmatcher(float nnratio, bool checkOri) : mfNNratio(nnratio), mbCheckOrientation(checkOri) {}

int ORBmatcher::SearchForInitialization(Frame& F1, Frame& F2, std::vector<int>& vnMatches12,
                                        int windowSize) {
  int nmatches = 0;
  vnMatches12 = std::vector<int>(F1.mvKeysUn.size(), -1);

  // 初始化角度直方图
  std::vector<int> rotHist[HISTO_LENGTH];
  for (int i = 0; i < HISTO_LENGTH; i++) {
    rotHist[i].reserve(500);
  }
  //! 原作者代码是 const float factor = 1.0f/HISTO_LENGTH; 是错误的，更改为下面代码
  // const float factor = 1.0f/HISTO_LENGTH;
  const float factor = HISTO_LENGTH/360.0f;
  

  // 开始进行匹配 —— 对于第一帧中的每个特征点，寻找第二帧中的匹配点
  // 记录F2中每个点被匹配的最近距离
  std::vector<int> vMatchedDistance(F2.N, INT_MAX);
  // 记录F2中每个点被匹配到的initFrame中的点的索引
  std::vector<int> vnMatches21(F2.N, -1);

  // 统计各种不合格匹配点的数量
  int invalidMatchByDistance = 0, invalidMatchByRatio = 0, invalidMatchByOrientation = 0;

  // 从第一帧中的每个特征点开始寻找匹配点
  for (size_t i1 = 0, iend1 = F1.N; i1 < iend1; i1++) {
    cv::KeyPoint kp1 = F1.mvKeysUn[i1];
    int level1 = kp1.octave;
    if (level1 > 0) continue;  // 只搜索最精细层的特征点,其他层的特征点直接跳过

    // Step 1 --------------------------------------------------------------
    // 从F2中找出点kp1所在半径范围内的特征点
    // 这里minLevel和maxLevel都是0，所以只会搜索最精细层的特征点
    std::vector<size_t> vIndices2 = F2.GetFeaturesInArea(kp1.pt.x, kp1.pt.y, windowSize, level1, level1);
    if (vIndices2.empty()) continue;

    // Step 2 --------------------------------------------------------------
    // 遍历F2中kp1所在区域内的特征点，找出最近的点

    // 取出点kp1的描述子
    cv::Mat d1 = F1.mDescriptors.row(i1);

    int bestDist = INT_MAX; // 最近距离
    int bestDist2 = INT_MAX; // 次近距离
    int bestIdx2 = -1; // 最近距离对应的特征点索引(在F2中)

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

  } // 遍历F1中的每个特征点

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
  } // mbCheckOrientation

  // print statistics
  std::cout << "SearchForInitialization done ----------------------" << std::endl;
  std::cout << "invalidMatchByDistance: " << invalidMatchByDistance << std::endl;
  std::cout << "invalidMatchByRatio: " << invalidMatchByRatio << std::endl;
  std::cout << "invalidMatchByOrientation: " << invalidMatchByOrientation << std::endl;

  return nmatches;
} // SearchForInitialization

void ORBmatcher::ComputeThreeMaxima(std::vector<int>* histo, const int L, int& ind1, int& ind2, int& ind3) {
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
} // ComputeThreeMaxima
}  // namespace ORB_SLAM_Tracking
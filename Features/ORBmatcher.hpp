#pragma once

#include "SlamTypes/Frame.hpp"

namespace ORB_SLAM_Tracking {

class ORBmatcher {
 public:
  /**
   * @brief Construct a new ORBmatcher object
   *
   * @param nnratio 寻找匹配点的时候，最小距离/次小距离的比值需要小于这个值才认为是匹配点
   * @param checkOri 通过旋转角直方图来判断匹配点的方向是否一致
   */
  ORBmatcher(float nnratio = 0.6, bool checkOri = true);

  /**
   * @brief 用于前两帧初始化时的关键点匹配，只会搜索图像金字塔的最底层的特征的（最精细层）
   * 对于F1中的每个特征点，寻找F2中的匹配点
   * 
   * 这里不单纯是对F1中的每个特征点找F2中最近的点，
   * 在F1中可能有多个点匹配到了F2中的同一个点，
   * 这里会选择最近的点作为匹配点，其他点不认为是匹配点
   * 
   * 这里在实现的时候还是非常巧妙的，在做F1 -> F2的匹配的时候，
   * 同时维护一个F2 -> F1的匹配关系和最近距离，这个匹配关系和最近距离如果
   * 遇到一个更近的点就会被更新
   * 
   *
   * @param F1 原始帧
   * @param F2 当前帧
   * @param vnMatches12 第一帧中特征点在第二帧中的匹配点索引
   * @param windowSize 搜索窗口的大小
   * @return int 返回匹配点的数量
   */
  int SearchForInitialization(Frame &F1, Frame &F2, std::vector<int> &vnMatches12, int windowSize = 100);


 private:
 
  /**
   * 计算直方图中的三个最大值的索引。
   * 这里还会检查max2和max3与max1的差距是否太大（小于0.1），如果太大则认为第二和第三大的值不可靠，直接忽略
   * 
   * @param histo 直方图数组
   * @param L 直方图数组的长度
   * @param ind1 存储第一个最大值的索引
   * @param ind2 存储第二个最大值的索引
   * @param ind3 存储第三个最大值的索引
   */
  void ComputeThreeMaxima(std::vector<int>* histo, const int L, int& ind1, int& ind2, int& ind3);

 public:
  // 旋转角直方图的长度（分辨率）
  static const int HISTO_LENGTH;
  // 特征距离的阈值 —— 较严格阈值，使用在初始化/BoW匹配/三角化/关键帧的重投影搜索
  static const int TH_LOW;
  // 特征距离的阈值 —— 较宽松阈值，使用在重投影搜索等
  static const int TH_HIGH;
 private:
  float mfNNratio;          // 最小距离/次小距离的比值需要小于这个值才认为是匹配点
  bool mbCheckOrientation;  // 通过旋转角直方图来判断匹配点的方向是否一致
};
}  // namespace ORB_SLAM_Tracking
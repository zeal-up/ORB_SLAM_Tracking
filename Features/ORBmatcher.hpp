#pragma once

#include "SlamTypes/Frame.hpp"
#include "SlamTypes/KeyFrame.hpp"

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
   * @param vbPreMatched
   * 这是一个比较重要的参数，保存F1的关键点在F2中的初始位置。一开始这个向量直接用F1的关键点初始化，但是在搜索过程中，
   * 对于找到匹配点的关键点，会更新到vbPreMatched中，这样如果当前初始化失败，F2变成下一帧，那么关键点的初始位置会更接近准确的位置，这样才能够
   * 正常进行匹配
   * @param vnMatches12 第一帧中特征点在第二帧中的匹配点索引
   * @param windowSize 搜索窗口的大小
   * @return int 返回匹配点的数量
   */
  int SearchForInitialization(Frame& F1, Frame& F2, std::vector<cv::Point2f>& vbPreMatched,
                              std::vector<int>& vnMatches12, int windowSize = 100);

  /**
   * @brief 通过词袋模型进行特征点匹配。在curF中寻找与refF匹配的特征点。
   * 注意这里是从关键帧到普通帧的匹配，用在TrackReferenceKeyFrame()函数中，此时普通帧没有地图点，因此是查找curF中匹配的描述子
   *
   * @param refKF 参考关键帧
   * @param curF 当前帧
   * @param vpMatches12 存储匹配点的指针的向量, vpMatches12[i]表示curF中与refKF中第i个地图点匹配的点
   * @return int 返回匹配点的数量
   */
  int SearchByBoW(KeyFramePtr refF, FramePtr curF, std::vector<MapPoint*>& vpMatches21);

  /**
   * @brief 将上一帧的地图点投影到当前帧进行匹配搜索。
   * 用在跟踪上一帧（TrackWithMotionModel）
   *
   * @param curF
   * @param lastF
   * @param radius 搜索半径（像素）
   * @return int 匹配点对数量
   */
  int SearchByProjection(FramePtr curF, const FramePtr lastF, const float radius);

  // 将局部地图点投影到当前帧进行匹配搜索. 搜索半径在内部逻辑中会被设置为4.0
  // 如果设置expand, 会等比例扩大搜索半径
  int SearchByProjection(FramePtr curF, const std::vector<MapPoint*>& vpMapPoints,
                         const float expand = 1.0);

  // 使用词袋模型，对KF1和KF2进行特征点匹配搜索
  // SearchForInitialization()虽然也是搜索之后用来做三角化，但是只在最底层的金字塔层进行搜索
  // 同时，SearchForInitialization()没有用词袋模型进行搜索，而是直接用WindowSearch()进行搜索
  // 这里会跳过KF1/KF2中已经有了地图点的特征点，也就是说，三角化是对KF1和KF2中都没有地图点的特征点进行的
  // 另外，这里还会检查是否满足极线约束，因此没有像SearchForInitialization()那样需要满足对称匹配关系（
  // KF1在KF2找到的最近点，同时也要满足KF2在KF1找到的最近点）
  int SearchForTriangulation(KeyFramePtr pKF1, KeyFramePtr pKF2, Eigen::Matrix3d F12,
                             std::vector<cv::KeyPoint>& vMatchedKeysUn1,
                             std::vector<cv::KeyPoint>& vMatchedKeysUn2,
                             std::vector<std::pair<size_t, size_t>>& vMatchedIndices);

  // 投影搜索，在pKF中找到与vpMapPoints中地图点匹配的特征点
  // 如果此特征点没有对应的地图点，则将找到的3D点加入pKF
  // 如果此特征点已经有了对应的地图点，则比较两个地图点的观测数，用观测数多的地图点replace观测数少的地图点
  // @return 返回成功融合的特征点数量
  int Fuse(KeyFramePtr pKF, const std::vector<MapPoint*>& vpMapPoints, const float th = 3.0);

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

  /**
   * @brief 根据论文ORB-SLAM1中的V-D章节，将地图点投影到帧上进行3D-2D点匹配
   * 1. 视野范围：投影点需要在图像范围内;
   * 2. 距离范围：地图点到相机中心的距离需要在地图点的最大最小距离范围内;
   * 3. 视角限制(默认60度内)：地图点与相机中心连线与地图点的平均观测方向的夹角需要在视角限制内
   * 4. 特征点层级限制：地图点会根据距离预测一个特征点层级，2D点需要在这个层级的前后层级内
   * 
   * @param tgtF 
   * @param vpMapPoints 
   * @param th 
   * @param bestDistIdx out: 存储地图点到帧的最近距离以及匹配的2D点索引
   * @param secondBestDistIdx  out: 存储地图点到帧的次近距离以及匹配的2D点索引
   * @param bestLevel12 out: 存储地图点到帧的最近2D点和次近2D点的特征点层级
   */
  void SearchByProjection_(Frame* tgtF, const std::vector<MapPoint*>& vpMapPoints,
                          const float th, std::vector<std::pair<int, int>>& bestDistIdx, 
                          std::vector<std::pair<int, int>>& secondBestDistIdx,
                          std::vector<std::pair<int, int>>& bestLevel12);

 public:
  // 旋转角直方图的长度（分辨率）
  static const int HISTO_LENGTH;
  // 特征距离的阈值 —— 较严格阈值，使用在初始化/BoW匹配/三角化/关键帧的重投影搜索
  static const int TH_LOW;
  // 特征距离的阈值 —— 较宽松阈值，使用在重投影搜索等
  static const int TH_HIGH;

 private:
  float mfNNratio;  // 最小距离/次小距离的比值需要小于这个值才认为是匹配点
  bool mbCheckOrientation;  // 通过旋转角直方图来判断匹配点的方向是否一致
};
}  // namespace ORB_SLAM_Tracking
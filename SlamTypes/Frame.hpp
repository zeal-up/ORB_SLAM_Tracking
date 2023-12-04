#pragma once

#include <opencv2/opencv.hpp>

#include "Features/ORBVocabulary.hpp"
#include "Features/ORBextractor.hpp"

#include "SlamTypes/BasicTypes.hpp"

namespace ORB_SLAM_Tracking {

// 将图片切分成FRAME_GRID_ROWS * FRAME_GRID_COLS个区域
// 这样做特征点匹配的时候只需要在区域内进行搜索
// 同时需要一个数组，存储每个区域内的特征点索引
#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

class MapPoint;

class Frame {
 public:
  Frame();
  Frame(const Frame& frame);
  Frame(cv::Mat& im, const double& timestamp, ORBextractor* extractor, ORBVocabulary* voc, cv::Mat& K,
        cv::Mat& distCoef);

  
  /**
   * 计算特征点被划分到哪个区域中
   * 
   * @param kp 特征点坐标.
   * @param posX 网格的X索引 (output parameter).
   * @param posY 网格的Y索引 (output parameter).
   * @return 如果特征点超出图像范围，则返回false
   */
  bool PosInGrid(const cv::KeyPoint& kp, int& posX, int& posY);

  /**
   * @brief 设置位姿
   *
   * @param Tcw 世界坐标系到相机坐标系的变换矩阵
   */
  void SetPose(const PoseT& Tcw);

  /**
   * @brief 获取指定区域内的特征点索引
   * 
   * @param x 区域中心点的 x 坐标
   * @param y 区域中心点的 y 坐标
   * @param r 区域的半径
   * @param minLevel 最小金字塔层级（可选，默认为 -1）
   * @param maxLevel 最大金字塔层级（可选，默认为 -1）
   * @return std::vector<size_t> 包含指定区域内特征点索引的向量;如果区域内没有特征点,或者给入的x,y超出图像范围，则返回空向量
   */
  std::vector<size_t> GetFeaturesInArea(const float& x, const float& y, const float& r, const int minLevel = -1,
                                        const int maxLevel = -1) const;

 public:
  ORBVocabulary* mpORBvocabulary;
  ORBextractor* mpORBextractor;

  // Frame Image
  cv::Mat im;
  // Frame timestamp
  double mTimestamp;

  // Camera intrinsics && distortion coefficients
  cv::Mat mK;
  static float fx, fy, cx, cy;
  cv::Mat mDistCoef;

  // ORB related
  int N;  // Number of keypoints
  // vector of keypoints (original - for visualization))
  std::vector<cv::KeyPoint> mvKeys;
  // vector of keypoints (undistorted - used for slam)
  std::vector<cv::KeyPoint> mvKeysUn;
  // BoW vector
  DBoW2::BowVector mBowVec;
  // Feature vector (store the direct-index)
  DBoW2::FeatureVector mFeatVec;
  // ORB descriptors, each row associated to a keypoint
  cv::Mat mDescriptors;

  // Current and Next Frame id
  // Next Frame id; static member to share between all frames
  static long unsigned int nNextId;  
  // Current Frame id
  long unsigned int mnId;            

  // 地图点相关
  std::vector<MapPoint*> mvpMapPoints;  // 地图点 —— 存储的是指针，指向地图中的3D点
  std::vector<bool> mvbOutlier;         // 是否为外点 —— 在BA后会将一些点标记为外点

  // 将图片切分成FRAME_GRID_ROWS * FRAME_GRID_COLS个区域
  // 这样做特征点匹配的时候只需要在区域内进行搜索
  // 同时需要一个数组，存储每个区域内的特征点索引
  std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

  // static member - 静态变量，主要是一些相机、图像参数 -------------------------
  // 去畸变后的图像范围
  static int mnMinX;
  static int mnMaxX;
  static int mnMinY;
  static int mnMaxY;

  // 特征点被划分到不同的图像区域中，这样在进行投影-匹配时可以加速
  // x_coor * mfGridElementWidthInv = 区域的列索引
  static float mfGridElementWidthInv;
  // y_coor * mfGridElementHeightInv = 区域的行索引
  static float mfGridElementHeightInv;

  // 初始化第一帧时会计算上面这些参数，后续帧不再计算
  static bool mbInitialComputations;
  // ---------------------------------------------------------------------------

  // 位姿 -- 世界坐标系到相机坐标系的变换矩阵
  PoseT mTcw;
  bool mbPoseSet = false;


 private:
  /**
   * @brief 使用畸变系数对特征点进行去畸变
   *
   */
  void UndistortKeyPoints();

  /**
   * @brief 计算图像去畸变后的边界;
   * 使用原始图像的边界点，进行去畸变，得到去畸变后的边界点
   * 然后使用去畸变后的边界点计算mnMinX, mnMaxX, mnMinY, mnMaxY
   *
   */
  void ComputeImageBounds();
};

}  // namespace ORB_SLAM_Tracking
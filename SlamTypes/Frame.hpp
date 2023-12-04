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
  // 因为有锁（没法进行拷贝），所以没法自动生成默认的赋值运算符
  Frame& operator=(const Frame& frame) = delete;

  int GetId() const { return mnId; }
  static void Reset() { nNextId = 0; }

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
	 * @brief Get the Pose object
	 * 
	 * @return PoseT Tcw 世界坐标系到相机坐标系的变换矩阵
	 */
  PoseT GetPose();

  // 获取当前帧的相机中心在世界坐标系下的坐标
  // Tcw 是向量从世界坐标系到相机坐标系的变换矩阵
  // 要求相机中心在世界坐标系中的坐标：Ow = -R^T * t
  // 等价于：Ow = Tcw.inverse().translation()
  Point3dT GetCameraCenter();

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

  void ComputeBoW();

  // ----------------------- 地图点相关 --------------------------------------------

  /**
   * @brief 判断地图点是否在视野内：
   * 1. 视野范围：投影点需要在图像范围内;
   * 2. 距离范围：地图点到相机中心的距离需要在地图点的最大最小距离范围内;
   * 3. 视角限制(默认60度内)：地图点与相机中心连线与地图点的平均观测方向的夹角需要在视角限制内
   * 
   * 这里的判断基本是根据ORB-SLAM1论文中的V-D章节
   * 
   * @param pMP 
   * @param viewingCosLimit 观察角度的余弦值 相机光心与地图点连线，与地图点的平均观测方向的夹角
   * @param outU 输出：地图点在图像上的投影坐标 u
   * @param outV 输出：地图点在图像上的投影坐标 v
   * @param outViewCos 输出：相机光心与地图点连线，与地图点的平均观测方向的夹角的余弦值
   * @param outLevel 输出：地图点所在的金字塔层级
   * @return true 
   * @return false 
   */
  bool isInFrustum(MapPoint* pMP, float viewingCosLimit, float& outU, float& outV,
                   float& outViewCos, int& outLevel);

  bool isInFrustum(MapPoint* pMP, float viewingCosLimit) {
    float u, v, viewCos;
    int level;
    return isInFrustum(pMP, viewingCosLimit, u, v, viewCos, level);
  }

  // 获取有效的关联地图点数量：1.非空; 2.不是外点; 3.观测数>0
  int GetValidMapPointCount();

  // 将序号为idx的地图点置为空
  void EraseMapPoint(const size_t& idx);

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
  // Current and Next Frame id
  // Next Frame id; static member to share between all frames
  static long unsigned int nNextId;  
  // Current Frame id
  long unsigned int mnId;

	// 互斥锁
	std::mutex mMutexPose;
  std::mutex mMutexFeatures;

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

typedef std::shared_ptr<Frame> FramePtr;

}  // namespace ORB_SLAM_Tracking
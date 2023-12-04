#include "SlamTypes/Frame.hpp"

namespace ORB_SLAM_Tracking {
float Frame::fx, Frame::fy, Frame::cx, Frame::cy;
bool Frame::mbInitialComputations = true;
int Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;
long unsigned int Frame::nNextId = 0;

Frame::Frame() {}

// copy constructor
Frame::Frame(const Frame& frame) {
    im = frame.im.clone();
    mTimestamp = frame.mTimestamp;
    mpORBextractor = frame.mpORBextractor;
    mpORBvocabulary = frame.mpORBvocabulary;
    mK = frame.mK.clone();
    mDistCoef = frame.mDistCoef.clone();
    mvKeys = frame.mvKeys;
    mDescriptors = frame.mDescriptors.clone();
    N = frame.N;
    mvKeysUn = frame.mvKeysUn;
    mnId = frame.mnId;
    mvbOutlier = frame.mvbOutlier;
    mvpMapPoints = frame.mvpMapPoints;

    for (int i = 0; i < FRAME_GRID_COLS; i++) {
        for (int j = 0; j < FRAME_GRID_ROWS; j++) {
            mGrid[i][j] = frame.mGrid[i][j];
        }
    }

    // set pose
    if (frame.mbPoseSet) {
        SetPose(frame.mTcw);
    }
}

Frame::Frame(cv::Mat& im, const double& timestamp, ORBextractor* extractor, ORBVocabulary* voc, cv::Mat& K,
             cv::Mat& distCoef)
    : im(im), mTimestamp(timestamp), mpORBextractor(extractor), mpORBvocabulary(voc), mK(K), mDistCoef(distCoef) {
  // 当第一帧时计算一些静态变量 —— 主要是一些图像参数
  if (mbInitialComputations) {
    ComputeImageBounds();
    mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);
    mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY);

    fx = mK.at<float>(0, 0);
    fy = mK.at<float>(1, 1);
    cx = mK.at<float>(0, 2);
    cy = mK.at<float>(1, 2);

    mbInitialComputations = false;
  }

  // 提取ORB特征 —— 调用ORBextractor::operator()()函数
  std::vector<int> unused{0,0};
  (*mpORBextractor)(im, cv::Mat(), mvKeys, mDescriptors, unused);
  N = mvKeys.size();
  if (mvKeys.empty()) return;

  // 对特征点去畸变
  UndistortKeyPoints();
  N = mvKeysUn.size();

  // 当前帧ID —— 同时将nNextId加1
  mnId = nNextId++;

  // 将特征点分配到不同的区域中
  for (size_t i = 0; i < mvKeysUn.size(); i++) {
    const cv::KeyPoint& kp = mvKeysUn[i];
    int nGridPosX, nGridPosY;
    if (PosInGrid(kp, nGridPosX, nGridPosY)) {
      mGrid[nGridPosX][nGridPosY].push_back(i);
    }
  }

  // 初始化其他成员变量
  mvbOutlier = std::vector<bool>(N, false);
  mvpMapPoints = std::vector<MapPoint*>(N, static_cast<MapPoint*>(NULL));
}

void Frame::SetPose(const PoseT& Tcw) {
  mTcw = Tcw;
  mbPoseSet = true;
}

bool Frame::PosInGrid(const cv::KeyPoint& kp, int& posX, int& posY) {
  posX = round((kp.pt.x - mnMinX) * mfGridElementWidthInv);
  posY = round((kp.pt.y - mnMinY) * mfGridElementHeightInv);

  // 特征点超出图像范围
  if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS) {
    return false;
  }

  return true;
}

void Frame::ComputeImageBounds() {
  if (mDistCoef.at<float>(0) != 0.0) {
    cv::Mat mat(4, 2, CV_32F);
    // 左上角
    mat.at<float>(0, 0) = 0.0;
    mat.at<float>(0, 1) = 0.0;
    // 右上角
    mat.at<float>(1, 0) = im.cols;
    mat.at<float>(1, 1) = 0.0;
    // 左下角
    mat.at<float>(2, 0) = 0.0;
    mat.at<float>(2, 1) = im.rows;
    // 右下角
    mat.at<float>(3, 0) = im.cols;
    mat.at<float>(3, 1) = im.rows;

    // 去畸变
    mat = mat.reshape(2);  // channel 改为2，shape = （4,1）
    cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
    mat = mat.reshape(1);  // channel 改为1，shape = （4,2）

    // 计算去畸变后的图像范围
    mnMinX = std::min(mat.at<float>(0, 0), mat.at<float>(2, 0));
    mnMaxX = std::max(mat.at<float>(1, 0), mat.at<float>(3, 0));
    mnMinY = std::min(mat.at<float>(0, 1), mat.at<float>(1, 1));
    mnMaxY = std::max(mat.at<float>(2, 1), mat.at<float>(3, 1));
  } else {
    // 如果没有畸变，则图像范围就是图像的大小
    mnMinX = 0;
    mnMaxX = im.cols;
    mnMinY = 0;
    mnMaxY = im.rows;
  }
}

void Frame::UndistortKeyPoints() {
  if (mDistCoef.at<float>(0) == 0.0) {
    mvKeysUn = mvKeys;
    return;
  }

  // 将vector<cv::KeyPoint>转换成cv::Mat
  cv::Mat mat(N, 2, CV_32F);
  for (int i = 0; i < N; i++) {
    mat.at<float>(i, 0) = mvKeys[i].pt.x;
    mat.at<float>(i, 1) = mvKeys[i].pt.y;
  }

  mat = mat.reshape(2);
  cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
  mat = mat.reshape(1);

  // 将cv::Mat转换成vector<cv::KeyPoint>
  mvKeysUn.resize(N);
  for (int i = 0; i < N; i++) {
    cv::KeyPoint kp = mvKeys[i];
    kp.pt.x = mat.at<float>(i, 0);
    kp.pt.y = mat.at<float>(i, 1);
    mvKeysUn[i] = kp;
  }
}

std::vector<size_t> Frame::GetFeaturesInArea(const float& x, const float& y, const float& r, const int minLevel,
                                             const int maxLevel) const {
  std::vector<size_t> vIndices;
  vIndices.reserve(N);
  int mMinCellX = std::max(0, (int)floor((x - mnMinX - r) * mfGridElementWidthInv));
  if (mMinCellX >= FRAME_GRID_COLS) return vIndices;

  int mMaxCellX = std::min((int)FRAME_GRID_COLS - 1, (int)ceil((x - mnMinX + r) * mfGridElementWidthInv));
  if (mMaxCellX < 0) return vIndices;

  int nMinCellY = std::max(0, (int)floor((y - mnMinY - r) * mfGridElementHeightInv));
  if (nMinCellY >= FRAME_GRID_ROWS) return vIndices;

  int nMaxCellY = std::min((int)FRAME_GRID_ROWS - 1, (int)ceil((y - mnMinY + r) * mfGridElementHeightInv));
  if (nMaxCellY < 0) return vIndices;

  bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

  for (int ix = mMinCellX; ix <= mMaxCellX; ix++) {
    for (int iy = nMinCellY; iy <= nMaxCellY; iy++) {
      // mGrid[ix][iy]存储的是区域内特征点的索引
      const std::vector<size_t>& vCell = mGrid[ix][iy];
      if (vCell.empty()) continue;

      for (size_t j = 0, jend = vCell.size(); j < jend; j++) {
        const cv::KeyPoint& kpUn = mvKeysUn[vCell[j]];
        if (bCheckLevels) {
          // 检查特征点的层级是否在指定范围内
          if (!(kpUn.octave >= minLevel && kpUn.octave <= maxLevel)) continue;
        }

        // 通过点坐标再次检查是否位于半径之内
        const float distx = kpUn.pt.x - x;
        const float disty = kpUn.pt.y - y;
        if (fabs(distx) < r && fabs(disty) < r) {
          vIndices.push_back(vCell[j]);
        }
      }
    }
  }

  return vIndices;

}  // function GetFeaturesInArea

}  // namespace ORB_SLAM_Tracking
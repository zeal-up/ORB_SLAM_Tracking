#include "Modules/LocalMapping.hpp"

#include <glog/logging.h>

#include <opencv2/core/eigen.hpp>
#include <unordered_set>

#include "Features/ORBmatcher.hpp"
#include "Optimization/Optimizer.hpp"
#include "SlamTypes/MapPoint.hpp"
#include "Utils/Converter.hpp"
#include "Utils/UtilsCV.hpp"

namespace ORB_SLAM_Tracking {

LocalMapping::LocalMapping(Map* pMap) : mpMap(pMap) {}

bool LocalMapping::AddKeyFrame(KeyFramePtr pKF) {
  LOG(INFO) << "Enter " << __FUNCTION__;

  // Step 1 : 进行新关键帧的各种预处理 ------------------------------------
  mpCurrentKF = pKF;
  ProcessNewKeyFrame();

  // Step 2 : 对近期增加的地图点（3个关键帧之内）进行筛选 ---------------------
  MapPointCulling();

  // Step 3 : 使用对极几何创建新的地图点 ------------------------------------
  CreateNewMapPoints();

  // Step 4 : 地图点冗余去除 --------------------------------
  SearchInNeighbors();

  // Step 5 : 局部BA优化 ---------------------------------------------------
  OptimizeLocalMap();

  // Step 6 : 关键帧剔除 ---------------------------------------------------
  KeyFrameCulling();

  LOG(INFO) << "Exit " << __FUNCTION__;
  return true;
}

void LocalMapping::ProcessNewKeyFrame() {
  mpCurrentKF->ComputeBoW();  // 计算当前关键帧的BoW向量

  for (size_t i = 0; i < mpCurrentKF->N; ++i) {
    MapPoint* pMP = mpCurrentKF->mvpMapPoints[i];
    if (!pMP || pMP->isBad()) continue;
    // 如果pMP还没有被当前关键帧观测到，则添加观测
    if (!pMP->IsKeyFrameInObservations(mpCurrentKF)) {
      pMP->AddObservation(mpCurrentKF.get(), i);
      pMP->UpdateNormalAndDepth();
      pMP->ComputeDistinctiveDescriptors();
    }
  }  // loop for mpCurrentKF's keypoints

  // 第一个关键帧，需要将地图点加入到mlpRecentAddedMapPoints中
  if (mpCurrentKF->GetId() == 1) {
    for (auto pMP : mpCurrentKF->mvpMapPoints) {
      if (pMP) {
        mlpRecentAddedMapPoints.push_back(pMP);
      }
    }
  }  // if first keyframe

  mpCurrentKF->UpdateConnections();
  mpMap->AddKeyFrame(mpCurrentKF.get());
}  // ProcessNewKeyFrame

void LocalMapping::MapPointCulling() {
  LOG(INFO) << "Enter " << __FUNCTION__;

  auto lit = mlpRecentAddedMapPoints.begin();
  const auto& nCurrentKFid = mpCurrentKF->GetId();
  int deleteByRatio = 0, deleteByFrameObs = 0;
  int acceptToMap = 0;
  while (lit != mlpRecentAddedMapPoints.end()) {
    MapPoint* pMP = *lit;
    if (pMP->isBad()) {
      lit = mlpRecentAddedMapPoints.erase(lit);
    } else if (pMP->GetFoundKeepRatio() < 0.25f) {
      pMP->SetBadFlag();
      lit = mlpRecentAddedMapPoints.erase(lit);
      deleteByRatio++;
    } else if ((nCurrentKFid - pMP->mnFirstKfId) >= 2 && pMP->Observations() <= 2) {
      pMP->SetBadFlag();
      lit = mlpRecentAddedMapPoints.erase(lit);
      deleteByFrameObs++;
    } else if ((nCurrentKFid - pMP->mnFirstKfId) >= 3) {
      lit = mlpRecentAddedMapPoints.erase(lit);
      acceptToMap++;
    } else {
      ++lit;
    }
  }

  LOG(INFO) << "Exit " << __FUNCTION__ << " deleteByRatio: " << deleteByRatio
            << " deleteByFrameObs: " << deleteByFrameObs << " acceptToMap: " << acceptToMap;
}  // MapPointCulling

void LocalMapping::CreateNewMapPoints() {
  LOG(INFO) << "Enter " << __FUNCTION__;

  // Step 1 : 获取当前关键帧的共视关键帧 ------------------------------
  std::vector<KeyFrame*> vpNeighKFs = mpCurrentKF->GetBestCovisibilityKeyFrames(20);

  ORBmatcher matcher(0.6, false);

  // Step 2 : 获取当前关键帧的参数 ------------------------------------
  const PoseT Tcw1 = mpCurrentKF->GetPose();
  // 当前帧的相机中心在世界坐标系下的坐标
  const Point3dT Oc1 = mpCurrentKF->GetCameraCenter();
  const float& fx1 = mpCurrentKF->fx;
  const float& fy1 = mpCurrentKF->fy;
  const float& cx1 = mpCurrentKF->cx;
  const float& cy1 = mpCurrentKF->cy;
  const float& invfx1 = 1.0 / fx1;
  const float& invfy1 = 1.0 / fy1;
  const RotationT Rcw1 = Tcw1.rotation();
  const RotationT Rwc1 = Rcw1.inverse();

  // 在KF1和KF2找到的特征点的层级比例不能太大
  const float ratioFactor = 1.5 * mpCurrentKF->mpORBextractor->GetScaleFactor();

  // 统计失败的数量
  int failedByBaseline = 0, failedByParallax = 0, failedByDepth = 0, failedByProjErr1 = 0,
      failedByProjErr2 = 0, failedByScale = 0;
  int nNew = 0;

  // Step 3 : 遍历共视关键帧，创建地图点 ------------------------------------------
  int nNewMPs = 0;
  for (const auto& pKF2 : vpNeighKFs) {
    // Step 3.1 : 计算KF1和KF2之间的baseilne, 不能太近 -------------------------------
    // baseline的计算根据KF2的平均深度动态计算
    // 如果KF2的平均深度较大，也就是物体离相机较远，则需要更大的baseline距离
    const Point3dT Oc2 = pKF2->GetCameraCenter();
    const Point3dT bBaseline = Oc2 - Oc1;
    const float baseline = bBaseline.norm();
    const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth();
    const float ratioBaselineDepth = baseline / medianDepthKF2;
    if (ratioBaselineDepth < 0.01) {
      failedByBaseline++;
      continue;
    }

    // Step 3.2 : 计算KF1和KF2之间的基础矩阵 -------------------------------------------
    Eigen::Matrix3d F12 = UtilsCV::ComputeF21(Tcw1.matrix(), pKF2->GetPose().matrix(),
                                              Converter::toMatrix3f(mpCurrentKF->mK),
                                              Converter::toMatrix3f(pKF2->mK));

    // Step 3.3 : 进行特征点匹配 ------------------------------------------------------
    std::vector<cv::KeyPoint> vMatchedKeysUn1, vMatchedKeysUn2;
    std::vector<std::pair<size_t, size_t>> vMatchedIndices;
    matcher.SearchForTriangulation(mpCurrentKF, KeyFramePtr(pKF2), F12, vMatchedKeysUn1,
                                   vMatchedKeysUn2, vMatchedIndices);

    // Step 3.4 : 遍历匹配点对，进行三角化 -----------------------------------------------
    const PoseT Tcw2 = pKF2->GetPose();
    Eigen::Matrix4d K1 = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d K2 = Eigen::Matrix4d::Identity();
    K1.block<3, 3>(0, 0) = Converter::toMatrix3f(mpCurrentKF->mK);
    K2.block<3, 3>(0, 0) = Converter::toMatrix3f(pKF2->mK);
    Eigen::Matrix4d projMatEi1 = K1 * Tcw1.matrix();
    Eigen::Matrix4d projMatEi2 = K2 * Tcw2.matrix();
    cv::Mat projMat1, projMat2;
    cv::eigen2cv(projMatEi1, projMat1);
    cv::eigen2cv(projMatEi2, projMat2);
    // 提取前3行即可
    projMat1 = projMat1.rowRange(0, 3);
    projMat2 = projMat2.rowRange(0, 3);
    projMat1.convertTo(projMat1, CV_32F);  // 3 x 4
    projMat2.convertTo(projMat2, CV_32F);  // 3 x 4
    std::vector<cv::Point2f> vP1, vP2;     // N x 2
    cv::Mat vP3Dw;                         // N x 3, world frame
    cv::Mat vP4Dw;                         // N x 4, world frame
    for (size_t i = 0, iend = vMatchedKeysUn1.size(); i < iend; ++i) {
      const auto& kp1 = vMatchedKeysUn1[i];
      const auto& kp2 = vMatchedKeysUn2[i];
      vP1.push_back(kp1.pt);
      vP2.push_back(kp2.pt);
    }
    cv::triangulatePoints(projMat1, projMat2, vP1, vP2, vP4Dw);

    // Step 3.5 : 遍历3D点，去掉不合格的点 ------------------------------------------------
    const float& fx2 = pKF2->fx;
    const float& fy2 = pKF2->fy;
    const float& cx2 = pKF2->cx;
    const float& cy2 = pKF2->cy;
    const float& invfx2 = 1.0 / fx2;
    const float& invfy2 = 1.0 / fy2;
    const RotationT Rcw2 = Tcw2.rotation();
    const RotationT Rwc2 = Rcw2.inverse();

    for (size_t idx = 0; idx < vMatchedIndices.size(); idx++) {
      if (vP4Dw.at<float>(3, idx) == 0) {
        continue;
      }
      const int idxInKF1 = vMatchedIndices[idx].first;
      const int idxInKF2 = vMatchedIndices[idx].second;
      const cv::KeyPoint& kp1 = vMatchedKeysUn1[idx];
      const cv::KeyPoint& kp2 = vMatchedKeysUn2[idx];
      cv::Mat x3D = vP4Dw.col(idx);
      x3D /= x3D.at<float>(3);
      x3D = x3D.t();
      const Point3dT p3Dw = Converter::toVector3f(x3D);

      // 过滤1 ：计算关键点于相机连线之间的夹角 -------------------------------------------
      Eigen::Vector3f xn1, xn2;  // 两个关键点的归一化平面坐标
      xn1 << (kp1.pt.x - cx1) * invfx1, (kp1.pt.y - cy1) * invfy1, 1;
      xn2 << (kp2.pt.x - cx2) * invfx2, (kp2.pt.y - cy2) * invfy2, 1;
      // 将归一化坐标转换到世界坐标系下
      const Point3dT ray1 = Rwc1 * xn1;
      const Point3dT ray2 = Rwc2 * xn2;
      // 计算两个向量的夹角余弦值
      const float cosParallaxRays = ray1.dot(ray2) / (ray1.norm() * ray2.norm());
      // 夹角大于90度或者小于1.14度的不要。arccos(0.9998) = 1.14593469(degree)
      if (cosParallaxRays < 0 || cosParallaxRays > 0.9998) {
        failedByParallax++;
        continue;
      }

      // 过滤2 ：检查3D点在在两个关键帧中深度>0 -------------------------------------------
      Point3dT p3Dc1 = Rcw1 * p3Dw + Tcw1.translation();
      if (p3Dc1.z() <= 0) {
        failedByDepth++;
        continue;
      }
      Point3dT p3Dc2 = Rcw2 * p3Dw + Tcw2.translation();
      if (p3Dc2.z() <= 0) {
        failedByDepth++;
        continue;
      }

      // 过滤3 ：检查重投影误差 -------------------------------------------------------------
      // 计算在图像1中的重投影误差
      float sigmaSauqre1 = mpCurrentKF->mpORBextractor->GetScaleSigmaSquares()[kp1.octave];
      float invz1 = 1.0 / p3Dc1.z();
      const float u1 = fx1 * p3Dc1.x() * invz1 + cx1;
      const float v1 = fy1 * p3Dc1.y() * invz1 + cy1;
      const float errX1 = u1 - kp1.pt.x;
      const float errY1 = v1 - kp1.pt.y;
      const float err1 = errX1 * errX1 + errY1 * errY1;
      if (err1 > 5.991 * sigmaSauqre1) {
        failedByProjErr1++;
        continue;
      }

      // 计算在图像2中的重投影误差
      float sigmaSauqre2 = pKF2->mpORBextractor->GetScaleSigmaSquares()[kp2.octave];
      float invz2 = 1.0 / p3Dc2.z();
      const float u2 = fx2 * p3Dc2.x() * invz2 + cx2;
      const float v2 = fy2 * p3Dc2.y() * invz2 + cy2;
      const float errX2 = u2 - kp2.pt.x;
      const float errY2 = v2 - kp2.pt.y;
      const float err2 = errX2 * errX2 + errY2 * errY2;
      if (err2 > 5.991 * sigmaSauqre2) {
        failedByProjErr2++;
        continue;
      }

      // 过滤4 ：检查3D点在两帧相机坐标系下的尺度是否一致 --------------------------------------
      // 实际距离的比值 ratioDist = dist1 / dist2
      // 关键点层级的比值 ratioOctave = scale1 / scale2
      // 这两个比值需要比较接近才符合要求
      // 前面定义阈值 ratioFactor = 1.5 * scaleFactor
      // 要求： 1/ratioFactor < ratioDist / ratioOctave < ratioFactor
      float dist1 = (p3Dw - Oc1).norm();
      float dist2 = (p3Dw - Oc2).norm();
      if (dist1 == 0 || dist2 == 0) {
        failedByScale++;
        continue;
      }
      float ratioDist = dist1 / dist2;
      float ratioOctave = mpCurrentKF->mpORBextractor->GetScaleFactors()[kp1.octave] /
                          pKF2->mpORBextractor->GetScaleFactors()[kp2.octave];
      if (ratioDist * ratioFactor < ratioOctave || ratioDist > ratioOctave * ratioFactor) {
        failedByScale++;
        continue;
      }

      // 通过过滤条件，正式接纳为地图点 -------------------------------------------------------
      MapPoint* pMP = new MapPoint(p3Dw, mpCurrentKF.get(), mpMap);
      pMP->AddObservation(mpCurrentKF.get(), idxInKF1);
      pMP->AddObservation(pKF2, idxInKF2);
      mpCurrentKF->AddMapPoint(pMP, idxInKF1);
      pKF2->AddMapPoint(pMP, idxInKF2);
      pMP->ComputeDistinctiveDescriptors();
      pMP->UpdateNormalAndDepth();
      mpMap->AddMapPoint(pMP);
      mlpRecentAddedMapPoints.push_back(pMP);
      nNew++;

    }  // 遍历3D点进行检查
  }    // 遍历共视关键帧

  // 打印统计信息
  LOG(INFO) << "\tIter " << vpNeighKFs.size() << " KeyFrames, " << failedByBaseline
            << " failed by baseline check";
  LOG(INFO) << "\tFailed by parallax: " << failedByParallax;
  LOG(INFO) << "\tFailed by depth: " << failedByDepth;
  LOG(INFO) << "\tFailed by proj err1: " << failedByProjErr1;
  LOG(INFO) << "\tFailed by proj err2: " << failedByProjErr2;
  LOG(INFO) << "\tFailed by scale: " << failedByScale;
  LOG(INFO) << "\tCreated " << nNew << " new MapPoints";
  return;
}

void LocalMapping::SearchInNeighbors() {
  LOG(INFO) << "Enter " << __FUNCTION__;

  // Step 1 : 获取当前关键帧的第二共视帧 ------------------------------
  std::vector<KeyFrame*> vpNeighKFs = mpCurrentKF->GetBestCovisibilityKeyFrames(20);
  std::vector<KeyFrame*> vpSecondNeighKFs;
  std::unordered_set<KeyFrame*> spFirstSecondNeighKFs;

  for (const auto& pNeighKF : vpNeighKFs) {
    if (pNeighKF->isBad()) continue;
    spFirstSecondNeighKFs.insert(pNeighKF);
    std::vector<KeyFrame*> vpSecondNeighsi = pNeighKF->GetBestCovisibilityKeyFrames(5);
    for (const auto& pSecondNeighKF : vpSecondNeighsi) {
      if (pSecondNeighKF->isBad() || pSecondNeighKF->GetId() == mpCurrentKF->GetId() ||
          spFirstSecondNeighKFs.count(pSecondNeighKF) > 0) {
        continue;
      }
      spFirstSecondNeighKFs.insert(pSecondNeighKF);
    }  // 遍历第二共视帧
  }    // 遍历第一共视帧

  // Step 2 : 将当前帧的地图点融合到第一第二共视帧 ------------------------------
  // 这里不单单是做3D-2D投影搜索匹配，匹配完之后如果两帧的对应特征点都有地图点，则会对比两个
  // 地图点的观测数量，用观测数量多的替换掉观测数量少的地图点
  // 这个策略也是在去除地图点冗余
  ORBmatcher matcher(0.6, true);
  auto& vpCurFMapPoints = mpCurrentKF->mvpMapPoints;
  for (auto& pNeighKF : spFirstSecondNeighKFs) {
    matcher.Fuse(KeyFramePtr(pNeighKF), vpCurFMapPoints, 3.0);
  }

  // Step 3 : 将第一第二共视帧的地图点融合到当前帧 ------------------------------
  // 这里的逻辑和上面的逻辑是一样的，只不过是将第一第二共视帧的地图点融合到当前帧

  // 获取第一第二共视帧的所有地图点（使用set去重）
  std::unordered_set<MapPoint*> spNeighMapPoints;
  for (const auto& pNeighKF : spFirstSecondNeighKFs) {
    if (!pNeighKF || pNeighKF->isBad()) continue;
    for (const auto& pMP : pNeighKF->mvpMapPoints) {
      if (!pMP || pMP->isBad()) continue;
      spNeighMapPoints.insert(pMP);
    }
  }
  // 将set转换为vector
  std::vector<MapPoint*> vpNeighMapPoints(spNeighMapPoints.begin(), spNeighMapPoints.end());
  // 将第一第二共视帧的地图点融合到当前帧
  matcher.Fuse(mpCurrentKF, vpNeighMapPoints, 3.0);

  // Step 4 : 更新地图点的参数（观测方向、描述子、深度） ------------------------------
  // 为什么这里只更新了当前帧的地图点，而没有更新第一第二共视帧的地图点？
  // 对于被Replace的地图点，在MapPoint::Replace(MapPoint*)函数中已经进行了地图点参数的更新
  // 这里其实是对那些原本是空的特征点位置，但是通过投影搜索匹配到了地图点的地图点进行更新
  for (auto& pMP : mpCurrentKF->mvpMapPoints) {
    if (!pMP || pMP->isBad()) continue;
    pMP->UpdateNormalAndDepth();
    pMP->ComputeDistinctiveDescriptors();
  }

  // Step 5 : 更新当前帧的共视关系 ------------------------------------------------------
  // 这里似乎忘记了对第一第二共视帧的共视关系进行更新了，不过第一第二共视帧的共视关系在这里的变化不大
  mpCurrentKF->UpdateConnections();

  LOG(INFO) << "Exit " << __FUNCTION__;

}  // SearchInNeighbors

void LocalMapping::OptimizeLocalMap() {
  LOG(INFO) << "Enter " << __FUNCTION__;

  if (mpMap->KeyFramesInMap() > 2) {
    Optimizer::LocalBundleAdjustment(mpCurrentKF, mpMap);
  }

  LOG(INFO) << "Exit " << __FUNCTION__;
}  // OptimizeLocalMap

void LocalMapping::KeyFrameCulling() {
  LOG(INFO) << "Enter " << __FUNCTION__;
  std::vector<KeyFrame*> vpLocalKFs = mpCurrentKF->GetBestCovisibilityKeyFrames(-1);

  for (const auto& pKF : vpLocalKFs) {
    if (!pKF || pKF->isBad() || pKF->GetId() == 0) continue;
    const auto& vpMapPoints = pKF->mvpMapPoints;

    int nRedundantObservations = 0;  // 能够被其他三帧及以上观测到的地图点数量
    int nMPs = 0;                    // 地图点总数量

    // 遍历地图点，查找能够被其他三帧及以上观测到的地图点
    for (size_t idxMP = 0, idxEnd = vpMapPoints.size(); idxMP < idxEnd; ++idxMP) {
      MapPoint* pMP = vpMapPoints[idxMP];
      if (!pMP || pMP->isBad()) continue;
      nMPs++;
      if (pMP->Observations() > 3) {
        const auto& vpObservations = pMP->GetObservations();
        int scaleLevel = pKF->mvKeysUn[idxMP].octave;
        int nObs = 0;  // 统计被其他帧观测到的次数
        for (const auto& pObsKFidx : vpObservations) {
          auto& pObsKF = pObsKFidx.first;
          if (!pObsKF || pObsKF->isBad()) continue;
          if (pObsKF->GetId() == pKF->GetId()) continue;

          // 查看特征点的层级是否接近
          int scaleLevelObs = pObsKF->mvKeysUn[pObsKFidx.second].octave;
          if (scaleLevelObs <= scaleLevel + 1) {
            nObs++;
            if (nObs >= 3) break;  // 已经有三帧观测到了，跳出检查
          }
        }  // 遍历当前地图点的观测帧
        if (nObs >= 3) {
          nRedundantObservations++;
        }
      }  // if pMP->Observations() > 3
    }    // 遍历地图点

    // 如果能够被其他三帧及以上观测到的地图点数量大于90%，则认为这个关键帧是多余的
    if (nRedundantObservations > 0.9 * nMPs) {
      mpMap->EraseKeyFrame(pKF);  // 从地图中删除此多余的关键帧
    }
  }  // 遍历共视关键帧

  LOG(INFO) << "Exit " << __FUNCTION__;
}

}  // namespace ORB_SLAM_Tracking
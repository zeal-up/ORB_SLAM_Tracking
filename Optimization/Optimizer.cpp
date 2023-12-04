#include "Optimization/Optimizer.hpp"

#include <unordered_set>
#include <glog/logging.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_sba.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include "Utils/Converter.hpp"

namespace ORB_SLAM_Tracking {

// typedef g2o::BlockSolver<g2o::BlockSolverTraits<-1, -1>> SlamBlockSolver;
typedef g2o::BlockSolver_6_3 SlamBlockSolver;
typedef g2o::LinearSolverEigen<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;

void Optimizer::GlobalBundleAdjustment(Map* pMap, const int& nIterations, bool* pbStopFlag) {
  std::vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
  std::vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();
  BundleAdjustment(vpKFs, vpMPs, nIterations, pbStopFlag);
}  // GlobalBundleAdjustment


void Optimizer::BundleAdjustment(const std::vector<KeyFrame*>& vpKFs,
                                 const std::vector<MapPoint*>& vpMPs, const int& nIterations,
                                 bool* pbStopFlag) {
  g2o::SparseOptimizer optimizer;
  // g2o step 1 : set up linear solver type
  // linear solver - LinearSolverEigen
  // linear solver which uses the sparse Cholesky solver from Eigen
  auto linearSolver = std::make_unique<SlamLinearSolver>();
  linearSolver->setBlockOrdering(false);

  // g2o step 2 : construct blocksolver
  auto blockSolver = std::make_unique<SlamBlockSolver>(std::move(linearSolver));
  auto solver = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));

  // g2o step 3 : set up optimizer
  optimizer.setAlgorithm(solver);

  if (pbStopFlag) {
    optimizer.setForceStopFlag(pbStopFlag);
  }

  long unsigned int maxKFid = 0;

  // g2o step 4 : add vertex -- pose
  for (size_t i = 0; i < vpKFs.size(); ++i) {
    KeyFrame* pKF = vpKFs[i];
    if (pKF->isBad()) {
      continue;
    }
    g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
    vSE3->setId(pKF->GetId());
    vSE3->setFixed(pKF->GetId() == 0);
    optimizer.addVertex(vSE3);
    if (pKF->GetId() > maxKFid) {
      maxKFid = pKF->GetId();
    }
  }

  const float thHuber = sqrt(5.991);

  // g2o step 5 : add vertex -- landmark
  for (size_t i = 0; i < vpMPs.size(); ++i) {
    MapPoint* pMP = vpMPs[i];
    if (pMP->isBad()) {
      continue;
    }
    g2o::VertexPointXYZ* vPoint = new g2o::VertexPointXYZ();
    vPoint->setEstimate(pMP->GetWorldPos());
    int id = pMP->mnId + maxKFid + 1;  // make sure unique vertex id
    vPoint->setId(id);
    vPoint->setMarginalized(true);
    optimizer.addVertex(vPoint);

    // g2o step 6 : add edge -- pose-landmark
    std::map<KeyFrame*, size_t> observations = pMP->GetObservations();
    for (auto iter = observations.begin(); iter != observations.end(); ++iter) {
      KeyFrame* pKF = iter->first;
      if (pKF->isBad()) {
        continue;
      }
      const cv::KeyPoint& kpUn = pKF->mvKeysUn[iter->second];
      Eigen::Matrix<double, 2, 1> obs;
      obs << kpUn.pt.x, kpUn.pt.y;
      g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
      e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
      e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->GetId())));
      e->setMeasurement(obs);
      const float& invSigma2 = pKF->mpORBextractor->GetInverseScaleSigmaSquares()[kpUn.octave];
      e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
      g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
      rk->setDelta(thHuber);
      e->setRobustKernel(rk);
      e->fx = pKF->fx;
      e->fy = pKF->fy;
      e->cx = pKF->cx;
      e->cy = pKF->cy;
      optimizer.addEdge(e);
    }
  }

  // g2o step 7 : add edge -- pose-pose
  // 这里并没有加入frame-frame的边，可能是因为全局BA只在地图初始化的时候做，并没有回环检测的帧
  // pose-pose Edge只在出现回环检测的时候才会被加入

  // g2o step 8 : optimize
  optimizer.initializeOptimization();
  optimizer.optimize(nIterations);

  // g2o step 9 : update keyframe pose
  for (size_t i = 0; i < vpKFs.size(); ++i) {
    KeyFrame* pKF = vpKFs[i];
    if (pKF->isBad()) {
      continue;
    }
    g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->GetId()));
    g2o::SE3Quat SE3quat = vSE3->estimate();
    pKF->SetPose(PoseT(g2o::Isometry3(SE3quat)));
  }

  // g2o step 10 : update landmark position
  for (size_t i = 0; i < vpMPs.size(); ++i) {
    MapPoint* pMP = vpMPs[i];
    if (pMP->isBad()) {
      continue;
    }
    g2o::VertexPointXYZ* vPoint =
        static_cast<g2o::VertexPointXYZ*>(optimizer.vertex(pMP->mnId + maxKFid + 1));
    pMP->SetWorldPos(vPoint->estimate());
    pMP->UpdateNormalAndDepth();
  }

}  // BundleAdjustment

int Optimizer::PoseOptimization(FramePtr pFrame) {
  g2o::SparseOptimizer optimizer;
  // g2o step 1 : set up linear solver type
  // linear solver - LinearSolverEigen
  // linear solver which uses the sparse Cholesky solver from Eigen
  auto linearSolver = std::make_unique<SlamLinearSolver>();
  linearSolver->setBlockOrdering(false);

  // g2o step 2 : construct blocksolver
  auto blockSolver = std::make_unique<SlamBlockSolver>(std::move(linearSolver));
  auto solver = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));

  // g2o step 3 : set up optimizer
  optimizer.setAlgorithm(solver);

  int nInitialCorrespondences = 0;

  // g2o step 4 : add vertex -- pose
  g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
  vSE3->setEstimate(Converter::toSE3Quat(pFrame->GetPose()));
  vSE3->setId(0);
  vSE3->setFixed(false);
  optimizer.addVertex(vSE3);

  // g2o step 5 : add vertex -- landmark
  const int N = pFrame->N;
  std::vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdges;;
  std::vector<size_t> vnIndexEdge;
  vpEdges.reserve(N);
  vnIndexEdge.reserve(N);

  const float thHuber = sqrt(5.991);

  std::vector<MapPoint*> vpMapPoints = pFrame->mvpMapPoints;
  for (int i = 0; i < N; ++i) {
    MapPoint* pMP = vpMapPoints[i];
    if (!pMP) continue;

    nInitialCorrespondences++;
    pFrame->mvbOutlier[i] = false;

    Eigen::Vector2d obs;
    const cv::KeyPoint& kpUn = pFrame->mvKeysUn[i];
    obs << kpUn.pt.x, kpUn.pt.y;

    g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

    // set mappoint position
    Point3dT worldPos = pMP->GetWorldPos();
    e->Xw = worldPos;

    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
    e->setMeasurement(obs);
    const float& invSigma2 = pFrame->mpORBextractor->GetInverseScaleSigmaSquares()[kpUn.octave];
    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    rk->setDelta(thHuber);
    e->setRobustKernel(rk);

    e->fx = pFrame->fx;
    e->fy = pFrame->fy;
    e->cx = pFrame->cx;
    e->cy = pFrame->cy;

    optimizer.addEdge(e);
    vpEdges.push_back(e);
    vnIndexEdge.push_back(i);
  }

  // g2o step 6 : optimize
  if (nInitialCorrespondences < 3) {
    return 0;
  }

  // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
  // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
  const float chi2Thres[4]={5.991,5.991,5.991,5.991};
  const int its[4]={10,10,10,10};

  int nBad = 0;
  for (size_t it = 0; it < 4; ++it) {
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->GetPose()));
    optimizer.initializeOptimization(0);
    optimizer.optimize(its[it]); // optimize 10 iterations
    nBad = 0;

    for (size_t edgeIdx = 0, endIdx = vpEdges.size(); edgeIdx < endIdx; ++edgeIdx) {
      g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdges[edgeIdx];

      const size_t ptIdx = vnIndexEdge[edgeIdx];

      if (pFrame->mvbOutlier[ptIdx]) {
        e->computeError();
      }

      const float chi2 = e->chi2();
      if (chi2 > chi2Thres[it]) {
        pFrame->mvbOutlier[ptIdx] = true;
        e->setLevel(1); // disable this edge
        nBad++;
      } else {
        pFrame->mvbOutlier[ptIdx] = false;
        e->setLevel(0);
      }

      if (it == 2) {
        e->setRobustKernel(nullptr);
      } // disable robust kernel in the last second iteration
    } // all edge

    if (optimizer.edges().size() < 10) {
      break;
    }
  } // for (size_t it = 0; it < 4; ++it)

  // g2o step 7 : update keyframe pose
  PoseT Tcw = Converter::toPoseT(vSE3->estimate());
  pFrame->SetPose(Tcw);

  return nInitialCorrespondences - nBad;  

} // PoseOptimization


void Optimizer::LocalBundleAdjustment(KeyFramePtr pKF, Map* pMap) {
  LOG(INFO) << "LocalBundleAdjustment start";
  // Step 1 : 获取当前帧的共视帧  ----------------------------------------------
  std::unordered_set<KeyFrame*> spConnectedKFs; // 使用set去重
  spConnectedKFs.insert(pKF.get());
  // 原始代码这里KF新增一个获取所有共视帧的接口，这里直接用前200帧共视帧代替
  for (const auto& pNeighKF : pKF->GetBestCovisibilityKeyFrames(200)) {
    if (!pNeighKF || pNeighKF->isBad()) {
      continue;
    }
    spConnectedKFs.insert(pNeighKF);
  }

  // Step 2 : 获取局部关键帧的地图点 ----------------------------------------------
  std::unordered_set<MapPoint*> spConnectedMPs;
  for (const auto& pNeighKF : spConnectedKFs) {
    const auto& vpMPs = pNeighKF->mvpMapPoints;
    for (const auto& pMP : vpMPs) {
      if (pMP && !pMP->isBad()) {
        spConnectedMPs.insert(pMP);
      }
    }
  }

  // Step 3 : 获取固定帧（观测到局部地图点但是不在局部关键帧中） ----------------------------------------------
  std::unordered_set<KeyFrame*> spFixedKFs;
  for (const auto& pMP : spConnectedMPs) {
    const auto& observations = pMP->GetObservations();
    for (const auto& pNeighKF : observations) {
      if (pNeighKF.first->isBad() || spConnectedKFs.count(pNeighKF.first) > 0) {
        continue;
      }
      spFixedKFs.insert(pNeighKF.first);
    }
  }

  // Step 4 : 构建优化图 ----------------------------------------------
  g2o::SparseOptimizer optimizer;
  auto linearSolver = std::make_unique<SlamLinearSolver>();
  auto blockSolver = std::make_unique<SlamBlockSolver>(std::move(linearSolver));
  g2o::OptimizationAlgorithm* solver = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));
  optimizer.setAlgorithm(solver);
  // optimizer.setForceStopFlag(pbStopFlag);  // check pbStopFlag in each iteration

  // Step 4.1 : 添加关键帧节点 ----------------------------------------------
  // 记录最大的关键帧id，用于后面添加地图点节点时的id。其实可不可以用虚拟ID代替帧ID？
  // 后面还需要从优化图中取出关键帧，用来构建边，所以如果用虚拟ID,还需要记录虚拟ID和关键帧的对应关系
  unsigned long maxKFid = 0;  
  for (const auto& pKF : spConnectedKFs) {
    g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
    vSE3->setId(pKF->GetId());
    vSE3->setFixed(pKF->GetId() == 0);
    optimizer.addVertex(vSE3);
    if (pKF->GetId() > maxKFid) {
      maxKFid = pKF->GetId();
    }
  }

  // Step 4.2 : 添加固定关键帧节点 ----------------------------------------------
  for (const auto& pKF : spFixedKFs) {
    g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
    vSE3->setId(pKF->GetId());
    vSE3->setFixed(true);
    optimizer.addVertex(vSE3);
    if (pKF->GetId() > maxKFid) {
      maxKFid = pKF->GetId();
    }
  }

  // 阈值，同时也是Huber核函数的参数
  const float thHuber = sqrt(5.991);
  const float thEdgeOut = 5.991;

  // Step 4.3 : 添加地图点节点 ----------------------------------------------
  const int nExpectedSize = (spConnectedKFs.size() + spFixedKFs.size()) * spConnectedMPs.size();
  // 下面三个Vector用来缓存位姿-地图点的边，后续直接遍历EdgeVector检查边是否
  // chi2>5.991，然后删除chi2>5.991的边（从帧删除点观测，从点删除帧观测）
  // 为什么没法从优化图直接获取这些关系？
  // 优化图可以遍历Edge,然后拿出节点的ID,但是从节点ID找到对应的关键帧和节点则比较复杂
  std::vector<g2o::EdgeSE3ProjectXYZ*> vpEdges; // 用于后面删除chi2>5.991的边
  vpEdges.reserve(nExpectedSize);
  std::vector<KeyFrame*> vpEdgesKF;
  vpEdgesKF.reserve(nExpectedSize);
  std::vector<MapPoint*> vpEdgesMP;
  vpEdgesMP.reserve(nExpectedSize);
  for (const auto& pLocalMP : spConnectedMPs) {
    g2o::VertexPointXYZ* vPoint = new g2o::VertexPointXYZ();
    vPoint->setEstimate(pLocalMP->GetWorldPos());
    int id = pLocalMP->mnId + maxKFid + 1;  // make sure unique vertex id
    vPoint->setId(id);
    vPoint->setMarginalized(true);  // 加速求解[这里有一个相关的讨论issue](https://github.com/jingpang/LearnVIORB/issues/30)
    optimizer.addVertex(vPoint);

    // Step 4.4 : 添加位姿-地图点边 ----------------------------------------------
    const auto& observations = pLocalMP->GetObservations();
    for (const auto& observe : observations) {
      KeyFrame* pKF = observe.first;
      if (!pKF || pKF->isBad()) continue;
      const cv::KeyPoint& kpUn = pKF->mvKeysUn[observe.second];
      Eigen::Matrix<double, 2, 1> obs;
      obs << kpUn.pt.x, kpUn.pt.y;
      g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
      e->setVertex(0, optimizer.vertex(id)); // 地图点节点
      e->setVertex(1, optimizer.vertex(pKF->GetId())); // 关键帧节点
      e->setMeasurement(obs);
      const float& invSigma2 = pKF->mpORBextractor->GetInverseScaleSigmaSquares()[kpUn.octave];
      e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
      g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
      rk->setDelta(thHuber);
      e->setRobustKernel(rk);

      e->fx = pKF->fx;
      e->fy = pKF->fy;
      e->cx = pKF->cx;
      e->cy = pKF->cy;

      optimizer.addEdge(e);
      vpEdges.push_back(e);
      vpEdgesKF.push_back(pKF);
      vpEdgesMP.push_back(pLocalMP);
    }
  } // 遍历局部地图点

  // Step 5 : 执行第一次优化（5次迭代） -----------------------------------------------------------
  optimizer.initializeOptimization();
  optimizer.optimize(5);

  // Step 6 : 删除chi2>5.991的边以及地图点在Camera下深度小于0的边（设置Level为1） --------------------
  // 执行完第一次优化之后，还会将robustkernel取消
  for (const auto& edge : vpEdges) {
    if (edge->chi2() > thEdgeOut || !edge->isDepthPositive()) {
      edge->setLevel(1);
    }
    edge->setRobustKernel(nullptr);
  }

  // Step 7 : 执行第二次取出outliers的优化（10次迭代） ----------------------------------------------
  optimizer.initializeOptimization(0);  // level = 0
  optimizer.optimize(10);

  // Step 8 : 将chi2>5.991的边所连接的关键帧和地图点，两者的观测关系删除 -------------------------------
  for (size_t eIdx = 0, iend = vpEdges.size(); eIdx < iend; ++eIdx) {
    if (vpEdges[eIdx]->chi2() > thEdgeOut || !vpEdges[eIdx]->isDepthPositive()) {
      KeyFrame* pKF = vpEdgesKF[eIdx];
      MapPoint* pMP = vpEdgesMP[eIdx];

      // Erase MapPoint in pKF
      size_t mpIdxInKF = -1;
      const auto& observations = pMP->GetObservations();
      if (observations.count(pKF) > 0) {
        mpIdxInKF = observations.at(pKF);
      }
      if (mpIdxInKF) pKF->EraseMapPoint(mpIdxInKF);

      // Erase KeyFrame in MapPoint
      if (pMP && !pMP->isBad()) pMP->EraseObservation(pKF);

    }
  } // 删除chi2>5.991的边所连接的关键帧和地图点

  // Step 9 : 更新关键帧位姿和地图点位置 -----------------------------------------------------------
  // Step 9.1 : 更新关键帧位姿 -----------------------------------------------------------
  for (const auto& pLocalKF : spConnectedKFs) {
    g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pLocalKF->GetId()));
    g2o::SE3Quat SE3quat = vSE3->estimate();
    pLocalKF->SetPose(PoseT(g2o::Isometry3(SE3quat)));
  }
  // Step 9.2 : 更新地图点位置 -----------------------------------------------------------
  for (const auto& pLocalMP : spConnectedMPs) {
    g2o::VertexPointXYZ* vPoint = static_cast<g2o::VertexPointXYZ*>(optimizer.vertex(pLocalMP->mnId + maxKFid + 1));
    pLocalMP->SetWorldPos(vPoint->estimate());
    pLocalMP->UpdateNormalAndDepth();
  }

} // LocalBundleAdjustment

}  // namespace ORB_SLAM_Tracking
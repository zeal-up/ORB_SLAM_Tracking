/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez
 * Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 * Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós,
 * University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * ORB-SLAM3. If not, see <http://www.gnu.org/licenses/>.
 */
#pragma once

#include <list>
#include <opencv2/opencv.hpp>
#include <vector>

namespace ORB_SLAM_Tracking {

/**
 * @brief 作者使用四叉树来分割图像，然后进行均匀地计算特征点
 */
class ExtractorNode {
 public:
  ExtractorNode() : bNoMore(false) {}

  /**
   * @brief
   * 将提取器节点分成4个子节点，同时也完成图像区域的划分、特征点归属的划分，以及相关标志位的置位
   *
   * @param[in & out] n1  提取器节点1：左上
   * @param[in & out] n2  提取器节点1：右上
   * @param[in & out] n3  提取器节点1：左下
   * @param[in & out] n4  提取器节点1：右下
   */
  void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3,
                  ExtractorNode &n4);

  // 存储该节点的图像区域中对应的特征点坐标
  std::vector<cv::KeyPoint> vKeys;
  cv::Point2i UL, UR, BL, BR;
  std::list<ExtractorNode>::iterator lit;
  bool bNoMore;
};

class ORBextractor {
 public:
  enum { HARRIS_SCORE = 0, FAST_SCORE = 1 };

  /**
   * @brief ORBextractor 构造函数
   * @param[in] nfeatures
   * 要提取的特征点数量，最后提取出的特征点可能会大于等于这个值
   * @param[in] scaleFactor 图像金字塔的缩放系数
   * @param[in] nlevels 图像金字塔的层数
   * @param[in] iniThFAST FAST特征点提取阈值(较大)，可以提取出最明显的角点
   * @param[in] minThFAST 如果角点数量不够，则降低阈值
   */
  ORBextractor(int nfeatures, float scaleFactor, int nlevels, int iniThFAST,
               int minThFAST);
  ORBextractor(){};
  ~ORBextractor() {}

  // Compute the ORB features and descriptors on an image.
  // ORB are dispersed on the image using an octree.
  // Mask is ignored in the current implementation.
  /**
   * @brief 用仿函数（重载括号运算符）方法来计算图像特征点
   *
   * @param[in] _image                    输入原始图的图像
   * @param[in] _mask                     掩膜mask，没有用到
   * @param[in & out] _keypoints                存储特征点关键点的向量
   * @param[in & out] _descriptors              存储特征点描述子的矩阵
   */
  int operator()(cv::InputArray _image, cv::InputArray _mask,
                 std::vector<cv::KeyPoint> &_keypoints,
                 cv::OutputArray _descriptors, std::vector<int> &vLappingArea1);

  int inline GetLevels() { return nlevels; }

  float inline GetScaleFactor() { return scaleFactor; }

  std::vector<float> inline GetScaleFactors() { return mvScaleFactor; }

  std::vector<float> inline GetInverseScaleFactors() {
    return mvInvScaleFactor;
  }

  /**
   * @brief 获取每层高斯模糊的参数。是使用scale的平方计算出来的
   */
  std::vector<float> inline GetScaleSigmaSquares() { return mvLevelSigma2; }

  std::vector<float> inline GetInverseScaleSigmaSquares() {
    return mvInvLevelSigma2;
  }

  std::vector<int> inline GetNumFeaturesPerLevel() {
    return mnFeaturesPerLevel;
  }

  // 存储图像金字塔的图像
  std::vector<cv::Mat> mvImagePyramid;

 protected:
  void ComputePyramid(cv::Mat image);
  void ComputeKeyPointsOctTree(
      std::vector<std::vector<cv::KeyPoint>> &allKeypoints);

  /**
   * @brief
   * 使用四叉树法对一个图像金字塔图层中的特征点进行平均和分发;OctTree指区域的划分方式，实际上划分
   * 了四个区域，因此是四叉树
   *
   * @param[in] vToDistributeKeys     等待进行分配到四叉树中的特征点
   * @param[in] minX
   * 当前图层的图像的边界，坐标都是在“半径扩充图像”坐标系下的坐标
   * @param[in] maxX
   * @param[in] minY
   * @param[in] maxY
   * @param[in] nFeatures             希望提取出的特征点个数
   * @return vector<cv::KeyPoint>     已经均匀分散好的特征点vector容器
   */
  std::vector<cv::KeyPoint> DistributeOctTree(
      const std::vector<cv::KeyPoint> &vToDistributeKeys, const int &minX,
      const int &maxX, const int &minY, const int &maxY, const int &nFeatures);
  void ComputeKeyPointsOld(
      std::vector<std::vector<cv::KeyPoint>> &allKeypoints);

  // 计算BRIEF描述子时，随机点对的采样点集
  std::vector<cv::Point> pattern;

  int nfeatures;
  double scaleFactor;
  int nlevels;
  int iniThFAST;
  int minThFAST;

  // 每层图像需要提取出来的特征点个数
  std::vector<int> mnFeaturesPerLevel;
  std::vector<int> umax;
  // 存储每层图像相对于原始图像的缩放系数
  std::vector<float> mvScaleFactor;
  // 存储每层图像相对于原始图像的缩放系数的倒数
  std::vector<float> mvInvScaleFactor;
  // 存储每层图像相对于原始图像的缩放系数的平方
  std::vector<float> mvLevelSigma2;
  // 存储每层图像相对于原始图像的缩放系数的平方的倒数
  std::vector<float> mvInvLevelSigma2;
};

}  // namespace ORB_SLAM_Tracking

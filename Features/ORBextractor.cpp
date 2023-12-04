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
/**
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Willow Garage nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "Features/ORBextractor.hpp"

#include <algorithm>
#include <iostream>
#include <list>
#include <utility>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using std::list;
using std::pair;
using std::vector;

/**
 * 这部分代码用来提取图像的ORB特征
 * ORB特征使用金字塔FAST角点+改进后的BRIEF描述子（计算区块的主方向）
 * 主要流程：
 *  1.
 *
 * PS:
 * opencv有对ORB特征提取的实现（https://github.com/opencv/opencv/blob/85b04f0b4d5026e46527adaf2e71a0a45e7903b0/modules/features2d/src/orb.cpp）
 * 不清楚为什么还需要自己实现一个ORB提取器，这里的很多实现应该也是参考Opencv的实现;
 * 回答：后面发现应该是ORB特征提取反而简单，作者在对图像均匀提取关键点部分做了较多工作
 */

namespace ORB_SLAM_Tracking {
// 计算描述子（descriptor）的区块大小，同时也用区块计算主方向
const int PATCH_SIZE = 31;
// 区块大小的一半
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 19;
// 乘数因子，一度对应着多少弧度
const float factorPI = CV_PI / 180.f;

/**
 * @brief 计算特征点的主方向
 * 使用的方法是灰度质心法：以几何中心和灰度质心的连线作为特征点的方向
 * 计算特征点的方向可以使特征具有旋转不变形
 * @param[in] image     要进行操作的某层金字塔图像
 * @param[in] pt        当前特征点的坐标
 * @param[in] u_max     圆形图像块在PATCH中的每一行的坐标边界u_max
 * @return float        特征点的角度，范围[0, 360)，精度为0.3°
 */
static float IC_Angle(const cv::Mat &image, cv::Point2f pt,
                      const vector<int> &u_max) {
  // 图像的矩。m_01是根据像素值对y坐标加权，m_10是根据像素值对x坐标加权
  int m_01 = 0, m_10 = 0;

  // 获取关键点的像素值;以指针的方式存储，这样可以方便进行前后索引
  const uchar *center = &image.at<uchar>(cvRound(pt.y), cvRound(pt.x));

  // 计算灰度的质心（像素值加权的中心坐标），有几点要注意
  //  1.
  //  质心的计算虽然是在正方形的PATCH区域内计算，但实际上使用u_max约束的在正方形区域
  //      中的圆形。u_max表示圆形区域在正方形区域每一行的最大坐标
  //  2.
  //  为了加速计算，这里采用对称的方法，一次性遍历上下对称的两行，因此中心行需要单独处理

  // Treat the center line differently, v=0
  // 这条v=0中心线的计算需要特殊对待
  // 由于是中心行+若干行对，所以PATCH_SIZE应该是个奇数
  for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
    // 注意这里的center下标u可以是负的！中心水平线上的像素按x坐标（也就是u坐标）加权
    m_10 += u * center[u];

  // Go line by line in the circular patch
  // step 表示image每一行的像素数量
  int step = image.step1();

  // 以水平中心线（v=0）为对称轴，遍历上下对称的行
  for (int v = 1; v <= HALF_PATCH_SIZE; ++v) {
    int v_sum = 0;
    // 获取某行横坐标的最大范围，注意这里的图像块是圆形的
    int d = u_max[v];
    // 在坐标范围内挨个像素遍历，实际是一次遍历2个
    // 假设每次处理的两个点坐标，中心线下方为(x,y),中心线上方为(x,-y)
    // 对于某次待处理的两个点：m_10 = Σ x*I(x,y) =  x*I(x,y) + x*I(x,-y) =
    // x*(I(x,y) + I(x,-y)) 对于某次待处理的两个点：m_01 = Σ y*I(x,y) = y*I(x,y)
    // - y*I(x,-y) = y*(I(x,y) - I(x,-y))
    for (int u = -d; u <= d; ++u) {
      // val_plus：在中心线下方y=v时的的像素灰度值
      // val_minus：在中心线上方y=-v时的像素灰度值
      int val_plus = center[v * step + u];
      int val_minus = center[-v * step + u];

      // 在y轴上，对称的两行所有像素灰度值的差（两行具有相同的y轴绝对值坐标，后面直接乘以这个绝对值就行）
      v_sum += (val_plus - val_minus);

      // x轴上进行坐标加权（上下对称的两行一起计算）
      m_10 += u * (val_plus + val_minus);
    }

    // 对y轴坐标加权求和
    m_01 += v * v_sum;
  }

  // 计算坐标（m_10, m_01）到原点（图像块原点）（0,0）的角度
  // 单位（度），精度（0.3度）
  return cv::fastAtan2(m_01, m_10);
}

/**
 * @brief
 * 计算ORB特征点的描述子。注意这个是全局的静态函数，只能是在本文件内被调用
 * @param[in] kpt       特征点对象
 * @param[in] img       提取出特征点的图像
 * @param[in] pattern   预定义好的随机采样点集
 * @param[out] desc     用作输出变量，保存计算好的描述子，长度为32*8bit
 */
static void computeOrbDescriptor(const cv::KeyPoint &kpt, const cv::Mat &img,
                                 const cv::Point *pattern, uchar *desc) {
  // 角度转弧度
  float angle = kpt.angle * factorPI;
  // 计算角度的正余弦值
  float ag_cos = cos(angle), ag_sin = sin(angle);

  // 获得图像中心指针
  const uchar *center = &img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
  // 获得图像的每行的像素数量
  const int step = img.step1();

// 原始的BRIEF描述子不具有方向信息，通过加入特征点的方向来计算描述子，
// 称之为Steer BRIEF，具有较好旋转不变特性
// 具体地，在计算的时候需要将这里选取的随机点点集的x轴方向旋转到特征点的方向。
// 获得随机“相对点集”中某个idx所对应的点的灰度,这里旋转前坐标为(x,y),
// 旋转后坐标(x',y')推导: x'= xcos(θ) - ysin(θ),  y'= xsin(θ) + ycos(θ)
#define GET_VALUE(idx)                                                       \
  center[cvRound(pattern[idx].x * ag_sin + pattern[idx].y * ag_cos) * step + \
         cvRound(pattern[idx].x * ag_cos - pattern[idx].y * ag_sin)]

  // brief描述子由32*8=256位组成
  // 其中每一位是来自于两个像素点灰度的直接比较，所以每比较出8bit结果，需要16个随机点
  // 这也就是为什么pattern需要+=16的原因(pattern是指针，因此这里直接修改了指针)
  for (int i = 0; i < 32; ++i, pattern += 16) {
    // t0为点对的第一个点的灰度值；t1为第二个点的灰度值;val为比较结果
    int t0, t1;
    uchar val;

    t0 = GET_VALUE(0);
    t1 = GET_VALUE(1);
    val = t0 < t1;  // 描述子本字节的bit0
    t0 = GET_VALUE(2);
    t1 = GET_VALUE(3);
    val |= (t0 < t1) << 1;  // 描述子本字节的bit1
    t0 = GET_VALUE(4);
    t1 = GET_VALUE(5);
    val |= (t0 < t1) << 2;  // 描述子本字节的bit2
    t0 = GET_VALUE(6);
    t1 = GET_VALUE(7);
    val |= (t0 < t1) << 3;  // 描述子本字节的bit3
    t0 = GET_VALUE(8);
    t1 = GET_VALUE(9);
    val |= (t0 < t1) << 4;  // 描述子本字节的bit4
    t0 = GET_VALUE(10);
    t1 = GET_VALUE(11);
    val |= (t0 < t1) << 5;  // 描述子本字节的bit5
    t0 = GET_VALUE(12);
    t1 = GET_VALUE(13);
    val |= (t0 < t1) << 6;  // 描述子本字节的bit6
    t0 = GET_VALUE(14);
    t1 = GET_VALUE(15);
    val |= (t0 < t1) << 7;  // 描述子本字节的bit7

    // 保存当前比较的出来的描述子的这个字节
    desc[i] = val;
  }
// 为了避免和程序中的其他部分冲突在，在使用完成之后就取消这个宏定义
#undef GET_VALUE
}

// 下面就是预先定义好的随机点集，这个点集看起来跟opencv的实现是一样的
// 256是指可以提取出256bit的描述子信息，每个bit由一对点比较得来；
// 4=2*2，前面的2是需要两个点（一对点）进行比较，后面的2是一个点有两个坐标
static int bit_pattern_31_[256 * 4] = {
    8,   -3,  9,   5 /*mean (0), correlation (0)*/,
    4,   2,   7,   -12 /*mean (1.12461e-05), correlation (0.0437584)*/,
    -11, 9,   -8,  2 /*mean (3.37382e-05), correlation (0.0617409)*/,
    7,   -12, 12,  -13 /*mean (5.62303e-05), correlation (0.0636977)*/,
    2,   -13, 2,   12 /*mean (0.000134953), correlation (0.085099)*/,
    1,   -7,  1,   6 /*mean (0.000528565), correlation (0.0857175)*/,
    -2,  -10, -2,  -4 /*mean (0.0188821), correlation (0.0985774)*/,
    -13, -13, -11, -8 /*mean (0.0363135), correlation (0.0899616)*/,
    -13, -3,  -12, -9 /*mean (0.121806), correlation (0.099849)*/,
    10,  4,   11,  9 /*mean (0.122065), correlation (0.093285)*/,
    -13, -8,  -8,  -9 /*mean (0.162787), correlation (0.0942748)*/,
    -11, 7,   -9,  12 /*mean (0.21561), correlation (0.0974438)*/,
    7,   7,   12,  6 /*mean (0.160583), correlation (0.130064)*/,
    -4,  -5,  -3,  0 /*mean (0.228171), correlation (0.132998)*/,
    -13, 2,   -12, -3 /*mean (0.00997526), correlation (0.145926)*/,
    -9,  0,   -7,  5 /*mean (0.198234), correlation (0.143636)*/,
    12,  -6,  12,  -1 /*mean (0.0676226), correlation (0.16689)*/,
    -3,  6,   -2,  12 /*mean (0.166847), correlation (0.171682)*/,
    -6,  -13, -4,  -8 /*mean (0.101215), correlation (0.179716)*/,
    11,  -13, 12,  -8 /*mean (0.200641), correlation (0.192279)*/,
    4,   7,   5,   1 /*mean (0.205106), correlation (0.186848)*/,
    5,   -3,  10,  -3 /*mean (0.234908), correlation (0.192319)*/,
    3,   -7,  6,   12 /*mean (0.0709964), correlation (0.210872)*/,
    -8,  -7,  -6,  -2 /*mean (0.0939834), correlation (0.212589)*/,
    -2,  11,  -1,  -10 /*mean (0.127778), correlation (0.20866)*/,
    -13, 12,  -8,  10 /*mean (0.14783), correlation (0.206356)*/,
    -7,  3,   -5,  -3 /*mean (0.182141), correlation (0.198942)*/,
    -4,  2,   -3,  7 /*mean (0.188237), correlation (0.21384)*/,
    -10, -12, -6,  11 /*mean (0.14865), correlation (0.23571)*/,
    5,   -12, 6,   -7 /*mean (0.222312), correlation (0.23324)*/,
    5,   -6,  7,   -1 /*mean (0.229082), correlation (0.23389)*/,
    1,   0,   4,   -5 /*mean (0.241577), correlation (0.215286)*/,
    9,   11,  11,  -13 /*mean (0.00338507), correlation (0.251373)*/,
    4,   7,   4,   12 /*mean (0.131005), correlation (0.257622)*/,
    2,   -1,  4,   4 /*mean (0.152755), correlation (0.255205)*/,
    -4,  -12, -2,  7 /*mean (0.182771), correlation (0.244867)*/,
    -8,  -5,  -7,  -10 /*mean (0.186898), correlation (0.23901)*/,
    4,   11,  9,   12 /*mean (0.226226), correlation (0.258255)*/,
    0,   -8,  1,   -13 /*mean (0.0897886), correlation (0.274827)*/,
    -13, -2,  -8,  2 /*mean (0.148774), correlation (0.28065)*/,
    -3,  -2,  -2,  3 /*mean (0.153048), correlation (0.283063)*/,
    -6,  9,   -4,  -9 /*mean (0.169523), correlation (0.278248)*/,
    8,   12,  10,  7 /*mean (0.225337), correlation (0.282851)*/,
    0,   9,   1,   3 /*mean (0.226687), correlation (0.278734)*/,
    7,   -5,  11,  -10 /*mean (0.00693882), correlation (0.305161)*/,
    -13, -6,  -11, 0 /*mean (0.0227283), correlation (0.300181)*/,
    10,  7,   12,  1 /*mean (0.125517), correlation (0.31089)*/,
    -6,  -3,  -6,  12 /*mean (0.131748), correlation (0.312779)*/,
    10,  -9,  12,  -4 /*mean (0.144827), correlation (0.292797)*/,
    -13, 8,   -8,  -12 /*mean (0.149202), correlation (0.308918)*/,
    -13, 0,   -8,  -4 /*mean (0.160909), correlation (0.310013)*/,
    3,   3,   7,   8 /*mean (0.177755), correlation (0.309394)*/,
    5,   7,   10,  -7 /*mean (0.212337), correlation (0.310315)*/,
    -1,  7,   1,   -12 /*mean (0.214429), correlation (0.311933)*/,
    3,   -10, 5,   6 /*mean (0.235807), correlation (0.313104)*/,
    2,   -4,  3,   -10 /*mean (0.00494827), correlation (0.344948)*/,
    -13, 0,   -13, 5 /*mean (0.0549145), correlation (0.344675)*/,
    -13, -7,  -12, 12 /*mean (0.103385), correlation (0.342715)*/,
    -13, 3,   -11, 8 /*mean (0.134222), correlation (0.322922)*/,
    -7,  12,  -4,  7 /*mean (0.153284), correlation (0.337061)*/,
    6,   -10, 12,  8 /*mean (0.154881), correlation (0.329257)*/,
    -9,  -1,  -7,  -6 /*mean (0.200967), correlation (0.33312)*/,
    -2,  -5,  0,   12 /*mean (0.201518), correlation (0.340635)*/,
    -12, 5,   -7,  5 /*mean (0.207805), correlation (0.335631)*/,
    3,   -10, 8,   -13 /*mean (0.224438), correlation (0.34504)*/,
    -7,  -7,  -4,  5 /*mean (0.239361), correlation (0.338053)*/,
    -3,  -2,  -1,  -7 /*mean (0.240744), correlation (0.344322)*/,
    2,   9,   5,   -11 /*mean (0.242949), correlation (0.34145)*/,
    -11, -13, -5,  -13 /*mean (0.244028), correlation (0.336861)*/,
    -1,  6,   0,   -1 /*mean (0.247571), correlation (0.343684)*/,
    5,   -3,  5,   2 /*mean (0.000697256), correlation (0.357265)*/,
    -4,  -13, -4,  12 /*mean (0.00213675), correlation (0.373827)*/,
    -9,  -6,  -9,  6 /*mean (0.0126856), correlation (0.373938)*/,
    -12, -10, -8,  -4 /*mean (0.0152497), correlation (0.364237)*/,
    10,  2,   12,  -3 /*mean (0.0299933), correlation (0.345292)*/,
    7,   12,  12,  12 /*mean (0.0307242), correlation (0.366299)*/,
    -7,  -13, -6,  5 /*mean (0.0534975), correlation (0.368357)*/,
    -4,  9,   -3,  4 /*mean (0.099865), correlation (0.372276)*/,
    7,   -1,  12,  2 /*mean (0.117083), correlation (0.364529)*/,
    -7,  6,   -5,  1 /*mean (0.126125), correlation (0.369606)*/,
    -13, 11,  -12, 5 /*mean (0.130364), correlation (0.358502)*/,
    -3,  7,   -2,  -6 /*mean (0.131691), correlation (0.375531)*/,
    7,   -8,  12,  -7 /*mean (0.160166), correlation (0.379508)*/,
    -13, -7,  -11, -12 /*mean (0.167848), correlation (0.353343)*/,
    1,   -3,  12,  12 /*mean (0.183378), correlation (0.371916)*/,
    2,   -6,  3,   0 /*mean (0.228711), correlation (0.371761)*/,
    -4,  3,   -2,  -13 /*mean (0.247211), correlation (0.364063)*/,
    -1,  -13, 1,   9 /*mean (0.249325), correlation (0.378139)*/,
    7,   1,   8,   -6 /*mean (0.000652272), correlation (0.411682)*/,
    1,   -1,  3,   12 /*mean (0.00248538), correlation (0.392988)*/,
    9,   1,   12,  6 /*mean (0.0206815), correlation (0.386106)*/,
    -1,  -9,  -1,  3 /*mean (0.0364485), correlation (0.410752)*/,
    -13, -13, -10, 5 /*mean (0.0376068), correlation (0.398374)*/,
    7,   7,   10,  12 /*mean (0.0424202), correlation (0.405663)*/,
    12,  -5,  12,  9 /*mean (0.0942645), correlation (0.410422)*/,
    6,   3,   7,   11 /*mean (0.1074), correlation (0.413224)*/,
    5,   -13, 6,   10 /*mean (0.109256), correlation (0.408646)*/,
    2,   -12, 2,   3 /*mean (0.131691), correlation (0.416076)*/,
    3,   8,   4,   -6 /*mean (0.165081), correlation (0.417569)*/,
    2,   6,   12,  -13 /*mean (0.171874), correlation (0.408471)*/,
    9,   -12, 10,  3 /*mean (0.175146), correlation (0.41296)*/,
    -8,  4,   -7,  9 /*mean (0.183682), correlation (0.402956)*/,
    -11, 12,  -4,  -6 /*mean (0.184672), correlation (0.416125)*/,
    1,   12,  2,   -8 /*mean (0.191487), correlation (0.386696)*/,
    6,   -9,  7,   -4 /*mean (0.192668), correlation (0.394771)*/,
    2,   3,   3,   -2 /*mean (0.200157), correlation (0.408303)*/,
    6,   3,   11,  0 /*mean (0.204588), correlation (0.411762)*/,
    3,   -3,  8,   -8 /*mean (0.205904), correlation (0.416294)*/,
    7,   8,   9,   3 /*mean (0.213237), correlation (0.409306)*/,
    -11, -5,  -6,  -4 /*mean (0.243444), correlation (0.395069)*/,
    -10, 11,  -5,  10 /*mean (0.247672), correlation (0.413392)*/,
    -5,  -8,  -3,  12 /*mean (0.24774), correlation (0.411416)*/,
    -10, 5,   -9,  0 /*mean (0.00213675), correlation (0.454003)*/,
    8,   -1,  12,  -6 /*mean (0.0293635), correlation (0.455368)*/,
    4,   -6,  6,   -11 /*mean (0.0404971), correlation (0.457393)*/,
    -10, 12,  -8,  7 /*mean (0.0481107), correlation (0.448364)*/,
    4,   -2,  6,   7 /*mean (0.050641), correlation (0.455019)*/,
    -2,  0,   -2,  12 /*mean (0.0525978), correlation (0.44338)*/,
    -5,  -8,  -5,  2 /*mean (0.0629667), correlation (0.457096)*/,
    7,   -6,  10,  12 /*mean (0.0653846), correlation (0.445623)*/,
    -9,  -13, -8,  -8 /*mean (0.0858749), correlation (0.449789)*/,
    -5,  -13, -5,  -2 /*mean (0.122402), correlation (0.450201)*/,
    8,   -8,  9,   -13 /*mean (0.125416), correlation (0.453224)*/,
    -9,  -11, -9,  0 /*mean (0.130128), correlation (0.458724)*/,
    1,   -8,  1,   -2 /*mean (0.132467), correlation (0.440133)*/,
    7,   -4,  9,   1 /*mean (0.132692), correlation (0.454)*/,
    -2,  1,   -1,  -4 /*mean (0.135695), correlation (0.455739)*/,
    11,  -6,  12,  -11 /*mean (0.142904), correlation (0.446114)*/,
    -12, -9,  -6,  4 /*mean (0.146165), correlation (0.451473)*/,
    3,   7,   7,   12 /*mean (0.147627), correlation (0.456643)*/,
    5,   5,   10,  8 /*mean (0.152901), correlation (0.455036)*/,
    0,   -4,  2,   8 /*mean (0.167083), correlation (0.459315)*/,
    -9,  12,  -5,  -13 /*mean (0.173234), correlation (0.454706)*/,
    0,   7,   2,   12 /*mean (0.18312), correlation (0.433855)*/,
    -1,  2,   1,   7 /*mean (0.185504), correlation (0.443838)*/,
    5,   11,  7,   -9 /*mean (0.185706), correlation (0.451123)*/,
    3,   5,   6,   -8 /*mean (0.188968), correlation (0.455808)*/,
    -13, -4,  -8,  9 /*mean (0.191667), correlation (0.459128)*/,
    -5,  9,   -3,  -3 /*mean (0.193196), correlation (0.458364)*/,
    -4,  -7,  -3,  -12 /*mean (0.196536), correlation (0.455782)*/,
    6,   5,   8,   0 /*mean (0.1972), correlation (0.450481)*/,
    -7,  6,   -6,  12 /*mean (0.199438), correlation (0.458156)*/,
    -13, 6,   -5,  -2 /*mean (0.211224), correlation (0.449548)*/,
    1,   -10, 3,   10 /*mean (0.211718), correlation (0.440606)*/,
    4,   1,   8,   -4 /*mean (0.213034), correlation (0.443177)*/,
    -2,  -2,  2,   -13 /*mean (0.234334), correlation (0.455304)*/,
    2,   -12, 12,  12 /*mean (0.235684), correlation (0.443436)*/,
    -2,  -13, 0,   -6 /*mean (0.237674), correlation (0.452525)*/,
    4,   1,   9,   3 /*mean (0.23962), correlation (0.444824)*/,
    -6,  -10, -3,  -5 /*mean (0.248459), correlation (0.439621)*/,
    -3,  -13, -1,  1 /*mean (0.249505), correlation (0.456666)*/,
    7,   5,   12,  -11 /*mean (0.00119208), correlation (0.495466)*/,
    4,   -2,  5,   -7 /*mean (0.00372245), correlation (0.484214)*/,
    -13, 9,   -9,  -5 /*mean (0.00741116), correlation (0.499854)*/,
    7,   1,   8,   6 /*mean (0.0208952), correlation (0.499773)*/,
    7,   -8,  7,   6 /*mean (0.0220085), correlation (0.501609)*/,
    -7,  -4,  -7,  1 /*mean (0.0233806), correlation (0.496568)*/,
    -8,  11,  -7,  -8 /*mean (0.0236505), correlation (0.489719)*/,
    -13, 6,   -12, -8 /*mean (0.0268781), correlation (0.503487)*/,
    2,   4,   3,   9 /*mean (0.0323324), correlation (0.501938)*/,
    10,  -5,  12,  3 /*mean (0.0399235), correlation (0.494029)*/,
    -6,  -5,  -6,  7 /*mean (0.0420153), correlation (0.486579)*/,
    8,   -3,  9,   -8 /*mean (0.0548021), correlation (0.484237)*/,
    2,   -12, 2,   8 /*mean (0.0616622), correlation (0.496642)*/,
    -11, -2,  -10, 3 /*mean (0.0627755), correlation (0.498563)*/,
    -12, -13, -7,  -9 /*mean (0.0829622), correlation (0.495491)*/,
    -11, 0,   -10, -5 /*mean (0.0843342), correlation (0.487146)*/,
    5,   -3,  11,  8 /*mean (0.0929937), correlation (0.502315)*/,
    -2,  -13, -1,  12 /*mean (0.113327), correlation (0.48941)*/,
    -1,  -8,  0,   9 /*mean (0.132119), correlation (0.467268)*/,
    -13, -11, -12, -5 /*mean (0.136269), correlation (0.498771)*/,
    -10, -2,  -10, 11 /*mean (0.142173), correlation (0.498714)*/,
    -3,  9,   -2,  -13 /*mean (0.144141), correlation (0.491973)*/,
    2,   -3,  3,   2 /*mean (0.14892), correlation (0.500782)*/,
    -9,  -13, -4,  0 /*mean (0.150371), correlation (0.498211)*/,
    -4,  6,   -3,  -10 /*mean (0.152159), correlation (0.495547)*/,
    -4,  12,  -2,  -7 /*mean (0.156152), correlation (0.496925)*/,
    -6,  -11, -4,  9 /*mean (0.15749), correlation (0.499222)*/,
    6,   -3,  6,   11 /*mean (0.159211), correlation (0.503821)*/,
    -13, 11,  -5,  5 /*mean (0.162427), correlation (0.501907)*/,
    11,  11,  12,  6 /*mean (0.16652), correlation (0.497632)*/,
    7,   -5,  12,  -2 /*mean (0.169141), correlation (0.484474)*/,
    -1,  12,  0,   7 /*mean (0.169456), correlation (0.495339)*/,
    -4,  -8,  -3,  -2 /*mean (0.171457), correlation (0.487251)*/,
    -7,  1,   -6,  7 /*mean (0.175), correlation (0.500024)*/,
    -13, -12, -8,  -13 /*mean (0.175866), correlation (0.497523)*/,
    -7,  -2,  -6,  -8 /*mean (0.178273), correlation (0.501854)*/,
    -8,  5,   -6,  -9 /*mean (0.181107), correlation (0.494888)*/,
    -5,  -1,  -4,  5 /*mean (0.190227), correlation (0.482557)*/,
    -13, 7,   -8,  10 /*mean (0.196739), correlation (0.496503)*/,
    1,   5,   5,   -13 /*mean (0.19973), correlation (0.499759)*/,
    1,   0,   10,  -13 /*mean (0.204465), correlation (0.49873)*/,
    9,   12,  10,  -1 /*mean (0.209334), correlation (0.49063)*/,
    5,   -8,  10,  -9 /*mean (0.211134), correlation (0.503011)*/,
    -1,  11,  1,   -13 /*mean (0.212), correlation (0.499414)*/,
    -9,  -3,  -6,  2 /*mean (0.212168), correlation (0.480739)*/,
    -1,  -10, 1,   12 /*mean (0.212731), correlation (0.502523)*/,
    -13, 1,   -8,  -10 /*mean (0.21327), correlation (0.489786)*/,
    8,   -11, 10,  -6 /*mean (0.214159), correlation (0.488246)*/,
    2,   -13, 3,   -6 /*mean (0.216993), correlation (0.50287)*/,
    7,   -13, 12,  -9 /*mean (0.223639), correlation (0.470502)*/,
    -10, -10, -5,  -7 /*mean (0.224089), correlation (0.500852)*/,
    -10, -8,  -8,  -13 /*mean (0.228666), correlation (0.502629)*/,
    4,   -6,  8,   5 /*mean (0.22906), correlation (0.498305)*/,
    3,   12,  8,   -13 /*mean (0.233378), correlation (0.503825)*/,
    -4,  2,   -3,  -3 /*mean (0.234323), correlation (0.476692)*/,
    5,   -13, 10,  -12 /*mean (0.236392), correlation (0.475462)*/,
    4,   -13, 5,   -1 /*mean (0.236842), correlation (0.504132)*/,
    -9,  9,   -4,  3 /*mean (0.236977), correlation (0.497739)*/,
    0,   3,   3,   -9 /*mean (0.24314), correlation (0.499398)*/,
    -12, 1,   -6,  1 /*mean (0.243297), correlation (0.489447)*/,
    3,   2,   4,   -8 /*mean (0.00155196), correlation (0.553496)*/,
    -10, -10, -10, 9 /*mean (0.00239541), correlation (0.54297)*/,
    8,   -13, 12,  12 /*mean (0.0034413), correlation (0.544361)*/,
    -8,  -12, -6,  -5 /*mean (0.003565), correlation (0.551225)*/,
    2,   2,   3,   7 /*mean (0.00835583), correlation (0.55285)*/,
    10,  6,   11,  -8 /*mean (0.00885065), correlation (0.540913)*/,
    6,   8,   8,   -12 /*mean (0.0101552), correlation (0.551085)*/,
    -7,  10,  -6,  5 /*mean (0.0102227), correlation (0.533635)*/,
    -3,  -9,  -3,  9 /*mean (0.0110211), correlation (0.543121)*/,
    -1,  -13, -1,  5 /*mean (0.0113473), correlation (0.550173)*/,
    -3,  -7,  -3,  4 /*mean (0.0140913), correlation (0.554774)*/,
    -8,  -2,  -8,  3 /*mean (0.017049), correlation (0.55461)*/,
    4,   2,   12,  12 /*mean (0.01778), correlation (0.546921)*/,
    2,   -5,  3,   11 /*mean (0.0224022), correlation (0.549667)*/,
    6,   -9,  11,  -13 /*mean (0.029161), correlation (0.546295)*/,
    3,   -1,  7,   12 /*mean (0.0303081), correlation (0.548599)*/,
    11,  -1,  12,  4 /*mean (0.0355151), correlation (0.523943)*/,
    -3,  0,   -3,  6 /*mean (0.0417904), correlation (0.543395)*/,
    4,   -11, 4,   12 /*mean (0.0487292), correlation (0.542818)*/,
    2,   -4,  2,   1 /*mean (0.0575124), correlation (0.554888)*/,
    -10, -6,  -8,  1 /*mean (0.0594242), correlation (0.544026)*/,
    -13, 7,   -11, 1 /*mean (0.0597391), correlation (0.550524)*/,
    -13, 12,  -11, -13 /*mean (0.0608974), correlation (0.55383)*/,
    6,   0,   11,  -13 /*mean (0.065126), correlation (0.552006)*/,
    0,   -1,  1,   4 /*mean (0.074224), correlation (0.546372)*/,
    -13, 3,   -9,  -2 /*mean (0.0808592), correlation (0.554875)*/,
    -9,  8,   -6,  -3 /*mean (0.0883378), correlation (0.551178)*/,
    -13, -6,  -8,  -2 /*mean (0.0901035), correlation (0.548446)*/,
    5,   -9,  8,   10 /*mean (0.0949843), correlation (0.554694)*/,
    2,   7,   3,   -9 /*mean (0.0994152), correlation (0.550979)*/,
    -1,  -6,  -1,  -1 /*mean (0.10045), correlation (0.552714)*/,
    9,   5,   11,  -2 /*mean (0.100686), correlation (0.552594)*/,
    11,  -3,  12,  -8 /*mean (0.101091), correlation (0.532394)*/,
    3,   0,   3,   5 /*mean (0.101147), correlation (0.525576)*/,
    -1,  4,   0,   10 /*mean (0.105263), correlation (0.531498)*/,
    3,   -6,  4,   5 /*mean (0.110785), correlation (0.540491)*/,
    -13, 0,   -10, 5 /*mean (0.112798), correlation (0.536582)*/,
    5,   8,   12,  11 /*mean (0.114181), correlation (0.555793)*/,
    8,   9,   9,   -6 /*mean (0.117431), correlation (0.553763)*/,
    7,   -4,  8,   -12 /*mean (0.118522), correlation (0.553452)*/,
    -10, 4,   -10, 9 /*mean (0.12094), correlation (0.554785)*/,
    7,   3,   12,  4 /*mean (0.122582), correlation (0.555825)*/,
    9,   -7,  10,  -2 /*mean (0.124978), correlation (0.549846)*/,
    7,   0,   12,  -2 /*mean (0.127002), correlation (0.537452)*/,
    -1,  -6,  0,   -11 /*mean (0.127148), correlation (0.547401)*/
};

ORBextractor::ORBextractor(int nfeatures, float scaleFactor, int nlevels,
                           int iniThFAST, int minThFAST)
    : nfeatures(nfeatures),
      scaleFactor(scaleFactor),
      nlevels(nlevels),
      iniThFAST(iniThFAST),
      minThFAST(minThFAST) {
  mvScaleFactor.resize(nlevels);
  mvLevelSigma2.resize(nlevels);

  if (scaleFactor == 1.0 && nlevels > 1) {
    std::cerr << "scaleFactor == 1.0 but nlevels is not 1" << std::endl;
    exit(1);
  }

  // 对于第一层，也就是原始图像，缩放系数是1
  mvScaleFactor[0] = 1.0f;
  mvLevelSigma2[0] = 1.0f;
  // 逐层计算图像金字塔相对于原始图像的缩放系数
  for (int i = 1; i < nlevels; ++i) {
    mvScaleFactor[i] = mvScaleFactor[i - 1] * scaleFactor;
    mvLevelSigma2[i] = mvScaleFactor[i] * mvScaleFactor[i];
  }

  mvInvScaleFactor.resize(nlevels);
  mvInvLevelSigma2.resize(nlevels);
  for (int i = 0; i < nlevels; i++) {
    mvInvScaleFactor[i] = 1.0f / mvScaleFactor[i];
    mvInvLevelSigma2[i] = 1.0f / mvLevelSigma2[i];
  }

  // 调整图像金字塔vector以使得其符合咱们设定的图像层数
  mvImagePyramid.resize(nlevels);
  // 每层图像的特征点数量
  mnFeaturesPerLevel.resize(nlevels);

  // 图片降采样缩放系数的倒数;scaleFactor默认是1.2
  float factor = 1.0f / scaleFactor;

  // 计算第一层（scale=1）的特征点数量
  // 后面每一层的特征点数量都是前一层的数量乘以factor
  // 根据等比数列求和公式计算出来
  float nDesiredFeaturesPerScale =
      nfeatures * (1 - factor) /
      (1 - static_cast<float>(pow(factor, static_cast<float>(nlevels))));

  // 用于分配特征点数量，特征点累计清空
  int sumFeatures = 0;
  // 逐层计算要分配的特征点个数，顶层图像（最小层）分到剩下的特征点个数
  for (int level = 0; level < nlevels - 1; level++) {
    mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
    sumFeatures += mnFeaturesPerLevel[level];
    nDesiredFeaturesPerScale *= factor;
  }
  // 最后一层分到剩下的特征点个数，保证所有层的特征点个数和起来等于nfeatures
  mnFeaturesPerLevel[nlevels - 1] = std::max(nfeatures - sumFeatures, 0);
  sumFeatures += mnFeaturesPerLevel[nlevels - 1];
  std::cout << "Sum of features = " << sumFeatures << std::endl;

  // 计算BRIEF特征描述子时采样点集的点的个数（256*2）
  const int npoints = 512;
  const cv::Point *pattern0 = (const cv::Point *)bit_pattern_31_;
  // 将BRIEF特征描述子采样点集复制到成员变量pattern中
  std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));

  // This is for orientation
  // 下面的内容是和特征点的旋转计算有关的
  // pre-compute the end of a row in a circular patch
  // 预先计算圆形patch中行的结束位置
  // +1中的1表示那个圆的中间行
  umax.resize(HALF_PATCH_SIZE + 1);

  // -------------------------------------
  // 以下部分用来计算圆形的范围
  // 由于是在像素域（离散）下计算正方形的内接圆范围，同时，为了保证圆形尽可能对称，
  // 下面这部分一开始看起来会稍微比较乱，主要注意几点
  // 1. umax计算只需要在半圆计算即可，后面用的时候直接加正负号就行
  // 2. 先计算右上域1/4中0-45度范围的值，然后直接对折，计算45-90度的值
  // 3. umax指的是圆形区域的每一行的最大的横坐标
  // ------------------------------------

  // cvFloor返回不大于参数的最大整数值，cvCeil返回不小于参数的最小整数值，cvRound则是四舍五入
  int v;   // 循环辅助变量
  int v0;  // 辅助变量
  // 计算45度角（也就是对称轴）的v值（y值），后续用这个对称轴折叠计算上半部分的umax值
  // +1 确保在对称计算的过程中考虑到整个1/4圆的区域
  int vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
  // 这里的二分之根2就是对应那个45°圆心角
  int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);

  // 半径的平方
  const double hp2 = HALF_PATCH_SIZE * HALF_PATCH_SIZE;

  // 利用圆的方程计算右上角1/4圆的下45度角区域的umax值
  for (v = 0; v <= vmax; ++v) umax[v] = cvRound(sqrt(hp2 - v * v));

  // 根据45度对称轴计算上45度的umax值
  // 使用对称的方法计算，目的是为了保持严格的对称
  for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v) {
    while (umax[v0] == umax[v0 + 1]) ++v0;
    umax[v] = v0;
    ++v0;
  }
}

/**
 * @brief 计算特征点的方向
 * @param[in] image                 特征点所在当前金字塔的图像
 * @param[in & out] keypoints       特征点向量
 * @param[in] umax                  每个特征点所在图像区块的每行的边界 u_max
 * 组成的vector
 */
static void computeOrientation(const cv::Mat &image,
                               vector<cv::KeyPoint> &keypoints,
                               const vector<int> &umax) {
  // 遍历所有特征点
  for (vector<cv::KeyPoint>::iterator keypoint = keypoints.begin(),
                                      keypointEnd = keypoints.end();
       keypoint != keypointEnd; ++keypoint) {
    // 调用IC_Angle计算每一個特征点的方向
    // 返回的值单位是度，值的范围是[0,360)
    keypoint->angle = IC_Angle(image, keypoint->pt, umax);
  }
}

void ExtractorNode::DivideNode(ExtractorNode &n1, ExtractorNode &n2,
                               ExtractorNode &n3, ExtractorNode &n4) {
  // 得到当前提取器节点所在图像区域的一半长宽，结果取整
  const int halfX = ceil(static_cast<float>(UR.x - UL.x) / 2);
  const int halfY = ceil(static_cast<float>(BR.y - UL.y) / 2);

  // Define boundaries of childs
  // 下面的操作将一个图像区域再细分成四个子区域
  // 根据后面关键点的分配规则，区域的边界坐标是左闭右开，上闭下开
  // n1 存储左上区域边界
  n1.UL = UL;
  n1.UR = cv::Point2i(UL.x + halfX, UL.y);
  n1.BL = cv::Point2i(UL.x, UL.y + halfY);
  n1.BR = cv::Point2i(UL.x + halfX, UL.y + halfY);
  n1.vKeys.reserve(vKeys.size());

  // n2 存储右上区域的边界
  n2.UL = n1.UR;
  n2.UR = UR;
  n2.BL = n1.BR;
  n2.BR = cv::Point2i(UR.x, UL.y + halfY);
  n2.vKeys.reserve(vKeys.size());

  // n3 存储左下区域的边界
  n3.UL = n1.BL;
  n3.UR = n1.BR;
  n3.BL = BL;
  n3.BR = cv::Point2i(n1.BR.x, BL.y);
  n3.vKeys.reserve(vKeys.size());

  // n4 存储右下区域的边界
  n4.UL = n3.UR;
  n4.UR = n2.BR;
  n4.BL = n3.BR;
  n4.BR = BR;
  n4.vKeys.reserve(vKeys.size());

  // Associate points to childs
  // 将大图像的特征点分配到小图像中
  for (size_t i = 0; i < vKeys.size(); i++) {
    const cv::KeyPoint &kp = vKeys[i];
    if (kp.pt.x < n1.UR.x) {
      if (kp.pt.y < n1.BR.y)
        n1.vKeys.push_back(kp);
      else
        n3.vKeys.push_back(kp);
    } else if (kp.pt.y < n1.BR.y) {
      n2.vKeys.push_back(kp);
    } else {
      n4.vKeys.push_back(kp);
    }
  }

  // 判断每个子特征点提取器节点所在的图像中特征点的数目（就是分配给子节点的特征点数目），然后做标记
  // 这里判断是否数目等于1的目的是确定这个节点还能不能再向下进行分裂
  if (n1.vKeys.size() == 1) n1.bNoMore = true;
  if (n2.vKeys.size() == 1) n2.bNoMore = true;
  if (n3.vKeys.size() == 1) n3.bNoMore = true;
  if (n4.vKeys.size() == 1) n4.bNoMore = true;
}

/**
 * @brief
 * 比较特征提取节点，如果输入的pair中的int不相等，则用int值比较，否则比较提取节点的UL.x坐标;
 * int值表示的是该区域的特征点数量
 * @return true if e1 < e2
 */
static bool compareNodes(std::pair<int, ExtractorNode *> &e1,
                         std::pair<int, ExtractorNode *> &e2) {
  if (e1.first < e2.first) {
    return true;
  } else if (e1.first > e2.first) {
    return false;
  } else {
    if (e1.second->UL.x < e2.second->UL.x)
      return true;
    else
      return false;
  }
}

std::vector<cv::KeyPoint> ORBextractor::DistributeOctTree(
    const std::vector<cv::KeyPoint> &vToDistributeKeys, const int &minX,
    const int &maxX, const int &minY, const int &maxY, const int &nFeatures) {
  // Compute how many initial nodes
  // Step 1 根据宽高比确定初始节点数目
  // 计算应该生成的初始节点个数，根节点的数量nIni是根据边界的宽高比值确定的，一般是1或者2
  // 这里划分初始提取器是为了根提取器的图像区域尽可能是正方形
  // ! bug: 如果宽高比小于0.5，nIni=0, 后面hx会报错
  const int nIni = round(static_cast<float>(maxX - minX) / (maxY - minY));

  // 一个初始的节点的x方向有多少个像素
  const float hX = static_cast<float>(maxX - minX) / nIni;

  // 存储所有提取器节点的列表
  std::list<ExtractorNode> lNodes;
  // 存储初始提取器节点指针的vector
  vector<ExtractorNode *> vpIniNodes;
  // 然后重新设置其大小
  vpIniNodes.resize(nIni);

  // Step2 生成初始提取器节点
  for (int i = 0; i < nIni; i++) {
    // 生成一个提取器节点
    ExtractorNode ni;

    // 设置提取器节点的图像边界
    // 注意这里和提取FAST角点区域相同，都是“半径扩充图像”，特征点坐标从0 开始
    ni.UL = cv::Point2i(hX * static_cast<float>(i), 0);      // UpLeft
    ni.UR = cv::Point2i(hX * static_cast<float>(i + 1), 0);  // UpRight
    ni.BL = cv::Point2i(ni.UL.x, maxY - minY);               // BottomLeft
    ni.BR = cv::Point2i(ni.UR.x, maxY - minY);               // BottomRight

    // 重设vkeys大小
    ni.vKeys.reserve(vToDistributeKeys.size());

    // 将刚才生成的提取节点添加到列表中
    // 虽然这里的ni是局部变量，但是由于这里的push_back()是拷贝参数的内容到一个新的对象中然后再添加到列表中
    // 所以当本函数退出之后这里的内存不会成为“野指针”
    lNodes.push_back(ni);
    // 存储这个初始的提取器节点句柄
    vpIniNodes[i] = &lNodes.back();
  }

  // Associate points to childs
  // Step 3 将特征点分配到子提取器节点中
  for (size_t i = 0; i < vToDistributeKeys.size(); i++) {
    // 获取这个特征点对象
    const cv::KeyPoint &kp = vToDistributeKeys[i];
    // 按特征点的横轴位置，分配给属于那个图像区域的提取器节点（最初的提取器节点）
    vpIniNodes[kp.pt.x / hX]->vKeys.push_back(kp);
  }

  // Step 4
  // 遍历根提取器节点列表，标记那些不可再分裂的节点，删除那些没有分配到特征点的节点
  list<ExtractorNode>::iterator lit = lNodes.begin();
  while (lit != lNodes.end()) {
    // 如果初始的提取器节点所分配到的特征点个数为1
    if (lit->vKeys.size() == 1) {
      // 那么就标志位置位，表示此节点不可再分
      lit->bNoMore = true;
      // 更新迭代器
      lit++;
      // 如果一个提取器节点没有被分配到特征点，那么就从列表中直接删除它
    } else if (lit->vKeys.empty()) {
      // 注意，由于是直接删除了它，所以这里的迭代器没有必要更新；否则反而会造成跳过元素的情况
      lit = lNodes.erase(lit);
    } else {
      // 如果上面的这些情况和当前的特征点提取器节点无关，那么就只是更新迭代器
      lit++;
    }
  }

  // 结束标志位清空
  bool bFinish = false;
  // 记录迭代次数，只是记录，并未起到作用
  int iteration = 0;
  // 需要展开的节点计数，这个一直保持累计，不清零
  // int nToExpand = 0;

  // 声明一个vector用于存储节点的vSize和句柄对
  // 这个变量记录了在一次分裂循环中，那些可以再继续进行分裂的节点中包含的特征点数目和其句柄
  vector<pair<int, ExtractorNode *>> vSizeAndPointerToNode;

  // 调整大小，一个节点最多分裂出4个子区域
  vSizeAndPointerToNode.reserve(lNodes.size() * 4);

  // Step 5 根据兴趣点分布,利用4叉树方法对图像进行划分区域
  while (!bFinish) {
    // 更新迭代次数计数器，只是记录，并未起到作用
    iteration++;

    // 保存当前节点个数，prev在这里理解为“保留”比较好
    int prevSize = lNodes.size();

    // 重新定位迭代器指向列表头部
    lit = lNodes.begin();

    // 需要展开的节点计数，这个一直保持累计，不清零
    int nToExpand = 0;

    // 因为是在循环中，前面的循环体中可能污染了这个变量，so清空这个vector
    // 这个变量也只是统计了某一个循环中的点
    // 这个变量记录了在一次分裂循环中，那些可以再继续进行分裂的节点中包含的特征点数目和其句柄
    vSizeAndPointerToNode.clear();

    // 将目前的子区域进行划分
    // 开始遍历列表中所有的提取器节点，并进行分解或者保留
    while (lit != lNodes.end()) {
      // 如果提取器节点只有一个特征点，
      if (lit->bNoMore) {
        // If node only contains one point do not subdivide and continue
        // 那么就没有必要再进行细分了
        lit++;
        // 跳过当前节点，继续下一个
        continue;
      } else {
        // If more than one point, subdivide
        // 如果当前的提取器节点具有超过一个的特征点，那么就要进行继续细分
        ExtractorNode n1, n2, n3, n4;

        // 再细分成四个子区域
        lit->DivideNode(n1, n2, n3, n4);

        // Add childs if they contain points
        // 如果这里分出来的子区域中有特征点，那么就将这个子区域的节点添加到提取器节点的列表中
        // 注意这里的条件是，有特征点即可
        if (n1.vKeys.size() > 0) {
          // 注意这里也是添加到列表前面的
          lNodes.push_front(n1);

          // 再判断其中子提取器节点中的特征点数目是否大于1
          if (n1.vKeys.size() > 1) {
            // 如果有超过一个的特征点，那么“待展开的节点计数++”
            nToExpand++;

            // 保存这个特征点数目和节点指针的信息
            vSizeAndPointerToNode.push_back(
                std::make_pair(n1.vKeys.size(), &lNodes.front()));

            // ?这个访问用的句柄貌似并没有用到？
            // lNodes.front().lit 和前面的迭代的lit 不同，只是名字相同而已
            // lNodes.front().lit是node结构体里的一个指针用来记录节点的位置
            // 迭代的lit 是while循环里作者命名的遍历的指针名称
            lNodes.front().lit = lNodes.begin();
          }
        }
        // 后面的操作都是相同的，这里不再赘述
        if (n2.vKeys.size() > 0) {
          lNodes.push_front(n2);
          if (n2.vKeys.size() > 1) {
            nToExpand++;
            vSizeAndPointerToNode.push_back(
                std::make_pair(n2.vKeys.size(), &lNodes.front()));
            lNodes.front().lit = lNodes.begin();
          }
        }
        if (n3.vKeys.size() > 0) {
          lNodes.push_front(n3);
          if (n3.vKeys.size() > 1) {
            nToExpand++;
            vSizeAndPointerToNode.push_back(
                std::make_pair(n3.vKeys.size(), &lNodes.front()));
            lNodes.front().lit = lNodes.begin();
          }
        }
        if (n4.vKeys.size() > 0) {
          lNodes.push_front(n4);
          if (n4.vKeys.size() > 1) {
            nToExpand++;
            vSizeAndPointerToNode.push_back(
                std::make_pair(n4.vKeys.size(), &lNodes.front()));
            lNodes.front().lit = lNodes.begin();
          }
        }

        // 当这个母节点expand之后就从列表中删除它了，能够进行分裂操作说明至少有一个子节点的区域中特征点的数量是>1的
        lit = lNodes.erase(lit);

        // 继续下一次循环，其实这里加不加这句话的作用都是一样的
        continue;
      }  // 判断当前遍历到的节点中是否有超过一个的特征点
    }    // 遍历列表中的所有提取器节点

    // Finish if there are more nodes than required features or all nodes
    // contain just one point 停止这个过程的条件有两个，满足其中一个即可：
    // 1、当前的节点数已经超过了要求的特征点数
    // 2、当前所有的节点中都只包含一个特征点
    // prevSize中保存的是分裂之前的节点个数，如果分裂之前和分裂之后的总节点个数一样，说明当前所有的
    // 节点区域中只有一个特征点，已经不能够再细分了
    if (static_cast<int>(lNodes.size()) >= nFeatures ||
        static_cast<int>(lNodes.size()) == prevSize) {
      // 停止标志置位
      bFinish = true;

      // Step 6 当再划分之后所有的Node数大于要求数目时,
      // 为了让分出来的区域数量尽量接近特征点数量
      // 这里做额外的操作，优先分裂特征点数量大的区域，直到总的区域超过要求的特征点数量
      // 可以展开的子节点个数nToExpand
      // x3，是因为一分四之后，会删除原来的主节点，所以乘以3
    } else if ((static_cast<int>(lNodes.size()) + nToExpand * 3) > nFeatures) {
      // 如果再分裂一次那么数目就要超了，这里想办法尽可能使其刚刚达到或者超过要求的特征点个数时就退出
      // 这里的nToExpand和vSizeAndPointerToNode不是一次循环对一次循环的关系，而是前者是累计计数，后者只保存某一个循环的
      // 一直循环，直到结束标志位被置位
      while (!bFinish) {
        // 获取当前的list中的节点个数
        prevSize = lNodes.size();

        // 缓存刚才那一次循环的节点和对应的特征点数量vector
        vector<pair<int, ExtractorNode *>> vPrevSizeAndPointerToNode =
            vSizeAndPointerToNode;
        vSizeAndPointerToNode.clear();

        // 对需要划分的节点进行排序，对pair对的第一个元素进行排序，默认是从小到大排序
        // 优先分裂特征点多的节点，使得特征点密集的区域保留更少的特征点
        sort(vPrevSizeAndPointerToNode.begin(), vPrevSizeAndPointerToNode.end(),
             compareNodes);

        // 遍历这个存储了pair对的vector，注意是从后往前遍历
        for (int j = vPrevSizeAndPointerToNode.size() - 1; j >= 0; j--) {
          ExtractorNode n1, n2, n3, n4;
          // 对每个需要进行分裂的节点进行分裂
          vPrevSizeAndPointerToNode[j].second->DivideNode(n1, n2, n3, n4);

          // Add childs if they contain points
          // 其实这里的节点可以说是二级子节点了，执行和前面一样的操作
          if (n1.vKeys.size() > 0) {
            lNodes.push_front(n1);
            if (n1.vKeys.size() > 1) {
              // 因为这里还有对于vSizeAndPointerToNode的操作，所以前面才会备份vSizeAndPointerToNode中的数据
              // 为可能的、后续的又一次for循环做准备
              vSizeAndPointerToNode.push_back(
                  std::make_pair(n1.vKeys.size(), &lNodes.front()));
              lNodes.front().lit = lNodes.begin();
            }
          }
          if (n2.vKeys.size() > 0) {
            lNodes.push_front(n2);
            if (n2.vKeys.size() > 1) {
              vSizeAndPointerToNode.push_back(
                  std::make_pair(n2.vKeys.size(), &lNodes.front()));
              lNodes.front().lit = lNodes.begin();
            }
          }
          if (n3.vKeys.size() > 0) {
            lNodes.push_front(n3);
            if (n3.vKeys.size() > 1) {
              vSizeAndPointerToNode.push_back(
                  std::make_pair(n3.vKeys.size(), &lNodes.front()));
              lNodes.front().lit = lNodes.begin();
            }
          }
          if (n4.vKeys.size() > 0) {
            lNodes.push_front(n4);
            if (n4.vKeys.size() > 1) {
              vSizeAndPointerToNode.push_back(
                  std::make_pair(n4.vKeys.size(), &lNodes.front()));
              lNodes.front().lit = lNodes.begin();
            }
          }

          // 删除母节点，在这里其实应该是一级子节点
          lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

          // 判断是是否超过了需要的特征点数？是的话就退出，不是的话就继续这个分裂过程，直到刚刚达到或者超过要求的特征点个数
          // 作者的思想其实就是这样的，再分裂了一次之后判断下一次分裂是否会超过N，如果不是那么就放心大胆地全部进行分裂（因为少了一个判断因此
          // 其运算速度会稍微快一些），如果会那么就引导到这里进行最后一次分裂
          if (static_cast<int>(lNodes.size()) >= nFeatures) break;
        }  // 结束对vPrevSizeAndPointerToNode的遍历

        // 判断是否达到了停止条件
        if (static_cast<int>(lNodes.size()) >= nFeatures ||
            static_cast<int>(lNodes.size()) == prevSize)
          bFinish = true;
      }  // 结束while循环
    }
  }

  // Retain the best point in each node
  // Step 7 保留每个区域响应值最大的一个兴趣点
  // 使用这个vector来存储我们感兴趣的特征点的过滤结果
  vector<cv::KeyPoint> vResultKeys;

  // 调整大小为要提取的特征点数目
  vResultKeys.reserve(nfeatures);

  // 遍历这个节点列表
  for (list<ExtractorNode>::iterator lit = lNodes.begin(); lit != lNodes.end();
       lit++) {
    // 得到这个节点区域中的特征点容器句柄
    vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;

    // 得到指向第一个特征点的指针，后面作为最大响应值对应的关键点
    cv::KeyPoint *pKP = &vNodeKeys[0];

    // 用第1个关键点响应值初始化最大响应值
    float maxResponse = pKP->response;

    // 开始遍历这个节点区域中的特征点容器中的特征点，注意是从1开始哟，0已经用过了
    for (size_t k = 1; k < vNodeKeys.size(); k++) {
      // 更新最大响应值
      if (vNodeKeys[k].response > maxResponse) {
        // 更新pKP指向具有最大响应值的keypoints
        pKP = &vNodeKeys[k];
        maxResponse = vNodeKeys[k].response;
      }
    }

    // 将这个节点区域中的响应值最大的特征点加入最终结果容器
    vResultKeys.push_back(*pKP);
  }

  // 返回最终结果容器，其中保存有分裂出来的区域中，我们最感兴趣、响应值最大的特征点
  return vResultKeys;
}

/**
 * @brief
 * 计算四叉树的特征点，函数名字后面的OctTree只是说明了在过滤和分配特征点时所使用的方式
 *  步骤：
 *      1. 遍历所有层级的图像
 *          1.1 每一个层级的图像划分为大小为35的Cell,对每个Cell提取FAST关键点
 *          1.2 将提取的关键点的坐标转换到图像坐标系
 *          1.3 合并所有cell提取出的关键点
 *          1.4
 * 使用DistributeOctTree将这一层图像的关键点分布到各个子区域，最终每层提取出固定数量的关键点
 *
 * @param[in & out] allKeypoints（vector<vector<cv::KeyPoint>>）所有特征点;allKeypoints[i]表示第i层图像的特征点
 */
void ORBextractor::ComputeKeyPointsOctTree(
    vector<vector<cv::KeyPoint>> &allKeypoints) {
  // 重新调整图像层数
  allKeypoints.resize(nlevels);

  // 图像cell的尺寸，是个正方形，可以理解为边长in像素坐标
  const float W = 35;

  /// 中间的+号区域是提取后的关键点所在区域，也就是W的大小
  /// 周围环绕的*号区域是在提取FAST特征点时需要往外扩张3个像素的区域
  //  *********************
  //  *********************
  //  *********************
  //  ***+++++++++++++++***
  //  ***+++++++++++++++***
  //  ***+++++++++++++++***
  //  ***+++++++++++++++***
  //  ***+++++++++++++++***
  //  ***+++++++++++++++***
  //  *********************
  //  *********************
  //  *********************

  // 对每一层图像做处理
  // 遍历所有图像
  for (int level = 0; level < nlevels; ++level) {
    // 计算这层图像的坐标边界
    // NOTICE
    // 注意这里是坐标边界，EDGE_THRESHOLD指的是可以提取特征点时去掉的图像边界
    // 这里的3是因为在计算FAST特征点的时候，需要建立一个半径为3的圆
    const int minBorderX = EDGE_THRESHOLD - 3;
    const int minBorderY = minBorderX;
    const int maxBorderX = mvImagePyramid[level].cols - EDGE_THRESHOLD + 3;
    const int maxBorderY = mvImagePyramid[level].rows - EDGE_THRESHOLD + 3;

    // 存储需要进行平均分配的特征点
    vector<cv::KeyPoint> vToDistributeKeys;
    // 一般地都是过量采集，所以这里预分配的空间大小是nfeatures*10
    vToDistributeKeys.reserve(nfeatures * 10);

    // 计算进行特征点提取的图像区域尺寸
    const float width = (maxBorderX - minBorderX);
    const float height = (maxBorderY - minBorderY);

    // 计算网格在当前层的图像有的行数和列数
    const int nCols = width / W;
    const int nRows = height / W;
    // 计算每个图像网格所占的像素行数和列数
    const int wCell = ceil(width / nCols);
    const int hCell = ceil(height / nRows);

    // 开始遍历图像网格，还是以行开始遍历的
    for (int i = 0; i < nRows; i++) {
      // 计算当前网格初始行坐标
      const float iniY = minBorderY + i * hCell;
      // 计算当前网格最大的行坐标，这里的+6=+3+3，即考虑到了多出来3是为了cell边界像素进行FAST特征点提取用
      // 前面的EDGE_THRESHOLD指的应该是提取后的特征点所在的边界，所以minBorderY是考虑了计算半径时候的图像边界
      // 目测一个图像网格的大小是25*25啊
      float maxY = iniY + hCell + 6;

      // 如果初始的行坐标就已经超过了有效的图像边界了，这里的“有效图像”是指原始的、可以提取FAST特征点的图像区域
      // BUG！这里应该是6;因为前后3个像素，则至少需要6个像素。后面对iniX的判断用的是6
      if (iniY >= maxBorderY - 3)
        // 那么就跳过这一行
        continue;
      // 如果图像的大小导致不能够正好划分出来整齐的图像网格，那么就要委屈最后一行了
      if (maxY > maxBorderY) maxY = maxBorderY;

      // 开始列的遍历
      for (int j = 0; j < nCols; j++) {
        // 计算初始的列坐标
        const float iniX = minBorderX + j * wCell;
        // 计算这列网格的最大列坐标，+6的含义和前面相同
        float maxX = iniX + wCell + 6;
        // 判断坐标是否在图像中
        if (iniX >= maxBorderX - 6) continue;
        // 如果最大坐标越界那么委屈一下
        if (maxX > maxBorderX) maxX = maxBorderX;

        // FAST提取兴趣点, 自适应阈值
        // 这个向量存储这个cell中的特征点
        vector<cv::KeyPoint> vKeysCell;
        // 调用opencv的库函数来检测FAST角点
        FAST(
            // 待检测的图像，这里就是当前遍历到的图像块
            mvImagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX),
            vKeysCell,  // 存储角点位置的容器
            iniThFAST,  // 检测阈值
            true);      // 使能非极大值抑制

        // 如果这个图像块中使用默认的FAST检测阈值没有能够检测到角点
        if (vKeysCell.empty()) {
          // 那么就使用更低的阈值来进行重新检测
          FAST(mvImagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX),
               vKeysCell,  // 存储角点位置的容器
               minThFAST,  // 更低的检测阈值
               true);      // 使能非极大值抑制
        }

        // 当图像cell中检测到FAST角点的时候执行下面的语句
        if (!vKeysCell.empty()) {
          // 遍历其中的所有FAST角点
          for (vector<cv::KeyPoint>::iterator vit = vKeysCell.begin();
               vit != vKeysCell.end(); vit++) {
            // NOTICE
            // 到目前为止，这些角点的坐标都是基于图像cell的，现在我们要先将其恢复到当前的【坐标边界】下的坐标
            // 这样做是因为在下面使用八叉树法整理特征点的时候将会使用得到这个坐标
            // 在后面将会被继续转换成为在当前图层的扩充图像坐标系下的坐标
            (*vit).pt.x += j * wCell;
            (*vit).pt.y += i * hCell;
            // 然后将其加入到”等待被分配“的特征点容器中
            vToDistributeKeys.push_back(*vit);
          }  // 遍历图像cell中的所有的提取出来的FAST角点，并且恢复其在整个金字塔当前层图像下的坐标
        }  // 当图像cell中检测到FAST角点的时候执行下面的语句
      }    // 开始遍历图像cell的列
    }      // 开始遍历图像cell的行

    // 声明一个对当前图层的特征点的容器的引用
    vector<cv::KeyPoint> &keypoints = allKeypoints[level];
    // 并且调整其大小为欲提取出来的特征点个数（当然这里也是扩大了的，因为不可能所有的特征点都是在这一个图层中提取出来的）
    keypoints.reserve(nfeatures);

    // 根据mnFeatuvector<KeyPoint> & keypoints =
    // allKeypoints[level];resPerLevel,即该层的兴趣点数,对特征点进行剔除
    // 返回值是一个保存有特征点的vector容器，含有剔除后的保留下来的特征点
    // 得到的特征点的坐标，依旧是在当前图层下来讲的
    keypoints = DistributeOctTree(
        vToDistributeKeys,  // 当前图层提取出来的特征点，也即是等待剔除的特征点
        minBorderX, maxBorderX,  // 当前图层图像的边界
        minBorderY, maxBorderY,
        mnFeaturesPerLevel[level]  // 希望保留下来的当前层图像的特征点个数
    );

    if (static_cast<int>(keypoints.size()) > mnFeaturesPerLevel[level]) {
      keypoints.resize(mnFeaturesPerLevel[level]);
    }

    // PATCH_SIZE是对于底层的初始图像来说的，现在要根据当前图层的尺度缩放倍数得到缩放前对应的PATCH大小
    //  实际上这个值应该没有被用到，特征点方向的计算是在缩放后的图像上计算的
    const int scaledPatchSize = PATCH_SIZE * mvScaleFactor[level];

    // Add border to coordinates and scale information
    // 获取剔除过程后保留下来的特征点数目
    const int nkps = keypoints.size();
    // 然后开始遍历这些特征点，恢复其在当前图层图像坐标系下的坐标
    for (int i = 0; i < nkps; i++) {
      // 对每一个保留下来的特征点，恢复到相对于当前图层“边缘扩充图像下”的坐标系的坐标
      keypoints[i].pt.x += minBorderX;
      keypoints[i].pt.y += minBorderY;
      // 记录特征点来源的图像金字塔图层
      keypoints[i].octave = level;
      // 记录计算方向的patch，缩放后对应的大小， 又被称作为特征点半径
      keypoints[i].size = scaledPatchSize;
    }
  }  // 遍历level

  // compute orientations
  // 然后计算这些特征点的方向信息，注意这里还是分层计算的
  for (int level = 0; level < nlevels; ++level)
    computeOrientation(
        mvImagePyramid[level],  // 对应的图层的图像
        allKeypoints[level],  // 这个图层中提取并保留下来的特征点容器
        umax);                // 以及PATCH的横坐标边界
}  // ComputeKeyPointsOctTree

/**
 * @brief ORB_SLAM1中计算关键点的方法
 * 步骤：
 *  1. 将图像根据目标关键点数量切割成小区域，希望每个区域提取5个特征点
 *  2. 对每个图像区域进行FAST关键点计算(可能有些区域没有足够的特征点)
 *  3.
 * 循环计算各个区域应该保留的关键点数量->当总的特征点数量不够时（缺少的总数不等于0）
 *      3.1 检查关键点数量不够的区域，统计缺少的总数
 *      3.2 将总数分配到剩下的关键点
 *      3.3 在关键点数量足够的区域继续循环
 *  4. 根据计算出的每个区域应该保留的关键点数量，保留区域中分数靠前的关键点
 *  5. 关键点坐标转换到整体图像坐标系
 *  6. 对关键点计算角度
 *
 * @param[out] allKeypoints 所有图层上的关键点
 */
void ORBextractor::ComputeKeyPointsOld(
    std::vector<std::vector<cv::KeyPoint>> &allKeypoints) {
  // 根据图像金字塔的图层数调整这个变量中的图层数
  allKeypoints.resize(nlevels);

  // 计算底层图像的长宽比，其实也就是所有图像的长宽比
  float imageRatio =
      static_cast<float>(mvImagePyramid[0].cols) / mvImagePyramid[0].rows;

  // 开始遍历所有图层的图像
  for (int level = 0; level < nlevels; ++level) {
    // 获取每层图像希望提取出来的特征点
    const int nDesiredFeatures = mnFeaturesPerLevel[level];

    // 计算当前图层中应该有几行cell和几行row
    // 每个cell提取5个关键点。可以看到这里levelCols*levelRows =
    // nDesiredFeatures/5
    const int levelCols =
        sqrt(static_cast<float>(nDesiredFeatures) / (5 * imageRatio));
    // 按照图像比例计算，省事
    const int levelRows = imageRatio * levelCols;

    // 这里指的应该是FAST角点可以存在的坐标位置范围，其实就是原始图像的坐标范围
    // 注意这里没有提前进行+3的操作，而是在后面计算每个网格的区域的时候使用-3的操作来处理FAST角点半径问题
    // 本质上和前面的思想是一样的
    const int minBorderX = EDGE_THRESHOLD;
    const int minBorderY = minBorderX;
    const int maxBorderX = mvImagePyramid[level].cols - EDGE_THRESHOLD;
    const int maxBorderY = mvImagePyramid[level].rows - EDGE_THRESHOLD;

    // 计算这个容许坐标区域的宽度和高度
    const int W = maxBorderX - minBorderX;
    const int H = maxBorderY - minBorderY;
    // 同时计算每个图像cell的宽度和高度
    const int cellW = ceil(static_cast<float>(W) / levelCols);
    const int cellH = ceil(static_cast<float>(H) / levelRows);

    // 计算本层图像中的总cell个数
    const int nCells = levelRows * levelCols;
    // ceil:返回大于或者等于表达式的最小整数，向上取整
    // 这里计算了每个cell中需要提取出来的特征点数量，由于存在小数取整问题，所以都是往多了取整
    const int nfeaturesCell =
        ceil(static_cast<float>(nDesiredFeatures) / nCells);

    // 以方便查找的格式存储从图像cell中提取出来的特征点，
    // 第三层vector-当前cell中的特征点向量
    // 第二层vector-包含了一行cell中，每个cell的上面的特征点容器
    // 第一层vector-包含了所有行的，存储“第二层vector”的容器
    vector<vector<vector<cv::KeyPoint>>> cellKeyPoints(
        levelRows, vector<vector<cv::KeyPoint>>(levelCols));

    // 每个cell中应该保留的特征点数量
    vector<vector<int>> nToRetain(levelRows, vector<int>(levelCols, 0));
    // 每个cell中实际提取出来的特征点的数量
    vector<vector<int>> nTotal(levelRows, vector<int>(levelCols, 0));
    // 每个cell中是否只提取出来了一个特征点的标记
    vector<vector<bool>> bNoMore(levelRows, vector<bool>(levelCols, false));
    // 保存每一个cell图像的x起始坐标和y起始坐标
    vector<int> iniXCol(levelCols);
    vector<int> iniYRow(levelRows);

    // 不再有冗余的关键点的区块计数
    int nNoMore = 0;
    // 存储需要进行分裂的图像cell计数
    int nToDistribute = 0;

    // 考虑到提取FAST特征点的图像半径时，计算cell边界所使用的增量
    // +6=+3+3，代表这起始侧和终止侧提取FAST特征点时的那个3个像素的圆半径
    // 也就是说，这里的cellH指的是有效的、FAST角点可以存在的图像长度
    float hY = cellH + 6;

    // 开始逐行遍历网格
    for (int i = 0; i < levelRows; i++) {
      // 计算用于进行FAST特征点提取的图像cell行边界，这里考虑到了半径为3的半径
      const float iniY = minBorderY + i * cellH - 3;
      // 记录，因为同样的网格分布在后面进行特征点的均匀和过滤操作中同样会用到
      iniYRow[i] = iniY;
      // 如果当前的行是最后一行
      if (i == levelRows - 1) {
        // 计算当前的起始位置到最终的图像终止位置（考虑FAST提取半径）增量
        hY = maxBorderY + 3 - iniY;
        // 如果为负说明这个地方不能够再有一行cell了
        /**
         * NOTICE 如果要是最后一行存在计算的意义，那么要求hY>0,即
         * maxBorderY+3-iniY>0
         * 而前面计算iniY时，实际上考虑到了提取FAST角点所用到的半径为3pixels的半径，所以已经减去了3，这里为了方便分析，不妨
         * 设原始的图像网格的起始坐标为borderY，那么就有：
         * iniY=boderY-3
         * 整理可得，使得最后一行计算有意义的要求为：
         * borderY < maxBorderY+6
         * 然而根据程序前面的计算，这里的borderY是不应该超过maxBorderY的，所以，这里的这个比较是否是没有必要呢？
         */

        if (hY <= 0) continue;
      }

      // 计算从起始的行位置到终止的行位置的坐标增量
      // +6=+3+3,前面的+3是弥补iniY相对于minBorderY +
      // i*cellH的-3，后面的+3则是在minBorderY + （i+1）*cellH
      // 的基础上，又+的3来表示考虑到提取FAST特征点的半径为3的圆
      float hX = cellW + 6;

      // 接下来开始遍历一行中每个列的图像cell
      for (int j = 0; j < levelCols; j++) {
        // 某个cell的x轴起始坐标
        float iniX;

        // 如果这个cell是在第一行
        if (i == 0) {
          // 那么就要计算初始边界
          iniX = minBorderX + j * cellW - 3;
          // 并且保存留备后用
          iniXCol[j] = iniX;
        } else {
          // 如果这个cell不是在第一行，那么就可以直接读取之前计算好的初始坐标了
          iniX = iniXCol[j];
        }  // 判断当前遍历的是否是第一行，是否需要计算初始坐标

        // 判断当前的cell是否是当前行的最后一个cell
        if (j == levelCols - 1) {
          // 如果是的话就要计算它的坐标增量，计算原因和前面是相同的
          hX = maxBorderX + 3 - iniX;
          // 如果这个坐标增量是负的说明这个小网格不能够用来提取FAST特征点，直接跳过（然后就不满足循环条件结束了）
          if (hX <= 0) continue;
        }

        // 从当前图层的图像中扣取这个图像cell的图像
        cv::Mat cellImage = mvImagePyramid[level]
                                .rowRange(iniY, iniY + hY)
                                .colRange(iniX, iniX + hX);

        // 过量提取特征点空间预留，这里预留了当前网格应该提取的特征点个数的5倍
        cellKeyPoints[i][j].reserve(nfeaturesCell * 5);

        // 调用opencv的库函数来检测FAST角点
        FAST(cellImage,            // cell中的图像
             cellKeyPoints[i][j],  // 用于保存提取出来的特征点的vector容器
             iniThFAST,            // 初步的FAST检测阈值
             true);                // 使能非极大值抑制

        // 如果当前cell中提取出来的特征点的个数小于3
        if (cellKeyPoints[i][j].size() <= 3) {
          // 那么首先清空刚才提取出来的特征点
          cellKeyPoints[i][j].clear();
          // 然后使用更小的参数阈值，进行重新提取
          FAST(cellImage,  // cell中的图像
               cellKeyPoints[i]
                            [j],  // 输出变量，用于保存提取出来的特征点的vector
               minThFAST,  // 较小的那个FAST阈值
               true);      // 使能非极大值抑制
        }  // 如果当前cell中提取出来的特征点的个数小于3

        // 得到提取到的特征点数目
        const int nKeys = cellKeyPoints[i][j].size();
        // 记录
        nTotal[i][j] = nKeys;

        // 如果这个数目已经满足了在这个cell中需要提取出来的特征点数目要求了
        if (nKeys > nfeaturesCell) {
          // 那么就将需要保留的特征点数目设置为我们想要这个cell具有的特征点数目
          nToRetain[i][j] = nfeaturesCell;
          // 这个网格中已经不再需要更多的点来补充了
          bNoMore[i][j] = false;
        } else {
          // 如果没有满足，那么也只能保留目前已经有的特征点数目
          nToRetain[i][j] = nKeys;
          // 需要进行分配到这里的特征点数目就是当前提取到的特征点数目和应该提取的数目之间差值，可以理解为特征点个数的缺口
          // 累加
          nToDistribute += nfeaturesCell - nKeys;
          // 置位表示这个cell真正提取到的特征点数目没有被满足，即需要更多的点
          bNoMore[i][j] = true;
          // 计数++
          nNoMore++;
        }  // 判断从当前cell中提取出来的特征点数目是否符合我们的预期要求
      }    // 接下来开始遍历一行中每个列的图像cell
    }      // 接下来开始遍历一行中每个列的图像cell

    // Retain by score
    // 根据评分数值决定哪个特征点将会被留下
    // 进行的条件：
    // 1、有需要匀给少特征点的cell的特征点数
    // 2、不是所有的cell都nomore了
    // 反着说，这个过程停止的条件一是没有需要在进行均匀的特征点个数，二是当前次循环时所有的特征点数目都不足以补充特征点个数的缺口了
    while (nToDistribute > 0 && nNoMore < nCells) {
      // 对于那些在上一轮中特征点个数达到要求的图像cell，计算出在本轮循环中，需要补足本层图像特征点总个数的缺口，
      // 这些cell所需要重新设定的“期望提取出来的”特征点个数
      int nNewFeaturesCell =
          nfeaturesCell +
          ceil(static_cast<float>(
                   nToDistribute) /  // 仍旧需要匀过来的特征点数量缺口
               (nCells - nNoMore));  // 那些满足特征点数目的cell数

      // 由于即使是补充，仍然会有一些cell能够满足第一轮的特征点数目要求，但是不能够满足第二轮（甚至根据情况会有第三轮，第四轮等等）
      // 的数目要求，这个特征点数量的缺口就记录在这个变量中，所以这里要对它进行清零操作
      nToDistribute = 0;

      // 接下来开始遍历每行cell
      for (int i = 0; i < levelRows; i++) {
        // 接着开始遍历一行中每个列中的图像cell
        for (int j = 0; j < levelCols; j++) {
          // 对那些在上一轮提取特征点时满足甚至超过的cell进行操作,就是那些bNoMore[i][j]
          // == false的点
          if (!bNoMore[i][j]) {
            // 判断在上一轮的特征点提取过程中这个cell提取到的特征点数量，是否满足在本轮中的特征点数量要求
            if (nTotal[i][j] > nNewFeaturesCell) {
              // 如果满足的话，那么就重新设置要保存的特征点数量
              nToRetain[i][j] = nNewFeaturesCell;
              // 因为减去在本轮要求的特征点数量之后还有剩余，所以这里设置为false
              bNoMore[i][j] = false;
            } else {
              // 如果不幸不能够满足在本轮中的特征点数目要求，那么也只好将就了
              // 只保存能够提取出来的特征点数目
              nToRetain[i][j] = nTotal[i][j];
              // 这部分特征点数目的缺口也只好由其他的部分来补充了，因此还是累加到这个变量中
              nToDistribute += nNewFeaturesCell - nTotal[i][j];
              // 这个cell中确实没有能够满足本轮，以及可能的、后面的几轮的特征点数目要求了
              bNoMore[i][j] = true;
              // 这个计数++
              nNoMore++;
            }  // 如果当前遍历到的这个点不能够满足在本轮中的特征点数目要求
          }  // 判断上一轮时这个cell是否满足了要提取的特征点数目
        }  // 遍历一行中每个列中的图像cell
      }    // 遍历每行cell
    }      // 判断是否达到了停止条件

    // 请注意，执行到这里，只是完成了每个图像cell中特征点的提取+每个cell中应该保留的特征点数量，下面才是正式地开始对每个
    // cell中vector中的特征点进行删减——或者说是过滤

    // 声明一个对当前图层中特征点vector容器的引用
    vector<cv::KeyPoint> &keypoints = allKeypoints[level];
    // 预分配2倍的期望特征点空间（因为实际的提取过程中，我们都是按照稍微超过期望值的特征点数目进行操作的）
    keypoints.reserve(nDesiredFeatures * 2);

    // 计算图像金字塔每一层图像的PATCH在原始分辨率下对应的PATCH的大小，但是这个变量没有被使用
    const int scaledPatchSize = PATCH_SIZE * mvScaleFactor[level];

    // Retain by score and transform coordinates
    // 根据响应值保留符合要求的特征点，并且进行坐标的转换
    // 开始遍历每行的cell
    for (int i = 0; i < levelRows; i++) {
      // 遍历某行中每列的cell
      for (int j = 0; j < levelCols; j++) {
        // 获取当前遍历到的cell所提取出来的特征点vector容器句柄
        vector<cv::KeyPoint> &keysCell = cellKeyPoints[i][j];
        // 这里是调用了opencv函数，根据特征点的评分（也就是响应值），保留一个cell中指定数量的特征点
        cv::KeyPointsFilter::retainBest(
            keysCell,  // 输入输出，用于提供待保留的特征点vector，操作完成后将保留的特征点存放在里面
                       // 其实就是把那些响应值低的点从vector中给删除了
            nToRetain[i][j]);  // 指定要保留的特征点数目
        // 如果由于小数取整等原因（前面都是向多了取整的），【对于当前的cell】，经过去除之后的特征点数目还是大于之前设定的要保留的特征点数目
        if (static_cast<int>(keysCell.size()) > nToRetain[i][j])
          // 那么就强制丢弃vector后面的特征点以满足保留的特征点数目要求
          keysCell.resize(nToRetain[i][j]);

        // 遍历剔除后的特征点vector，进行坐标的转换，以及添加相关的信息
        // NOTICE 这里的特征点还只是一个cell中的，其坐标也是在一个cell下的坐标
        for (size_t k = 0, kend = keysCell.size(); k < kend; k++) {
          // 转换坐标（这里就用上了之前保存的坐标）
          keysCell[k].pt.x += iniXCol[j];
          keysCell[k].pt.y += iniYRow[i];
          // 设置提取到该特征点的图层
          keysCell[k].octave = level;
          // 记录这个patch在这个图层上的大小
          keysCell[k].size = scaledPatchSize;
          // keypoints是对allKeypoints中存储当前图层所有特征点的vector的一个引用
          // 这里将转换后的结果追加到表示最终结果的vector上
          keypoints.push_back(keysCell[k]);
        }  // 遍历剔除后的特征点vector，进行坐标的转换，以及添加相关的信息
      }    // 遍历某行中每列的cell
    }      // 开始遍历每行的cell

    // 如果经过上面的剔除操作之后，最终，当前图层的特征点数目还是大于期望的
    // 和上面的不同，上面判断的是每个图像cell中的，这里则是判断整个图层中的特征点数目
    if (static_cast<int>(keypoints.size()) > nDesiredFeatures) {
      // 那么就再调用opencv的库函数对这组特征点按照其响应值进行一次剔除
      cv::KeyPointsFilter::retainBest(keypoints, nDesiredFeatures);
      // 并且强制丢弃还是超过期望结果的特征点
      keypoints.resize(nDesiredFeatures);
    }  // 如果最终结果中，本层图像的的特征点数目还是大于希望的，那么就再剔除一次
  }    // 遍历每个图层

  // and compute orientations
  // 最后计算这些特征点的方向信息
  // 遍历图像金字塔中的每个图层
  for (int level = 0; level < nlevels; ++level)
    // 计算这个图层所有特征点的方向信息
    computeOrientation(mvImagePyramid[level],  // 这个图层的图像
                       allKeypoints[level],  // 这个图层的特征点对象vector容器
                       umax);                // patch区域的边界
}

// 注意这是一个不属于任何类的全局静态函数，static修饰符限定其只能够被本文件中的函数调用
/**
 * @brief 计算某层金字塔图像上特征点的描述子
 *
 * @param[in] image                 某层金字塔图像
 * @param[in] keypoints             特征点vector容器
 * @param[out] descriptors          描述子
 * @param[in] pattern               计算描述子使用的固定随机点集
 */
static void computeDescriptors(const cv::Mat &image,
                               vector<cv::KeyPoint> &keypoints,
                               cv::Mat &descriptors,
                               const vector<cv::Point> &pattern) {
  // 清空保存描述子信息的容器,32x8=256位
  descriptors = cv::Mat::zeros(keypoints.size(), 32, CV_8UC1);

  // 开始遍历特征点
  for (size_t i = 0; i < keypoints.size(); i++)
    // 计算这个特征点的描述子
    computeOrbDescriptor(keypoints[i],  // 要计算描述子的特征点
                         image,         // 以及其图像
                         &pattern[0],   // 随机点集的首地址
                         descriptors.ptr(i));  // 提取出来的描述子的保存位置
}

/**
 * @brief 用仿函数（重载括号运算符）方法来计算图像特征点
 *
 * @param[in] _image                    输入原始图的图像
 * @param[in] _mask                     掩膜mask
 * @param[in & out] _keypoints                存储特征点关键点的向量
 * @param[in & out] _descriptors              存储特征点描述子的矩阵
 */
int ORBextractor::operator()(cv::InputArray _image, cv::InputArray _mask,
                             vector<cv::KeyPoint> &_keypoints,
                             cv::OutputArray _descriptors,
                             std::vector<int> &vLappingArea) {
  // Step 1 检查图像有效性。如果图像为空，那么就直接返回
  if (_image.empty()) return -1;
  (void)_mask;  // suppress unused parameter warning
  // 获取图像的大小
  cv::Mat image = _image.getMat();
  // 判断图像的格式是否正确，要求是单通道灰度值
  assert(image.type() == CV_8UC1);

  // Pre-compute the scale pyramid
  // Step 2 构建图像金字塔
  ComputePyramid(image);

  // Step 3
  // 计算图像的特征点，并且将特征点进行均匀化。均匀的特征点可以提高位姿计算精度
  // 存储所有的特征点，注意此处为二维的vector，第一维存储的是金字塔的层数，第二维存储的是那一层金字塔图像里提取的所有特征点
  vector<vector<cv::KeyPoint>> allKeypoints;
  // 使用四叉树的方式计算每层图像的特征点并进行分配
  ComputeKeyPointsOctTree(allKeypoints);

  // 使用传统的方法提取并平均分配图像的特征点，作者并未使用
  // ComputeKeyPointsOld(allKeypoints);

  // Step 4 拷贝图像描述子到新的矩阵descriptors
  cv::Mat descriptors;

  // 统计整个图像金字塔中的特征点
  int nkeypoints = 0;
  // 开始遍历每层图像金字塔，并且累加每层的特征点个数
  for (int level = 0; level < nlevels; ++level)
    nkeypoints += static_cast<int>(allKeypoints[level].size());

  // 如果本图像金字塔中没有任何的特征点
  if (nkeypoints == 0) {
    // 通过调用cv::mat类的.realse方法，强制清空矩阵的引用计数，这样就可以强制释放矩阵的数据了
    // 参考[https://blog.csdn.net/giantchen547792075/article/details/9107877]
    _descriptors.release();
  } else {
    // 如果图像金字塔中有特征点，那么就创建这个存储描述子的矩阵，注意这个矩阵是存储整个图像金字塔中特征点的描述子的
    _descriptors.create(nkeypoints,  // 矩阵的行数，对应为特征点的总个数
                        32,  // 矩阵的列数，对应为使用32*8=256位描述子
                        CV_8U);  // 矩阵元素的格式
    // 获取这个描述子的矩阵信息
    // 这里获取的是输出的OuputArray的Mat引用
    descriptors = _descriptors.getMat();
  }

  _keypoints = vector<cv::KeyPoint>(nkeypoints);
  // 因为遍历是一层一层进行的，但是描述子那个矩阵是存储整个图像金字塔中特征点的描述子，所以在这里设置了Offset变量来保存“寻址”时的偏移量，
  // 辅助进行在总描述子mat中的定位
  int offset = 0;
  // Modified for speeding up stereo fisheye matching
  int monoIndex = 0, stereoIndex = nkeypoints - 1;
  for (int level = 0; level < nlevels; ++level) {
    // 获取在allKeypoints中当前层特征点容器的句柄
    vector<cv::KeyPoint> &keypoints = allKeypoints[level];
    // 本层的特征点数
    int nkeypointsLevel = keypoints.size();

    if (nkeypointsLevel == 0) continue;

    // preprocess the resized image
    //  Step 5 对图像进行高斯模糊
    // 深拷贝当前金字塔所在层级的图像
    cv::Mat workingMat = mvImagePyramid[level].clone();

    // 注意：提取特征点的时候，使用的是清晰的原图像；这里计算描述子的时候，为了避免图像噪声的影响，使用了高斯模糊
    GaussianBlur(workingMat,  // 源图像
                 workingMat,  // 输出图像
                 cv::Size(7, 7),  // 高斯滤波器kernel大小，必须为正的奇数
                 2,               // 高斯滤波在x方向的标准差
                 2,               // 高斯滤波在y方向的标准差
                 cv::BORDER_REFLECT_101);  // 边缘拓展点插值类型

    // Compute the descriptors
    // desc存储当前图层的描述子
    // Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);
    cv::Mat desc = cv::Mat(nkeypointsLevel, 32, CV_8U);
    // Step 6 计算高斯模糊后图像的描述子
    computeDescriptors(workingMat,  // 高斯模糊之后的图层图像
                       keypoints,   // 当前图层中的特征点集合
                       desc,        // 存储计算之后的描述子
                       pattern);    // 随机采样点集

    // 更新偏移量的值(未使用的变量)
    offset += nkeypointsLevel;

    // Scale keypoint coordinates
    // Step 6 对非第0层图像中的特征点的坐标恢复到第0层图像（原图像）的坐标系下
    // ? 得到所有层特征点在第0层里的坐标放到_keypoints里面
    // 对于第0层的图像特征点，他们的坐标就不需要再进行恢复了
    float scale = mvScaleFactor[level];
    int i = 0;
    for (vector<cv::KeyPoint>::iterator keypoint = keypoints.begin(),
                                        keypointEnd = keypoints.end();
         keypoint != keypointEnd; ++keypoint) {
      // Scale keypoint coordinates
      if (level != 0) {
        // 特征点本身直接乘缩放倍数就可以了
        keypoint->pt *= scale;
      }
      // TODO(zeal) vLappingArea，这里还不知道为什么
      // 这里的lapping选择只有在单目模式下会传入0,1000;但是似乎没有被用到，可能是没有用的参数
      if (keypoint->pt.x >= vLappingArea[0] &&
          keypoint->pt.x <= vLappingArea[1]) {
        _keypoints.at(stereoIndex) = (*keypoint);
        desc.row(i).copyTo(descriptors.row(stereoIndex));
        stereoIndex--;
      } else {
        _keypoints.at(monoIndex) = (*keypoint);
        desc.row(i).copyTo(descriptors.row(monoIndex));
        monoIndex++;
      }
      i++;
    }
  }
  // cout << "[ORBextractor]: extracted " << _keypoints.size() << " KeyPoints"
  // << endl;
  return monoIndex;
}

/**
 * @brief 构建图像金字塔
 * @param image
 * 输入原图像，这个输入图像所有像素都是有效的，也就是说都是可以在其上提取出FAST角点的
 */
void ORBextractor::ComputePyramid(cv::Mat image) {
  for (int level = 0; level < nlevels; ++level) {
    float scale = mvInvScaleFactor[level];
    cv::Size sz(cvRound(image.cols * scale), cvRound(image.rows * scale));
    cv::Size wholeSize(sz.width + EDGE_THRESHOLD * 2,
                       sz.height + EDGE_THRESHOLD * 2);
    cv::Mat temp(wholeSize, image.type()), masktemp;
    // 这里要明白opencv的内存组织，cvMat只是数据头，指向一段内存，类似与shared_ptr的概念
    // 所以这里的mvImagePyramid指向了temp的中间区域
    mvImagePyramid[level] =
        temp(cv::Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

    // Compute the resized image
    // 计算第0层以上resize后的图像
    if (level != 0) {
      // 将上一层金字塔图像根据设定sz缩放到当前层级
      cv::resize(
          mvImagePyramid[level - 1],  // 输入图像
          mvImagePyramid[level],      // 输出图像
          sz,                         // 输出图像的尺寸
          0,  // 水平方向上的缩放系数，留0表示自动计算
          0,  // 垂直方向上的缩放系数，留0表示自动计算
          cv::INTER_LINEAR);  // 图像缩放的差值算法类型，这里的是线性插值算法

      // 把源图像拷贝到目的图像的中央，四面填充指定的像素。图片如果已经拷贝到中间，只填充边界
      // 这个操作有点不太明白，这样的操作结果似乎只是temp指向的内存的值被初始化一下，mvImagePyramid[level]
      // 指向的temp的中间图像，这样如果直接在mvImagePyramid做高斯卷积的化，边缘的点也有值
      // 但是在后面的高斯模块函数已经有了border的的参数，所以这里的操作感觉有点多余
      // BORDER_ISOLATED表示只使用原图像的像素（即使原图像是ROI）
      cv::copyMakeBorder(
          mvImagePyramid[level],  // 源图像
          temp,  // 目标图像（此时其实就已经有大了一圈的尺寸了）
          EDGE_THRESHOLD, EDGE_THRESHOLD,  // top & bottom 需要扩展的border大小
          EDGE_THRESHOLD, EDGE_THRESHOLD,  // left & right 需要扩展的border大小
          cv::BORDER_REFLECT_101 +
              cv::BORDER_ISOLATED);  // 扩充方式，opencv给出的解释：

      //  https://docs.opencv.org/3.4.4/d2/de8/group__core__array.html#ga2ac1049c2c3dd25c2b41bffe17658a36
      /*Various border types, image boundaries are denoted with '|'
       * BORDER_REPLICATE:     aaaaaa|abcdefgh|hhhhhhh
       * BORDER_REFLECT:       fedcba|abcdefgh|hgfedcb
       * BORDER_REFLECT_101:   gfedcb|abcdefgh|gfedcba
       * BORDER_WRAP:          cdefgh|abcdefgh|abcdefg
       * BORDER_CONSTANT:      iiiiii|abcdefgh|iiiiiii  with some specified 'i'
       * BORDER_ISOLATED       !< do not look outside of ROI
       */
    } else {
      // 对于底层图像，直接就扩充边界了
      cv::copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD,
                         EDGE_THRESHOLD, EDGE_THRESHOLD,
                         cv::BORDER_REFLECT_101);
    }
  }
}
}  // namespace ORB_SLAM_Tracking

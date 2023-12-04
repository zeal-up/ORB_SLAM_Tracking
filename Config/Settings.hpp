#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>


namespace ORB_SLAM_Tracking {

class Settings {
 public:
  // delete the default constructor
  Settings() = delete;

  // 使用给定的文件路径构造Settings对象，并调用readSettings()函数读取设置
  Settings(const std::string& filePath) : mfilePath_(filePath) {
    readSettings();
  }

  void readSettings() {
    cv::FileStorage fsSettings(mfilePath_, cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
      std::cerr << "ERROR: Wrong path to settings" << std::endl;
      exit(-1);
    } else {
      std::cout << "Reading settings file at " << mfilePath_ << std::endl;
    }

    // Camera calibration parameters
    fx = fsSettings["Camera.fx"];
    fy = fsSettings["Camera.fy"];
    cx = fsSettings["Camera.cx"];
    cy = fsSettings["Camera.cy"];
    mK = (cv::Mat_<float>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    // Camera distortion parameters
    k1 = fsSettings["Camera.k1"];
    k2 = fsSettings["Camera.k2"];
    p1 = fsSettings["Camera.p1"];
    p2 = fsSettings["Camera.p2"];
    k3 = 0;
    k3 = fsSettings["Camera.k3"];
    if (k3 != 0) {
      mDistCoef = (cv::Mat_<float>(4, 1) << k1, k2, p1, p2, k3);
    } else {
      mDistCoef = (cv::Mat_<float>(4, 1) << k1, k2, p1, p2);
    }
      

    // Camera frames per second
    fps = fsSettings["Camera.fps"];

    // Max/Min frames to insert keyframes
    minFrames = 0;
    maxFrames = 18 * fps / 30;

    // print camera settings
    std::cout << "Camera Parameters: " << std::endl;
    std::cout << "- fx: " << fx << std::endl;
    std::cout << "- fy: " << fy << std::endl;
    std::cout << "- cx: " << cx << std::endl;
    std::cout << "- cy: " << cy << std::endl;
    std::cout << "- k1: " << k1 << std::endl;
    std::cout << "- k2: " << k2 << std::endl;
    std::cout << "- p1: " << p1 << std::endl;
    std::cout << "- p2: " << p2 << std::endl;
    std::cout << "- fps: " << fps << std::endl;


    // Color order of the images (0: BGR, 1: RGB. It is ignored if images are
    // grayscale)
    int nRGB = fsSettings["Camera.RGB"];
    bRGB = nRGB;
    if (bRGB) {
      std::cout << "- color order: RGB (ignored if grayscale)" << std::endl;
    } else {
      std::cout << "- color order: BGR (ignored if grayscale)" << std::endl;
    }

    // ORB parameters
    nFeatures = fsSettings["ORBextractor.nFeatures"];
    scaleFactor = fsSettings["ORBextractor.scaleFactor"];
    nLevels = fsSettings["ORBextractor.nLevels"];
    iniThFAST = fsSettings["ORBextractor.iniThFAST"];
    minThFAST = fsSettings["ORBextractor.minThFAST"];

    // print ORB settings
    std::cout << std::endl << "ORB Parameters: " << std::endl;
    std::cout << "- nFeatures: " << nFeatures << std::endl;
    std::cout << "- scaleFactor: " << scaleFactor << std::endl;
    std::cout << "- nLevels: " << nLevels << std::endl;
    std::cout << "- iniThFAST: " << iniThFAST << std::endl;
    std::cout << "- minThFAST: " << minThFAST << std::endl;

    fsSettings.release();

    std::cout << "Settings read." << std::endl << std::endl;

  }

 private:
  std::string mfilePath_;

 public:
  // Camera calibration parameters
  float fx, fy, cx, cy;
  cv::Mat mK;

  // Camera distortion parameters
  float k1, k2, p1, p2, k3;
  cv::Mat mDistCoef;

  // Camera frames per second
  float fps;

  // Color order of the images (0: BGR, 1: RGB. It is ignored if images are
  // grayscale)
  bool bRGB;

  // ORB parameters
  int nFeatures;      // Number of features per image
  float scaleFactor;  // Scale factor between levels in the scale pyramid
  int nLevels;        // Number of levels in the scale pyramid
  int iniThFAST;      // ORB iniThFAST; First try to extract features with this threshold
  int minThFAST;      // ORB minThFAST; If extract too few features, decrease threshold as this

  // Max/Min frames to insert keyframes
  int maxFrames;
  int minFrames;
};

}  // namespace ORB_SLAM_Tracking

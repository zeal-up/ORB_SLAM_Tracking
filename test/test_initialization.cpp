#include "Initialization/Initializer.hpp"
#include "SlamTypes/Frame.hpp"
#include "DebugUtils.hpp"
#include <opencv2/core/eigen.hpp>

using namespace ORB_SLAM_Tracking;



int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "Usage: ./test_initialization data_path" << std::endl;
        return 1;
    }
    std::string data_path = argv[1];
    std::string kpKf1Path = data_path + "/keypoints_kf1.txt";
    std::string kpKf2Path = data_path + "/keypoints_kf2.txt";
    std::string p3dPath = data_path + "/points3D.txt";
    std::string posePath = data_path + "/pose.txt";
    std::string intrinsicPath = data_path + "/intrinsic.txt";
    std::string fundamentalPath = data_path + "/fundamental.txt";

    std::vector<cv::KeyPoint> kpKf1, kpKf2;
    std::vector<cv::Point3f> p3d;
    cv::Mat poseCv, intrinsicCv;
    cv::Mat fundamentalMat;
    bool readSuccess = true;
    readSuccess &= DebugUtils::ReadKeyPoints(kpKf1Path, kpKf1);
    readSuccess &= DebugUtils::ReadKeyPoints(kpKf2Path, kpKf2);
    readSuccess &= DebugUtils::ReadPoints3D(p3dPath, p3d);
    readSuccess &= DebugUtils::ReadPoseMat(posePath, poseCv);
    readSuccess &= DebugUtils::ReadIntrinsicMat(intrinsicPath, intrinsicCv);
    readSuccess &= DebugUtils::ReadIntrinsicMat(fundamentalPath, fundamentalMat);
    
    std::cout << "kpKf1.size = " << kpKf1.size() << std::endl;
    std::cout << "kpKf2.size = " << kpKf2.size() << std::endl;
    std::cout << "p3d.size = " << p3d.size() << std::endl;
    std::cout << "poseCv = " << poseCv << std::endl;
    std::cout << "intrinsicCv = " << intrinsicCv << std::endl;
    if (!readSuccess) {
        std::cerr << "Read data failed!" << std::endl;
        return 1;
    }

    cv::Mat R21 = poseCv.rowRange(0, 3).colRange(0, 3);
    cv::Mat t21 = poseCv.rowRange(0, 3).col(3);
    std::vector<std::pair<int,int>> matches;
    std::vector<bool> matchesInliers;
    for (int i = 0; i < kpKf1.size(); i++) {
        matches.push_back(std::make_pair(i, i));
        matchesInliers.push_back(true);
    }

    float thresh = 4.0;
    // for output variables
    std::vector<cv::Point3f> p3dOut;
    float parallaxOut;
    std::vector<bool> vbTriangulated;
    int triNum = 0;

    triNum = Initializer::CheckRT(R21, t21, kpKf1, kpKf2, matches, matchesInliers, intrinsicCv, p3dOut, thresh, vbTriangulated, parallaxOut);

    // print output variables
    std::cout << "triNum = " << triNum << std::endl;
    std::cout << "p3dOut.size = " << p3dOut.size() << std::endl;
    std::string p3dOutPath = data_path + "/p3dOut.txt";
    DebugUtils::SavePoints3D(p3dOutPath, p3dOut);

    if (triNum != p3d.size()) {
        std::cerr << "triNum != p3d.size()" << std::endl;
        return 1;
    }
    for (int i = 0; i < triNum; i++) {
        cv::Point3f& p3dOrg = p3d[i];
        cv::Point3f& p3dNew = p3dOut[i];
        if (std::abs(p3dOrg.x - p3dNew.x) > 0.1 ||
            std::abs(p3dOrg.y - p3dNew.y) > 0.1 ||
            std::abs(p3dOrg.z - p3dNew.z) > 0.1) {
            std::cerr << "Point i = " << i << " is not equal!" << " p3dOrg = " << p3dOrg << " p3dNew = " << p3dNew << std::endl;
            return 1;
        }
    }


    // ------------------------------ test with frame ------------------------------
    Frame frame1, frame2;
    frame1.mvKeysUn = kpKf1;
    frame2.mvKeysUn = kpKf2;
    frame1.mK = intrinsicCv;
    frame2.mK = intrinsicCv;

    Initializer initializer(frame1, 1.0, 400);
    PoseT T21;
    std::vector<bool> vbTriangulated2;
    std::vector<cv::Point3f> vP3D;
    std::vector<int> matches12;
    for (int i = 0; i < kpKf1.size(); i++) {
        matches12.push_back(i);
    }
    initializer.Initialize(frame2, matches12, T21, vP3D, vbTriangulated2);

    std::cout << "FundaMatrix = " << initializer.mF << std::endl;
    DebugUtils::SaveIntrinsicMat(data_path + "/fundamentalOut.txt", initializer.mF);
    cv::Mat Fnorm = initializer.mF;
    Fnorm /= Fnorm.at<float>(2, 2);
    std::cout << "Fnorm = " << Fnorm << std::endl;
    DebugUtils::SaveIntrinsicMat(data_path + "/fundamentalOutNorm.txt", Fnorm);

    cv::Mat ForgNorm;
    ForgNorm = fundamentalMat / fundamentalMat.at<float>(2, 2);
    std::cout << "ForgNorm = " << ForgNorm << std::endl;
    DebugUtils::SaveIntrinsicMat(data_path + "/fundamentalNorm.txt", ForgNorm);

    // ------------------------- test checkFundamental -------------------------
    // test CheckFundamental
    std::vector<bool> vbMatchesInliers;
    float sigma = 1.0;
    Eigen::Matrix3f fundamentalMatEigen;
    cv::cv2eigen(fundamentalMat, fundamentalMatEigen);
    float score = initializer.CheckFundamental(fundamentalMatEigen, vbMatchesInliers, sigma);
    int inlierNum = 0;
    inlierNum = std::count(vbMatchesInliers.begin(), vbMatchesInliers.end(), true);
    std::cout << "score = " << score << std::endl;
    std::cout << "inlierNum = " << inlierNum << std::endl;


    return 0;
}
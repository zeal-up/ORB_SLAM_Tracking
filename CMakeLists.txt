cmake_minimum_required(VERSION 2.8.12)
project(orb_slam_tracking)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)

if (CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.1)
  message(FATAL_ERROR "Insufficient g++ version, must >= 9.1 to support filesystem")

endif()

# Set the build type to Release (with optimizations) by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT BUILD_TEST)
  set(BUILD_TEST OFF)
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

# Find gtest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Find glog
set(Glog_DIR "/usr/lib/x86_64-linux-gnu/cmake/glog")
find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})
# message GLOG_FOUND
message(STATUS "GLOG_FOUND: ${Glog_FOUND}")

# 添加DBoW2库
add_subdirectory(Thirdparty/DBoW2)
include_directories(${DBoW2_INCLUDE_DIRS})

# 添加DLib库
add_subdirectory(Thirdparty/DLib)
include_directories(${DLib_INCLUDE_DIRS})

# 添加g2o库
option(G2O_BUILD_EXAMPLES "Build g2o examples" OFF)
add_subdirectory(Thirdparty/g2o-20230806_git)

# 将项目目录添加到头文件搜索路径
include_directories(${PROJECT_SOURCE_DIR})

# 添加SlamTypes库
include_directories(SlamTypes)
include_directories(Features)
add_library(
  SlamTypesLib
  SlamTypes/MapPoint.cpp
  SlamTypes/Frame.cpp
  SlamTypes/KeyFrame.cpp
  SlamTypes/Map.cpp
  SlamTypes/BuildLocalMap.cpp
  Features/ORBextractor.cpp
  Features/ORBmatcher.cpp)
target_link_libraries(SlamTypesLib ${OpenCV_LIBS} ${EIGEN3_LIBS} glog DBoW2 DLib UtilsLib)

# 添加Initialization库
include_directories(Initialization)
add_library(
  InitializationLib
  Initialization/Initializer.cpp)
target_link_libraries(InitializationLib ${OpenCV_LIBS} ${EIGEN3_LIBS} SlamTypesLib glog)

# 添加Utils库
include_directories(Utils)
add_library(
  UtilsLib
  Utils/Converter.cpp
  Utils/UtilsCV.cpp)
target_link_libraries(UtilsLib ${OpenCV_LIBS} ${EIGEN3_LIBS})

# Build the demo_ORBextractor executable
add_executable(demo_ORBextractor demo/demo_ORBextractor.cpp)
target_link_libraries(
  demo_ORBextractor
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  DBoW2
  SlamTypesLib
)

# Build the demo_initialization executable
add_executable(demo_initialization demo/demo_initialization.cpp)
target_link_libraries(
  demo_initialization
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  glog
  DBoW2
  DLib
  SlamTypesLib
  InitializationLib
  UtilsLib
)

# ORB_SLAM2 的原始实现
add_library(
  ORB_SLAM2_Lib
  SlamTypes/Frame2.cc
  Features/ORBextractor2.cc
  Features/ORBmatcher2.cc
  Initialization/Initializer2.cc
)
target_link_libraries(
  ORB_SLAM2_Lib
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  DBoW2
  DLib
  UtilsLib
)
add_executable(demo_initialization_origin demo/demo_initialization_origin.cpp)
target_link_libraries(
  demo_initialization_origin
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  glog
  DBoW2
  DLib
  SlamTypesLib
  InitializationLib
  UtilsLib
  ORB_SLAM2_Lib
)

# Build tests
if(BUILD_TEST)
  add_subdirectory(test)
endif()


# copy orb extractor test image
file(COPY demo/orbtest.png DESTINATION ${CMAKE_BINARY_DIR})
cmake_minimum_required(VERSION 2.8.12)
project(orb_slam_tracking)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 20)

# Set the build type to Release (with optimizations) by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
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

# 添加DBoW2库
add_subdirectory(Thirdparty/DBoW2)
include_directories(${DBoW2_INCLUDE_DIRS})
# 打印DBoW2库的头文件路径
message(STATUS "DBoW2_INCLUDE_DIRS: ${DBoW2_INCLUDE_DIRS}")

# 添加DLib库
add_subdirectory(Thirdparty/DLib)
include_directories(${DLib_INCLUDE_DIRS})

# 将项目目录添加到头文件搜索路径
include_directories(${PROJECT_SOURCE_DIR})

# 添加SlamTypes库
include_directories(SlamTypes)
add_library(
  SlamTypesLib
  SlamTypes/Frame.cpp)
target_link_libraries(SlamTypesLib ${OpenCV_LIBS} ${EIGEN3_LIBS} DBoW2 DLib)

# 添加Features库
# 设置Features库的头文件路径
include_directories(Features)
add_library(
  FeaturesLib
  Features/ORBextractor.cpp
  Features/ORBmatcher.cpp)
target_link_libraries(FeaturesLib ${OpenCV_LIBS} DBoW2 SlamTypesLib)

# 添加Initialization库
include_directories(Initialization)
add_library(
  InitializationLib
  Initialization/Initializer.cpp)
target_link_libraries(InitializationLib ${OpenCV_LIBS} ${EIGEN3_LIBS} SlamTypesLib)

# 添加Utils库
include_directories(Utils)
add_library(
  UtilsLib
  Utils/Converter.cpp)
target_link_libraries(UtilsLib ${OpenCV_LIBS} ${EIGEN3_LIBS})

# Build the demo_ORBextractor executable
add_executable(demo_ORBextractor demo/demo_ORBextractor.cpp)
target_link_libraries(
  demo_ORBextractor
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  DBoW2
  FeaturesLib
)

# Build the demo_initialization executable
add_executable(demo_initialization demo/demo_initialization.cpp)
target_link_libraries(
  demo_initialization
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  DBoW2
  DLib
  SlamTypesLib
  FeaturesLib
  InitializationLib
  UtilsLib
)

# 将初始化demo的两张图片复制到编译后的二进制目录
# file(COPY demo/init_01.png DESTINATION ${CMAKE_BINARY_DIR})
# file(COPY demo/init_02.png DESTINATION ${CMAKE_BINARY_DIR})
file(COPY demo/initImages DESTINATION ${CMAKE_BINARY_DIR})

# 将Settings.yaml复制到编译后的二进制目录
file(COPY Settings.yaml DESTINATION ${CMAKE_BINARY_DIR})
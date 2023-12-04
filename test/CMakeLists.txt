cmake_minimum_required(VERSION 2.8.12)
project(module_test)

# 打印当前test编译信息
message("Build test, CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ------------------ test orb_extractor ------------------

# add orb_extractor lib
add_library(org_orb_extractor org_orb_extractor_3/ORBextractor.h org_orb_extractor_3/ORBextractor.cc)
target_link_libraries(org_orb_extractor ${OpenCV_LIBS})

# add orb_extractor test
add_executable(org_orb_extractor_test test_orb_extractor.cpp)
target_link_libraries(org_orb_extractor_test org_orb_extractor)
target_link_libraries(org_orb_extractor_test SlamTypesLib)

# copy init_01.png to output dir
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/init_01.png DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# ------------------ test initialization ------------------
add_library(DebugUtils DebugUtils.hpp DebugUtils.cpp)
add_executable(test_initialization test_initialization.cpp)
target_link_libraries(
  test_initialization 
  DebugUtils
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  glog
  DBoW2
  DLib
  SlamTypesLib
  InitializationLib
  UtilsLib)

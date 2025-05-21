cmake_minimum_required(VERSION 3.0.2)
project(icp_node)

## 找到 catkin 包
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  nav_msgs
  sensor_msgs
  tf
  pcl_conversions
  pcl_ros
  octomap_ros
  octomap_msgs
)

## 找到 Eigen3
find_package(Eigen3 REQUIRED)

## 找到 PCL，尤其是 visualization 模块
find_package(PCL REQUIRED COMPONENTS common io visualization)

## catkin 包信息
catkin_package()

## 包含头文件
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
)

## 链接库目录（可选）
link_directories(${PCL_LIBRARY_DIRS})

## 编译目标
add_executable(icp_node src/gobal_location_icp.cpp)

## 链接库
target_link_libraries(icp_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

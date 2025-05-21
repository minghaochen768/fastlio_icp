#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <octomap/octomap.h>
#include <octomap/ColorOcTree.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>

// —— 全局 OctoMap 对象 & 发布者 —— 
std::shared_ptr<octomap::OcTree> octree;
ros::Publisher octomap_pub;

ros::Publisher octomap_cloud_pub;

ros::Publisher cloud_registered_map_pub;

bool icp_init=0;

bool icp_trigger = false;
Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();



// 全局变量
nav_msgs::Odometry     latest_odom;      // 最近一次收到的里程计
pcl::PCLPointCloud2    latest_cloud2;    // 最近一次收到的点云
bool                   has_odom = false,
                       has_cloud = false,
                       stop_icp = false; // ICP 停止标志

// 地图与 ICP
pcl::PointCloud<pcl::PointXYZ>::Ptr global_map(new pcl::PointCloud<pcl::PointXYZ>);

pcl::PointCloud<pcl::PointXYZ> global_map_octo;

pcl::IterativeClosestPoint<pcl::PointXYZ,pcl::PointXYZ> icp;
Eigen::Matrix4f        T_reloc_saved = Eigen::Matrix4f::Identity();

// ROS 发布者
ros::Publisher odom_pub;     // /odom_reloc
ros::Publisher path_pub;     // /path_reloc
ros::Publisher aligned_pub;  // /aligned_cloud
ros::Publisher map_pub;      // /global_map
nav_msgs::Path path_msg;     // 累积轨迹

// 工具：Odometry→Eigen
Eigen::Matrix4f odomToEigen(const nav_msgs::Odometry& odom) {
  Eigen::Affine3f pose = 
      Eigen::Translation3f(odom.pose.pose.position.x,
                          odom.pose.pose.position.y,
                          odom.pose.pose.position.z)
    * Eigen::Quaternionf(odom.pose.pose.orientation.w,
                        odom.pose.pose.orientation.x,
                        odom.pose.pose.orientation.y,
                        odom.pose.pose.orientation.z);
  return pose.matrix();
}

// 工具：Eigen→Odometry
void eigenToOdom(nav_msgs::Odometry& odom, const Eigen::Matrix4f& M) {
  Eigen::Affine3f a(M);
  Eigen::Vector3f t = a.translation();
  Eigen::Quaternionf q(a.rotation());
  odom.pose.pose.position.x    = t.x();
  odom.pose.pose.position.y    = t.y();
  odom.pose.pose.position.z    = t.z();
  odom.pose.pose.orientation.x = q.x();
  odom.pose.pose.orientation.y = q.y();
  odom.pose.pose.orientation.z = q.z();
  odom.pose.pose.orientation.w = q.w();
  odom.header.stamp = ros::Time::now();
}

// 回调：/stop_icp
void stopIcpCallback(const std_msgs::Bool::ConstPtr& msg) {
  if (msg->data) {
    stop_icp = true;
    T_reloc_saved = icp.getFinalTransformation();
    ROS_INFO_STREAM("[STOP] ICP manually stopped. Saved T_reloc:\n" << T_reloc_saved);
  } else if (!msg->data) {
    stop_icp = false;
    ROS_INFO("[RESUME] ICP manually resumed.");
  }
}


// 回调：/cloud_registered
void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
  pcl_conversions::toPCL(*msg, latest_cloud2);
  has_cloud = true;
}

// 回调：/Odometry
void odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
  latest_odom = *msg;
  has_odom = true;
  // 如果已经停止 ICP, 由 unified 逻辑处理发布
}

// —— 将下采样后的点云累加到 OctoMap —— 
void insertCloudToOctoMap(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, double max_range=50.0) {
  for (auto& pt : cloud->points) {
    if (std::abs(pt.x) > max_range || std::abs(pt.y) > max_range || std::abs(pt.z) > max_range)
      continue;
    octomap::point3d key(pt.x, pt.y, pt.z);
    octree->updateNode(key, true);
  }
  octree->updateInnerOccupancy();
}

pcl::PointCloud<pcl::PointXYZ>::Ptr extractOccupiedVoxels(octomap::OcTree* octree) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  for (octomap::OcTree::leaf_iterator it = octree->begin(), end = octree->end(); it != end; ++it) {
    if (octree->isNodeOccupied(*it)) {
      const octomap::point3d& pt = it.getCoordinate();
      cloud->push_back(pcl::PointXYZ(pt.x(), pt.y(), pt.z()));
    }
  }
  return cloud;
}

// 接收用户选定初始位姿
void selectedPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
  Eigen::Affine3f a =
    Eigen::Translation3f(msg->pose.position.x,
                        msg->pose.position.y,
                        msg->pose.position.z) *
    Eigen::Quaternionf(msg->pose.orientation.w,
                       msg->pose.orientation.x,
                       msg->pose.orientation.y,
                       msg->pose.orientation.z);
  init_guess = a.matrix();
  ROS_INFO("Received init_guess from /selected_pose");
}

void icpInitCallback(const std_msgs::Bool::ConstPtr& msg) {
  if (msg->data) {
    icp_trigger = true;
    ROS_INFO("ICP init triggered");
  } else if(!msg->data){
    icp_trigger = false;
    ROS_INFO("ICP init reset");
  }
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "global_location_icp");
  ros::NodeHandle nh("~");
  nav_msgs::Odometry odom_out;

  cloud_registered_map_pub = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_map", 1, true);

  // 1. 加载并下采样全局地图
  std::string map_path;
  nh.param<std::string>("map_path", map_path, "/home/cmh/scans.pcd");
  if (pcl::io::loadPCDFile(map_path, *global_map) < 0) {
    ROS_ERROR("Cannot load map: %s", map_path.c_str());
    return -1;
  }

  global_map_octo = *global_map;

  double leaf_size;
  nh.param("voxel_leaf_size", leaf_size, 0.5);
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  vg.setInputCloud(global_map);
  vg.setLeafSize(leaf_size, leaf_size, leaf_size);
  vg.filter(*global_map);
  ROS_INFO("Loaded map with %zu points", global_map->size());

  // 2. 发布全局地图一次
  map_pub = nh.advertise<sensor_msgs::PointCloud2>("global_map", 1, true);
  sensor_msgs::PointCloud2 map_msg;
  pcl::toROSMsg(*global_map, map_msg);
  map_msg.header.frame_id = "map";
  map_msg.header.stamp    = ros::Time::now();
  map_pub.publish(map_msg);

  // cloud_registered_map_pub.publish(map_msg);

  // 初始化 OctoMap 并插入全局地图
  double octo_res;
  nh.param("octomap_resolution", octo_res, 0.1);  // 从参数服务器读取，若无则默认 0.1

  octree.reset(new octomap::OcTree(octo_res));
  insertCloudToOctoMap(global_map);
  octomap_pub = nh.advertise<octomap_msgs::Octomap>("octomap_full", 1, true);
  octomap_msgs::Octomap octo_msg;
  octomap_msgs::fullMapToMsg(*octree, octo_msg);
  octo_msg.header.frame_id = "map";
  octo_msg.header.stamp = ros::Time::now();
  octomap_pub.publish(octo_msg);


  // 3. 初始化 ICP 参数
  icp.setMaxCorrespondenceDistance(20.0);
  icp.setMaximumIterations(50);
  icp.setTransformationEpsilon(1e-10);
  icp.setEuclideanFitnessEpsilon(1e-10);

  // 4. 订阅 & 发布
  ros::Subscriber sub_odom    = nh.subscribe("/Odometry",        10, odomCallback);
  ros::Subscriber sub_cloud   = nh.subscribe("/cloud_registered", 1, cloudCallback);
  ros::Subscriber sub_stop    = nh.subscribe("/stop_icp",         1, stopIcpCallback);
  ros::Subscriber sub_selected = nh.subscribe("/selected_pose", 1, selectedPoseCallback);
  ros::Subscriber sub_icp_init  = nh.subscribe("/icp_init", 1, icpInitCallback);

  odom_pub    = nh.advertise<nav_msgs::Odometry>("odom_reloc", 10);
  path_pub    = nh.advertise<nav_msgs::Path>("path_reloc", 10, true);
  aligned_pub = nh.advertise<sensor_msgs::PointCloud2>("aligned_cloud", 1);
  octomap_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("octomap_cloud", 1, true);
  cloud_registered_map_pub = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_map", 1, true);

  

  path_msg.header.frame_id = "map";

  ros::Rate rate(20);
  ros::Time last_print = ros::Time::now();
  while (ros::ok()) {
    ros::spinOnce();

    // —— 实时 ICP 阶段 ——
    if (has_odom && has_cloud && !stop_icp && icp_trigger) {
      // 下采样当前帧
      pcl::PointCloud<pcl::PointXYZ>::Ptr src(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::fromPCLPointCloud2(latest_cloud2, *src);
      pcl::VoxelGrid<pcl::PointXYZ> vg2;
      vg2.setInputCloud(src);
      vg2.setLeafSize(leaf_size, leaf_size, leaf_size);
      vg2.filter(*src);

      // 执行 ICP
      icp.setInputSource(src);
      icp.setInputTarget(global_map);
      pcl::PointCloud<pcl::PointXYZ> aligned;


      // 用用户提供的 init_guess
      icp.align(aligned, init_guess);
      ROS_INFO_STREAM_THROTTLE(1.0, "Current init_guess:\n" << init_guess);
      

      // icp.align(aligned, odomToEigen(latest_odom));

      double score = icp.getFitnessScore();
      if (icp.hasConverged()) {
        T_reloc_saved = icp.getFinalTransformation();
        ROS_INFO_STREAM_THROTTLE(1.0, "Current T_reloc_saved:\n" << T_reloc_saved);
        ROS_INFO_STREAM_THROTTLE(1.0, "ICP converged. score=" << score);
      } else {
        ROS_WARN_THROTTLE(1.0, "ICP not converged this frame. score = %.6f", score);
      }

      // 当 score < 0.1 时停止
      if (score < 0.1) {
        stop_icp = true;
        ROS_WARN_STREAM("ICP score < 0.1, stopping ICP. score=" << score);
        icp_init = 0;
        icp_trigger = false;
      }

      // 发布配准点云
      sensor_msgs::PointCloud2 aligned_msg;
      pcl::toROSMsg(aligned, aligned_msg);
      aligned_msg.header.frame_id = "map";
      aligned_msg.header.stamp    = ros::Time::now();
      aligned_pub.publish(aligned_msg);

      // 同步发布里程计到 map，并累积 path
      Eigen::Matrix4f T_map = T_reloc_saved * odomToEigen(latest_odom);
      nav_msgs::Odometry odom_out = latest_odom;
      eigenToOdom(odom_out, T_map);
      odom_out.header.frame_id = "map";
      odom_pub.publish(odom_out);

      geometry_msgs::PoseStamped ps;
      ps.header = odom_out.header;
      ps.pose   = odom_out.pose.pose;
      path_msg.header.stamp = ros::Time::now();
      path_msg.poses.push_back(ps);
      path_pub.publish(path_msg);

      // // 将 src 点云变换到 map 坐标系
      // pcl::PointCloud<pcl::PointXYZ>::Ptr src_map(new pcl::PointCloud<pcl::PointXYZ>);
      // pcl::transformPointCloud(*src, *src_map, T_map);
      // // 累加并发布 OctoMap
      // insertCloudToOctoMap(src_map);
      // octomap_msgs::Octomap octo_msg;
      // octomap_msgs::fullMapToMsg(*octree, octo_msg);
      // octo_msg.header.frame_id = "map";
      // octo_msg.header.stamp = ros::Time::now();
      // octomap_pub.publish(octo_msg);
      // // 发布 OctoMap 中占据体素中心作为点云
      // pcl::PointCloud<pcl::PointXYZ>::Ptr octo_cloud = extractOccupiedVoxels(octree.get());
      // sensor_msgs::PointCloud2 cloud_msg;
      // pcl::toROSMsg(*octo_cloud, cloud_msg);
      // cloud_msg.header.frame_id = "map";
      // cloud_msg.header.stamp = ros::Time::now();
      // octomap_cloud_pub.publish(cloud_msg);


    }

    // —— 停止后仅发布里程计 & path 阶段 ——
    else if (stop_icp && has_odom) {
      if(icp_init=1)
      {
        icp_init = 0;
        sensor_msgs::PointCloud2 map_msg;
        pcl::toROSMsg(*global_map, map_msg);
        map_msg.header.frame_id = "map";
        map_msg.header.stamp    = ros::Time::now();
        cloud_registered_map_pub.publish(map_msg);

      }
      // 持续把 FastLIO odom 转到 map 发布
      Eigen::Matrix4f T_map = T_reloc_saved * odomToEigen(latest_odom);
      odom_out = latest_odom;
      eigenToOdom(odom_out, T_map);
      odom_out.header.frame_id = "map";
      odom_pub.publish(odom_out);

      geometry_msgs::PoseStamped ps;
      ps.header = odom_out.header;
      ps.pose   = odom_out.pose.pose;
      path_msg.header.stamp = ros::Time::now();
      path_msg.poses.push_back(ps);
      path_pub.publish(path_msg);

      // 停止后也累加 OctoMap
      pcl::PointCloud<pcl::PointXYZ>::Ptr src(new pcl::PointCloud<pcl::PointXYZ>());
      pcl::fromPCLPointCloud2(latest_cloud2, *src);
      pcl::VoxelGrid<pcl::PointXYZ> vg2;
      // vg2.setInputCloud(src);
      // vg2.setLeafSize(leaf_size, leaf_size, leaf_size);
      // vg2.filter(*src);

      // 将 src 点云变换到 map 坐标系
      pcl::PointCloud<pcl::PointXYZ>::Ptr src_map(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::transformPointCloud(*src, *src_map, T_reloc_saved);
      // 发布配准点云
      sensor_msgs::PointCloud2 cloud_registered_map_msg;
      pcl::toROSMsg(*src_map, cloud_registered_map_msg);
      cloud_registered_map_msg.header.frame_id = "map";
      cloud_registered_map_msg.header.stamp    = ros::Time::now();
      cloud_registered_map_pub.publish(cloud_registered_map_msg);

      // insertCloudToOctoMap(src_map);
      // octomap_msgs::Octomap octo_msg;
      // octomap_msgs::fullMapToMsg(*octree, octo_msg);
      // octo_msg.header.frame_id = "map";
      // octo_msg.header.stamp = ros::Time::now();
      // octomap_pub.publish(octo_msg);
      // // 发布 OctoMap 中占据体素中心作为点云
      // pcl::PointCloud<pcl::PointXYZ>::Ptr octo_cloud = extractOccupiedVoxels(octree.get());
      // sensor_msgs::PointCloud2 cloud_msg;
      // pcl::toROSMsg(*octo_cloud, cloud_msg);
      // cloud_msg.header.frame_id = "map";
      // cloud_msg.header.stamp = ros::Time::now();
      // octomap_cloud_pub.publish(cloud_msg);

    }

    // —— 2 Hz 打印里程计信息 ——
    geometry_msgs::Quaternion q = odom_out.pose.pose.orientation;
    tf::Quaternion quat(q.x, q.y, q.z, q.w);
    double roll, pitch, yaw;
    tf::Matrix3x3(quat).getRPY(roll, pitch, yaw);

    ROS_INFO_THROTTLE(0.5, "[Odom in map] XYZ: [%.3f, %.3f, %.3f], RPY: [%.3f, %.3f, %.3f]",
                      odom_out.pose.pose.position.x,
                      odom_out.pose.pose.position.y,
                      odom_out.pose.pose.position.z,
                      roll, pitch, yaw);


    rate.sleep();
  }

  return 0;
}

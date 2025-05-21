#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>

#include <pcl/filters/voxel_grid.h>


// 全局变量
nav_msgs::Odometry     latest_odom;
pcl::PCLPointCloud2    latest_cloud2;
bool                   has_odom = false, has_cloud = false;
pcl::PointCloud<pcl::PointXYZ>::Ptr global_map(new pcl::PointCloud<pcl::PointXYZ>);
pcl::IterativeClosestPoint<pcl::PointXYZ,pcl::PointXYZ> icp;
ros::Publisher         odom_pub;

// 回调：接收里程计
void odomCallback(const nav_msgs::Odometry::ConstPtr& msg) 
{
  latest_odom = *msg;
  has_odom = true;  
}

// 回调：接收点云
void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) 
{
  pcl_conversions::toPCL(*msg, latest_cloud2);
  has_cloud = true;  
}

// 工具：Odometry -> Eigen 矩阵
Eigen::Matrix4f odomToEigen(const nav_msgs::Odometry& odom) 
{
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

// 工具：Eigen 矩阵 -> Odometry
void eigenToOdom(nav_msgs::Odometry& odom, const Eigen::Matrix4f& M) 
{
  Eigen::Affine3f a(M);
  Eigen::Vector3f t = a.translation();
  Eigen::Quaternionf q(a.rotation());
  odom.pose.pose.position.x = t.x();
  odom.pose.pose.position.y = t.y();
  odom.pose.pose.position.z = t.z();
  odom.pose.pose.orientation.x = q.x();
  odom.pose.pose.orientation.y = q.y();
  odom.pose.pose.orientation.z = q.z();
  odom.pose.pose.orientation.w = q.w();
  odom.header.stamp = ros::Time::now();
}

int main(int argc, char** argv) 
{
  ros::init(argc, argv, "relocalization_node");                      
  ros::NodeHandle nh("~");

  // 1. 只加载一次全局地图
  std::string map_path = "/home/cmh/scans.pcd";
  if (pcl::io::loadPCDFile(map_path, *global_map) < 0) {
    ROS_ERROR("无法加载全局地图: %s", map_path.c_str());
    return -1;
  }
  ROS_INFO("全局地图加载完成，点数: %zu", global_map->size());

  // 2. 订阅和发布
  ros::Subscriber odom_sub  = nh.subscribe("/Odometry",  1, odomCallback);      
  ros::Subscriber cloud_sub = nh.subscribe("/cloud_registered", 1, cloudCallback); 
  odom_pub = nh.advertise<nav_msgs::Odometry>("/Odometry_relocation", 1);
  ros::Publisher aligned_pub = nh.advertise<sensor_msgs::PointCloud2>("/aligned_cloud", 1);


  // 3. 配置 ICP 参数
  icp.setMaxCorrespondenceDistance(20.1);
  icp.setMaximumIterations(10);
  icp.setTransformationEpsilon(1e-3);
  icp.setEuclideanFitnessEpsilon(1);

  // 4. 主循环：ros::Rate + ros::spinOnce()
//   double hz = 20;
//   nh.param("relocalization_hz", hz, hz);
  ros::Rate rate(2);                                                     

  while (ros::ok()) 
  {
    ros::spinOnce();                                                     

    if (has_odom && has_cloud) {
      // 点云转换
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::fromPCLPointCloud2(latest_cloud2, *cloud_src);

      // ---------- 下采样开始 ----------
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_down_cur(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_down_gobal(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::VoxelGrid<pcl::PointXYZ> vg1;
      pcl::VoxelGrid<pcl::PointXYZ> vg2;
      vg1.setInputCloud(cloud_src);
      vg2.setInputCloud(global_map);
      // 这里设置叶子尺寸（以米为单位），可根据点云密度和精度需求调节
      double leaf_size = 0.5;  
      nh.getParam("voxel_leaf_size", leaf_size);  // 也可以通过 ROS 参数动态调整
      vg1.setLeafSize(leaf_size, leaf_size, leaf_size);
      vg1.filter(*cloud_down_cur);
      vg2.setLeafSize(leaf_size, leaf_size, leaf_size);
      vg2.filter(*cloud_down_gobal);
      // 替换原始点云以供 ICP 使用
      cloud_src.swap(cloud_down_cur);
      global_map.swap(cloud_down_gobal);
      // ---------- 下采样结束 ----------

      // 计算质心
      Eigen::Vector4f centroid;
      pcl::compute3DCentroid(*cloud_src, centroid);

      // 打印质心坐标
      std::cout << "点云中心坐标 (质心): "
                << "X: " << centroid[0] << ", "
                << "Y: " << centroid[1] << ", "
                << "Z: " << centroid[2] << std::endl;

      // 初始猜测
      Eigen::Matrix4f init_guess = odomToEigen(latest_odom);
      // init_guess = Eigen::Matrix4f::Identity();

      // 执行 ICP
      icp.setInputSource(cloud_src);
      icp.setInputTarget(global_map);
      pcl::PointCloud<pcl::PointXYZ> aligned;
      icp.align(*cloud_src);

      if (!icp.hasConverged()) {
        ROS_WARN("ICP no");                                         
      } else 
      {
        ROS_WARN("ICP ok");  
        // 发布修正里程计
        Eigen::Matrix4f T_reloc = icp.getFinalTransformation() ;

        nav_msgs::Odometry odom_out = latest_odom;

        Eigen::Matrix4f T_odom = odomToEigen(odom_out);

        Eigen::Matrix4f T_new =  T_reloc * T_odom;
        // Eigen::Matrix4f T_new = T_odom;

        // Eigen::Matrix4f T_new = T_odom;
        std::cout << "ICP 配准成功！分数：" << icp.getFitnessScore() << std::endl;
        std::cout << "T_reloc (4x4 Transformation Matrix):\n" << T_reloc << std::endl;
        
        eigenToOdom(odom_out, T_new);
        odom_pub.publish(odom_out);

        // 1.1 提取平移
        Eigen::Vector3f translation(
            odom_out.pose.pose.position.x,   
            odom_out.pose.pose.position.y,  
            odom_out.pose.pose.position.z   
        );

        // 1.2 提取旋转（四元数）
        Eigen::Quaternionf quaternion(
            odom_out.pose.pose.orientation.w,
            odom_out.pose.pose.orientation.x,
            odom_out.pose.pose.orientation.y,
            odom_out.pose.pose.orientation.z
        );

        // 1.3 构造 4×4 变换矩阵
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        transform.translation() = translation;
        transform.linear()      = quaternion.toRotationMatrix();  // 

        // 1.4 获取最终矩阵
        Eigen::Matrix4f mat_cam_body = transform.matrix();

        // 计算逆变换矩阵
        Eigen::Matrix4f T_inverse = mat_cam_body;

        // 假设 aligned 已含有 ICP 配准到 body 系的点
        pcl::PointCloud<pcl::PointXYZ> cloud_cam; 
        T_new = T_new.inverse();
        // 使用 PCL 原生接口应用矩阵变换
        pcl::transformPointCloud(aligned, cloud_cam, T_new);
        

        // 发布配准后的点云
        sensor_msgs::PointCloud2 aligned_ros;
        pcl::toROSMsg(*cloud_src, aligned_ros);
        aligned_ros.header.stamp = ros::Time::now();
        aligned_ros.header.frame_id = "camera_init";  // 或你自己的全局地图坐标系名称
        aligned_pub.publish(aligned_ros);


        // 广播 TF
        static tf::TransformBroadcaster br;
        tf::Transform tf;
        tf.setOrigin(tf::Vector3(
            odom_out.pose.pose.position.x,
            odom_out.pose.pose.position.y,
            odom_out.pose.pose.position.z));
        tf::Quaternion q(
            odom_out.pose.pose.orientation.x,
            odom_out.pose.pose.orientation.y,
            odom_out.pose.pose.orientation.z,
            odom_out.pose.pose.orientation.w);
        tf.setRotation(q);
        br.sendTransform(tf::StampedTransform(
            tf, odom_out.header.stamp,
            odom_out.header.frame_id,
            odom_out.child_frame_id));
      }
    }

    rate.sleep();                                                          
  }

  return 0;
}

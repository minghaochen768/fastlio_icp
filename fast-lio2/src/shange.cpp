#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>      // pcl::fromROSMsg
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <octomap/OcTree.h>
#include <octomap/Pointcloud.h>
#include <memory>                                  // std::make_shared
#include <unordered_set>                           // std::unordered_set
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>


void publishOccupiedVoxels(const std::shared_ptr<octomap::OcTree>& octree,
    ros::Publisher& pub,
    const std::string& frame_id)
{
    pcl::PointCloud<pcl::PointXYZ> pcl_cloud;

    for (auto it = octree->begin_leafs(), end = octree->end_leafs(); it != end; ++it)
    {
        if (octree->isNodeOccupied(*it)) {
            pcl::PointXYZ pt;
            pt.x = it.getX();
            pt.y = it.getY();
            pt.z = it.getZ();
            pcl_cloud.push_back(pt);
        }
    }

    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(pcl_cloud, ros_cloud);
    ros_cloud.header.frame_id = frame_id;
    ros_cloud.header.stamp = ros::Time::now();
    pub.publish(ros_cloud);
}




// 点云回调：更新占用栅格
void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
                        const std::shared_ptr<octomap::OcTree>& octree,
                        double origin_x, double origin_y, double origin_z,
                        double max_range)
{
    std::cout<<"shange huidiao"<<std::endl;
    // ROS → PCL
    pcl::PointCloud<pcl::PointXYZ> pcl_pc;
    pcl::fromROSMsg(*cloud_msg, pcl_pc);

    // PCL → octomap::Pointcloud
    octomap::Pointcloud oct_pc;
    for (const auto& pt : pcl_pc) {
        oct_pc.push_back(pt.x, pt.y, pt.z);
    }

    // 传感器原点（世界坐标）
    octomap::point3d sensor_origin(origin_x, origin_y, origin_z);

    // 插入占用和空闲栅格
    octree->insertPointCloud(oct_pc, sensor_origin,
                             max_range,
                             /* insertFreeSpace=*/true,
                             /* lazyEvaluation=*/false);
}

// float prob_threshold = 0.7;
// float logodds_threshold = std::log(prob_threshold / (1.0f - prob_threshold));  // ≈ 0.847


void removeLowConfidenceNodes(const std::shared_ptr<octomap::OcTree>& octree, float prob_threshold)
{
    float logodds_threshold = std::log(prob_threshold / (1.0f - prob_threshold));
    std::vector<octomap::OcTreeKey> keys_to_delete;

    for (auto it = octree->begin_leafs(), end = octree->end_leafs(); it != end; ++it)
    {
        if (it->getLogOdds() < logodds_threshold) {
            keys_to_delete.push_back(it.getKey());
        }
    }

    for (const auto& key : keys_to_delete) {
        octree->deleteNode(key);
    }
}



// 时间衰减：对数几率衰减
void decayMap(const std::shared_ptr<octomap::OcTree>& octree,
              double decay_factor)
{
    // 遍历所有节点
    for (auto it = octree->begin_tree(),
              end = octree->end_tree();
         it != end; ++it)
    {
        // 解引用到节点后调用
        double logodds = (*it).getLogOdds() * decay_factor;  // :contentReference[oaicite:2]{index=2}
        (*it).setLogOdds(static_cast<float>(logodds));       // :contentReference[oaicite:3]{index=3}
    }
}

// 差分剔除：移除消失的占用节点
void diffRemove(const std::shared_ptr<octomap::OcTree>& octree,
                std::unordered_set<octomap::OcTreeKey,
                                   octomap::OcTreeKey::KeyHash>& previous_keys)
{
    std::unordered_set<octomap::OcTreeKey,
                       octomap::OcTreeKey::KeyHash> current_keys;

    // 收集当前所有占用叶节点的 Key
    for (auto it = octree->begin_leafs(),
              end = octree->end_leafs();
         it != end; ++it)
    {
        if (octree->isNodeOccupied(*it)) {
            current_keys.insert(it.getKey());  // KeyHash 用于哈希 :contentReference[oaicite:4]{index=4}
        }
    }

    // 删除上次存在、当前消失的节点
    for (const auto& key : previous_keys) {
        if (!current_keys.count(key)) {
            octree->deleteNode(key);
        }
    }
    previous_keys.swap(current_keys);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "fastlio2_octomap");
    ros::NodeHandle nh("~");

    // 参数配置
    double resolution, origin_x, origin_y, origin_z,
           max_range, decay_factor;
    std::string cloud_topic;
    nh.param("resolution",    resolution,   0.5);
    nh.param("origin_x",      origin_x,     0.0);
    nh.param("origin_y",      origin_y,     0.0);
    nh.param("origin_z",      origin_z,     0.0);
    nh.param("max_range",     max_range,   50.0);
    nh.param("decay_factor",  decay_factor, 0.1);
    nh.param("cloud_topic",   cloud_topic,  std::string("/cloud_registered"));
    ros::Publisher map_pub = nh.advertise<octomap_msgs::Octomap>("octomap", 1);
    ros::Publisher center_pub = nh.advertise<sensor_msgs::PointCloud2>("octomap_point_cloud_centers", 1);



    // 创建 OctoMap 八叉树
    auto octree = std::make_shared<octomap::OcTree>(resolution);

    // 用于动态剔除的键集合
    std::unordered_set<octomap::OcTreeKey,
                       octomap::OcTreeKey::KeyHash> previous_keys;

    // 订阅点云话题，按值捕获 octree 等
    ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>(
        cloud_topic, 1,
        [octree, origin_x, origin_y, origin_z, max_range]
        (const sensor_msgs::PointCloud2ConstPtr& msg) {
            pointCloudCallback(msg, octree,
                               origin_x, origin_y, origin_z,
                               max_range);
        }
    );

    ros::Rate rate(10.0);
    while (ros::ok()) {
        removeLowConfidenceNodes(octree, 0.7);

        std::cout<<"shange"<<std::endl;
        decayMap(octree, decay_factor);
        diffRemove(octree, previous_keys);

        // 裁剪和持久化
        octree->prune();
        octree->writeBinary("/tmp/map.bt");

        ros::spinOnce();
        rate.sleep();

        octomap_msgs::Octomap map_msg;
        map_msg.header.frame_id = "camera_init";
        map_msg.header.stamp = ros::Time::now();
        if (octomap_msgs::binaryMapToMsg(*octree, map_msg)) {
            map_pub.publish(map_msg);
        }
        publishOccupiedVoxels(octree, center_pub, "camera_init");


    }

    return 0;
}

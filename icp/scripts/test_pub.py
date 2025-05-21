#!/usr/bin/env python3
import rospy
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import open3d as o3d
import tf.transformations
import random
import std_msgs.msg

# 随机生成变换矩阵
def generate_random_transform(max_trans=4.0, max_angle_deg=30.0):
    max_transx=6
    max_transy=6
    max_transz=6

    max_angle_degr=90
    max_angle_degp=90
    max_angle_degy=90

    dx = random.uniform(-max_transx, max_transx)
    dy = random.uniform(-max_transy, max_transy)
    dz = random.uniform(-max_transz, max_transz)

    roll  = np.radians(random.uniform(-max_angle_degr, max_angle_degr))
    pitch = np.radians(random.uniform(-max_angle_degp, max_angle_degp))
    yaw   = np.radians(random.uniform(-max_angle_degy, max_angle_degy))

    rot = tf.transformations.euler_matrix(roll, pitch, yaw)
    rot[0, 3] = dx
    rot[1, 3] = dy
    rot[2, 3] = dz

    return rot[:3, :]  # 返回 3x4

# ROS PointCloud2 转 Open3D PointCloud
def ros_to_open3d(msg):
    points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.array(points))
    return cloud

# Open3D PointCloud 转 ROS PointCloud2
def open3d_to_ros(cloud, frame_id):
    points = np.asarray(cloud.points)
    msg = pc2.create_cloud_xyz32(std_msgs.msg.Header(stamp=rospy.Time.now(), frame_id=frame_id), points.tolist())
    return msg

# 回调函数
def cloud_callback(msg):
    cloud_o3d = ros_to_open3d(msg)
    if len(cloud_o3d.points) == 0:
        rospy.logwarn("点云为空")
        return

    transform = generate_random_transform()
    cloud_o3d.transform(np.vstack((transform, [0, 0, 0, 1])))  # 4x4 变换

    msg_out = open3d_to_ros(cloud_o3d, msg.header.frame_id)
    cloud_pub.publish(msg_out)
    rospy.loginfo_throttle(1.0, "发布扰动后的点云 /cloud_registered_c")

if __name__ == "__main__":
    rospy.init_node("cloud_random_transformer")

    rospy.Subscriber("/cloud_registered", PointCloud2, cloud_callback, queue_size=1)
    cloud_pub = rospy.Publisher("/cloud_registered_c", PointCloud2, queue_size=1)

    rospy.loginfo("节点已启动，等待 /cloud_registered 点云数据...")
    rospy.spin()

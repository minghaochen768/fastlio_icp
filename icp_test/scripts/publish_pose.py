#!/usr/bin/env python3
# coding=utf8

import argparse
import rospy
import tf.transformations
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion
import math

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Publish Odometry msg with position and orientation (yaw, pitch, roll in degrees)")
    parser.add_argument('x', type=float)
    parser.add_argument('y', type=float)
    parser.add_argument('z', type=float)
    parser.add_argument('yaw', type=float)
    parser.add_argument('pitch', type=float)
    parser.add_argument('roll', type=float)
    args = parser.parse_args()

    # 欧拉角转四元数（角度转弧度）
    roll = math.radians(args.roll)
    pitch = math.radians(args.pitch)
    yaw = math.radians(args.yaw)
    quat = tf.transformations.quaternion_from_euler(roll, pitch, yaw)

    rospy.init_node('publish_odom_point')
    pub = rospy.Publisher('/set_point', Odometry, queue_size=1)

    odom_msg = Odometry()
    odom_msg.header.frame_id = 'camera_init'
    odom_msg.child_frame_id = 'base_link'  # 可根据需要调整
    odom_msg.pose.pose.position.x = args.x
    odom_msg.pose.pose.position.y = args.y
    odom_msg.pose.pose.position.z = args.z
    odom_msg.pose.pose.orientation = Quaternion(*quat)

    rospy.loginfo("Publishing Odometry at 10 Hz:\n  Position: (%.2f, %.2f, %.2f)\n  Orientation (Yaw=%.2f°, Pitch=%.2f°, Roll=%.2f°)",
                  args.x, args.y, args.z, args.yaw, args.pitch, args.roll)
                  

    # rate = rospy.Rate(10)  # 10 Hz
    # while not rospy.is_shutdown():
    rospy.sleep(1.0)     
    odom_msg.header.stamp = rospy.Time.now()
    pub.publish(odom_msg)
        # rate.sleep()

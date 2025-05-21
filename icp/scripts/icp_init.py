#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import threading, math
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from visualization_msgs.msg      import InteractiveMarker, InteractiveMarkerControl
from geometry_msgs.msg           import PoseStamped, TransformStamped
from sensor_msgs.msg             import PointCloud2
import tf2_ros
import tf.transformations as tft
from sensor_msgs import point_cloud2
from pynput import keyboard

# —— 全局状态 ——
current_pose = PoseStamped()
current_pose.header.frame_id       = "map"
current_pose.pose.orientation.w    = 1.0

# 缓存最新点云
latest_cloud2 = None

# 平移与旋转步长
STEP_POS = 0.05
STEP_ANG = math.radians(5)

# ROS 对象占位
server         = None
pub_pose       = None
cloud_pub      = None
tf_broadcaster = None
int_marker     = None

def broadcast_marker_tf():
    """发布 map->pose_marker TF"""
    t = TransformStamped()
    t.header.stamp    = rospy.Time.now()
    t.header.frame_id = "map"
    t.child_frame_id  = "pose_marker"
    p = current_pose.pose.position
    q = current_pose.pose.orientation
    t.transform.translation.x = p.x
    t.transform.translation.y = p.y
    t.transform.translation.z = p.z
    t.transform.rotation      = q
    tf_broadcaster.sendTransform(t)

def update_marker_and_cloud():
    """更新 InteractiveMarker 并变换、发布点云，同时打印当前位姿"""
    global latest_cloud2

    # —— 更新 InteractiveMarker —— 
    int_marker.pose = current_pose.pose
    server.setPose(int_marker.name, int_marker.pose)
    server.applyChanges()

    # 发布位姿
    pub_pose.publish(current_pose)
    broadcast_marker_tf()

    # —— 打印当前位姿 —— 
    q = current_pose.pose.orientation
    roll, pitch, yaw = tft.euler_from_quaternion((q.x, q.y, q.z, q.w))
    rospy.loginfo("[Marker Pose] x=%.3f, y=%.3f, z=%.3f | RPY=(%.1f°,%.1f°,%.1f°) | quat=(%.3f,%.3f,%.3f,%.3f)",
                  current_pose.pose.position.x,
                  current_pose.pose.position.y,
                  current_pose.pose.position.z,
                  math.degrees(roll), math.degrees(pitch), math.degrees(yaw),
                  q.x, q.y, q.z, q.w)

    # 若无点云，跳过
    if latest_cloud2 is None:
        return

    # 1) 读取所有点
    pts = list(point_cloud2.read_points(
        latest_cloud2, field_names=("x","y","z"), skip_nans=True
    ))

    # 2) 构造 4×4 变换矩阵
    p = current_pose.pose.position
    mat = tft.quaternion_matrix((q.x, q.y, q.z, q.w))
    mat[0][3], mat[1][3], mat[2][3] = p.x, p.y, p.z

    # 3) 对每个点应用变换
    out_pts = []
    for x, y, z in pts:
        vx, vy, vz, _ = mat.dot((x, y, z, 1.0))
        out_pts.append((vx, vy, vz))

    # 4) 打包并发布变换后的点云
    header = latest_cloud2.header
    header.frame_id = "map"  # 输出使用 map
    cloud_out = point_cloud2.create_cloud_xyz32(header, out_pts)
    cloud_pub.publish(cloud_out)

def on_press(key):
    """键盘：平移 q/w/e/r/t/y；旋转 a/s/d/f/g/h"""
    try:
        c = key.char.lower()
        if   c=='q': current_pose.pose.position.x += STEP_POS
        elif c=='w': current_pose.pose.position.x -= STEP_POS
        elif c=='e': current_pose.pose.position.y += STEP_POS
        elif c=='r': current_pose.pose.position.y -= STEP_POS
        elif c=='t': current_pose.pose.position.z += STEP_POS
        elif c=='y': current_pose.pose.position.z -= STEP_POS
        else:
            # 旋转
            q = current_pose.pose.orientation
            roll, pitch, yaw = tft.euler_from_quaternion((q.x,q.y,q.z,q.w))
            if   c=='a': roll  += STEP_ANG
            elif c=='s': roll  -= STEP_ANG
            elif c=='d': pitch += STEP_ANG
            elif c=='f': pitch -= STEP_ANG
            elif c=='g': yaw   += STEP_ANG
            elif c=='h': yaw   -= STEP_ANG
            else: return
            # 归一化到 [-π,π]
            roll  = (roll  +math.pi)%(2*math.pi)-math.pi
            pitch = (pitch +math.pi)%(2*math.pi)-math.pi
            yaw   = (yaw   +math.pi)%(2*math.pi)-math.pi
            qn = tft.quaternion_from_euler(roll,pitch,yaw)
            current_pose.pose.orientation.x = qn[0]
            current_pose.pose.orientation.y = qn[1]
            current_pose.pose.orientation.z = qn[2]
            current_pose.pose.orientation.w = qn[3]

        update_marker_and_cloud()
    except AttributeError:
        pass  # 忽略特殊键

def keyboard_listener():
    threading.Thread(
        target=lambda: keyboard.Listener(on_press=on_press).start(),
        daemon=True
    ).start()

def process_feedback(feedback):
    """鼠标拖拽/旋转回调"""
    current_pose.pose = feedback.pose
    update_marker_and_cloud()

def cloud_callback(msg: PointCloud2):
    """订阅 /cloud_registered_map，缓存最新点云并立即更新显示"""
    global latest_cloud2
    latest_cloud2 = msg
    update_marker_and_cloud()

if __name__ == "__main__":
    rospy.init_node("pose_marker_map", anonymous=True)

    # 1) Interactive Marker Server
    server     = InteractiveMarkerServer("pose_marker_server")
    int_marker = InteractiveMarker()
    int_marker.header.frame_id = "map"
    int_marker.name            = "pose_marker"
    int_marker.description     = "拖拽或按键移动"
    int_marker.scale           = 0.4
    ctrl = InteractiveMarkerControl()
    ctrl.interaction_mode = InteractiveMarkerControl.MOVE_ROTATE_3D
    ctrl.always_visible   = True
    int_marker.controls.append(ctrl)
    server.insert(int_marker, process_feedback)
    server.applyChanges()

    # 2) Publishers & TF
    pub_pose       = rospy.Publisher("selected_pose", PoseStamped, queue_size=1)
    cloud_pub      = rospy.Publisher("cloud_registered_map_moved", PointCloud2, queue_size=1)
    tf_broadcaster = tf2_ros.TransformBroadcaster()

    # 3) 仅订阅这一条点云话题
    rospy.Subscriber(
        "/cloud_registered",
        PointCloud2,
        cloud_callback,
        queue_size=1,
        tcp_nodelay=True
    )

    # 4) 启动键盘监听
    keyboard_listener()

    rospy.loginfo("启动完成：仅订阅 /cloud_registered_map，标记下实时变换点云。")
    rospy.spin()

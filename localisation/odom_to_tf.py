#!/usr/bin/env python3
"""Publish the odometry TF edge `base -> base_link` from the base's odometry topic.

slam_toolbox needs odometry as a TF edge. On this robot that edge is
`base -> base_link` -- there is no `odom` frame. The robot's own stack normally
publishes it; start_localisation.sh starts this bridge ONLY when it does not,
so base_link never has two sources.
"""
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from tf2_ros import TransformBroadcaster


class OdomToTf(Node):
    def __init__(self):
        super().__init__("ebim_odom_to_tf")
        self.br = TransformBroadcaster(self)
        self.create_subscription(Odometry, "/swerve_drive_controller/odom",
                                 self.cb, qos_profile_sensor_data)

    def cb(self, m: Odometry) -> None:
        t = TransformStamped()
        t.header.stamp = m.header.stamp
        if t.header.stamp.sec == 0 and t.header.stamp.nanosec == 0:
            t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "base"
        t.child_frame_id = "base_link"
        t.transform.translation.x = m.pose.pose.position.x
        t.transform.translation.y = m.pose.pose.position.y
        t.transform.translation.z = 0.0
        t.transform.rotation = m.pose.pose.orientation
        self.br.sendTransform(t)


def main() -> None:
    rclpy.init()
    rclpy.spin(OdomToTf())


if __name__ == "__main__":
    main()

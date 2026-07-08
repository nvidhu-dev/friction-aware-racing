#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PointStamped, Point
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker


class MPCVisualizer(Node):
    def __init__(self):
        super().__init__('mpc_visualizer')

        self.ref_point = None
        self.ref_path = None
        self.pred_path = None

        self.create_subscription(
            PointStamped, '/debug/mpc_ref_point', self.cb_ref_point, 10
        )
        self.create_subscription(
            Path, '/debug/mpc_ref_path', self.cb_ref_path, 10
        )
        self.create_subscription(
            Path, '/debug/mpc_pred_path', self.cb_pred_path, 10
        )

        self.marker_pub = self.create_publisher(Marker, '/debug/mpc_markers', 10)

        self.create_timer(0.05, self.publish_all)

    def cb_ref_point(self, msg):
        self.ref_point = msg

    def cb_ref_path(self, msg):
        self.ref_path = msg

    def cb_pred_path(self, msg):
        self.pred_path = msg

    def publish_all(self):
        if self.ref_point is not None:
            self.publish_ref_point()

        if self.ref_path is not None:
            self.publish_path_marker(
                self.ref_path,
                ns='mpc_ref_path',
                marker_id=1,
                r=0.0, g=1.0, b=0.0,
                width=0.05
            )

        if self.pred_path is not None:
            self.publish_path_marker(
                self.pred_path,
                ns='mpc_pred_path',
                marker_id=2,
                r=1.0, g=0.0, b=0.0,
                width=0.08
            )

    def publish_ref_point(self):
        marker = Marker()
        marker.header = self.ref_point.header
        marker.ns = 'mpc_ref_point'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position = self.ref_point.point
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2

        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        self.marker_pub.publish(marker)

    def publish_path_marker(self, path_msg, ns, marker_id, r, g, b, width):
        marker = Marker()
        marker.header = path_msg.header
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0

        marker.scale.x = width

        marker.color.a = 1.0
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b

        for pose in path_msg.poses:
            p = Point()
            p.x = pose.pose.position.x
            p.y = pose.pose.position.y
            p.z = 0.0
            marker.points.append(p)

        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = MPCVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
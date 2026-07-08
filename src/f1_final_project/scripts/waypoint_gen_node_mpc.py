#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import csv
from scipy.interpolate import splprep, splev

class WaypointRecorder(Node):
    def __init__(self):
        super().__init__("waypoint_recorder")
        self.declare_parameter("spacing", 0.75)
        self.declare_parameter("output_file", "waypoints.csv")
        self.declare_parameter("closed_loop", False)

        self.spacing = self.get_parameter("spacing").value
        self.output_file = self.get_parameter("output_file").value
        self.closed_loop = self.get_parameter("closed_loop").value

        self.raw_waypoints = []
        self.interp_waypoints = []

        self.sub = self.create_subscription(
            PoseWithCovarianceStamped, "/initialpose", self.pose_callback, 10
        )
        self.marker_pub = self.create_publisher(MarkerArray, "/waypoint_markers", 10)
        self.get_logger().info("Waypoint recorder running. Click poses in Foxglove.")

    def pose_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.raw_waypoints.append((x, y))
        self.get_logger().info(f"Added waypoint {len(self.raw_waypoints)}: {x:.2f}, {y:.2f}")

        if len(self.raw_waypoints) > 1:
            self.interp_waypoints = self.interpolate_waypoints(self.raw_waypoints)
            self.interp_waypoints = self.smooth_waypoints(self.interp_waypoints)
        else:
            self.interp_waypoints = self.raw_waypoints.copy()

        self.publish_markers()

    def interpolate_waypoints(self, pts):
        pts = np.array(pts)
        new_pts = []
        segments = [(pts[i], pts[i+1]) for i in range(len(pts)-1)]
        if self.closed_loop and len(pts) > 1:
            segments.append((pts[-1], pts[0]))
        for p0, p1 in segments:
            vec, dist = p1 - p0, np.linalg.norm(p1 - p0)
            if dist == 0: continue
            n = max(int(dist/self.spacing), 1)
            for i in range(n):
                new_pts.append(p0 + vec*i/n)
        new_pts.append(segments[-1][1])
        return new_pts

    def smooth_waypoints(self, pts):
        pts = np.array(pts).T
        n = pts.shape[1]
        if n < 2: return pts.T
        k = min(3, n - 1)   # spline degree: needs m > k
        tck, _ = splprep(pts, s=0.1, k=k)
        u = np.linspace(0, 1, max(2, n * 2))
        return np.array(splev(u, tck)).T

    def compute_speeds(
        self,
        pts,
        min_speed=0.5,
        max_speed=2.3,
        a_lat_max=0.8,     # lower = slower in turns
        a_long_max=1.0     # braking/accel limit (m/s^2)
    ):
        pts = np.array(pts)
        n = len(pts)
        if n < 3:
            return [max_speed] * n

        # --- curvature-limited speed ---
        v = np.zeros(n)
        for i in range(n):
            p0 = pts[i-1]
            p1 = pts[i]
            p2 = pts[(i+1) % n] if self.closed_loop else pts[min(i+1, n-1)]

            a = p1 - p0
            b = p2 - p1
            la = np.linalg.norm(a)
            lb = np.linalg.norm(b)

            if la < 1e-6 or lb < 1e-6:
                v[i] = max_speed
                continue

            angle = np.arccos(np.clip(np.dot(a, b) / (la * lb), -1, 1))
            curvature = angle / (la + lb)

            if curvature < 1e-6:
                v[i] = max_speed
            else:
                v[i] = np.sqrt(a_lat_max / curvature)

        v = np.clip(v, min_speed, max_speed)

        # --- braking BEFORE turns (backward pass) ---
        for i in reversed(range(n - 1)):
            ds = np.linalg.norm(pts[i+1] - pts[i])
            v_allow = np.sqrt(v[i+1]**2 + 2 * a_long_max * ds)
            v[i] = min(v[i], v_allow)

        # --- smooth acceleration AFTER turns (forward pass) ---
        for i in range(1, n):
            ds = np.linalg.norm(pts[i] - pts[i-1])
            v_allow = np.sqrt(v[i-1]**2 + 2 * a_long_max * ds)
            v[i] = min(v[i], v_allow)

        return v.tolist()

    def compute_yaws(self, pts):
        pts = np.array(pts)
        n = len(pts)
        yaws = np.zeros(n)
        for i in range(n - 1):
            dx = pts[i + 1, 0] - pts[i, 0]
            dy = pts[i + 1, 1] - pts[i, 1]
            yaws[i] = np.arctan2(dy, dx)
        # last point: same heading as second-to-last
        yaws[-1] = yaws[-2] if n > 1 else 0.0
        return yaws.tolist()

    def publish_markers(self):
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        for i, (x,y) in enumerate(self.raw_waypoints):
            m = Marker()
            m.header.frame_id = "map"; m.header.stamp = stamp
            m.ns, m.id, m.type, m.action = "raw_waypoints", i, Marker.SPHERE, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), 0.0
            m.scale.x = m.scale.y = m.scale.z = 0.25
            m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 0.0, 1.0, 1.0
            marker_array.markers.append(m)

        if len(self.interp_waypoints) == 0:
            m = Marker()
            m.header.frame_id = "map"; m.header.stamp = stamp
            m.ns, m.id, m.action, m.scale.x, m.scale.y, m.scale.z, m.color.a = "interp_waypoints", 0, Marker.ADD, 0.0,0.0,0.0,0.0
            marker_array.markers.append(m)
        else:
            speeds = self.compute_speeds(self.interp_waypoints)
            min_s, max_s = min(speeds), max(speeds)
            for i, ((x,y), s) in enumerate(zip(self.interp_waypoints, speeds)):
                m = Marker()
                m.header.frame_id = "map"; m.header.stamp = stamp
                m.ns, m.id, m.type, m.action = "interp_waypoints", 10000+i, Marker.SPHERE, Marker.ADD
                m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), 0.0
                m.scale.x = m.scale.y = m.scale.z = 0.12
                t = 0.0 if max_s-min_s==0 else (s-min_s)/(max_s-min_s)
                m.color.r, m.color.g, m.color.b, m.color.a = float(1.0-t), float(t), 0.0, 1.0
                marker_array.markers.append(m)

            if len(self.interp_waypoints) > 1:
                line = Marker()
                line.header.frame_id = "map"; line.header.stamp = stamp
                line.ns, line.id, line.type, line.action = "interp_path", 20000, Marker.LINE_STRIP, Marker.ADD
                line.scale.x = 0.05
                line.color.r, line.color.g, line.color.b, line.color.a = 1.0,0.0,0.0,1.0
                for x,y in self.interp_waypoints:
                    p = Point(); p.x, p.y, p.z = float(x), float(y), 0.0
                    line.points.append(p)
                marker_array.markers.append(line)

        self.marker_pub.publish(marker_array)

    def save_waypoints(self):
        if len(self.interp_waypoints) == 0:
            return

        with open(self.output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "speed", "yaw"])

            speeds = self.compute_speeds(self.interp_waypoints)
            yaws = np.unwrap(self.compute_yaws(self.interp_waypoints))
            for (x, y), s, yaw in zip(self.interp_waypoints, speeds, yaws):
                writer.writerow([
                    f"{x:.2f}",
                    f"{y:.2f}",
                    f"{s:.2f}",
                    f"{yaw:.4f}",
                ])

        self.get_logger().info(
            f"Saved {len(self.interp_waypoints)} waypoints to {self.output_file}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = WaypointRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.save_waypoints()
    node.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()

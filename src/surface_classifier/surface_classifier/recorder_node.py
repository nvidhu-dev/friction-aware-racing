#!/usr/bin/env python3
"""ROS 2 node: crop the same fixed ROI as the classifier and save labeled patches to disk.

Active label is set via the `current_label` parameter, changeable at runtime:
    ros2 param set /recorder_node current_label grass
so you don't need to relaunch when walking the car to a new surface.
"""

import os
import time

import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class RecorderNode(Node):
    def __init__(self):
        super().__init__('recorder_node')

        self.declare_parameter('image_topic', '/camera/rgb/raw')
        self.declare_parameter('output_dir', '~/surface_data')
        self.declare_parameter('current_label', 'unlabeled')
        self.declare_parameter('save_hz', 2.0)
        self.declare_parameter('roi_x', 220)
        self.declare_parameter('roi_y', 320)
        self.declare_parameter('roi_w', 200)
        self.declare_parameter('roi_h', 160)

        gp = self.get_parameter
        self.output_dir = os.path.expanduser(gp('output_dir').value)
        self.save_period = 1.0 / max(0.1, float(gp('save_hz').value))
        self.roi = (int(gp('roi_x').value), int(gp('roi_y').value),
                    int(gp('roi_w').value), int(gp('roi_h').value))

        os.makedirs(self.output_dir, exist_ok=True)

        self.bridge = CvBridge()
        cam_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub = self.create_subscription(
            Image, gp('image_topic').value, self._on_image, cam_qos)

        self._latest_frame = None
        self._last_save = 0.0
        self.timer = self.create_timer(self.save_period, self._tick)

        self.get_logger().info(
            f"recorder_node ready | output={self.output_dir} | roi(xywh)={self.roi} "
            f"| save_hz={1.0 / self.save_period:.2f}")
        self.get_logger().info(
            "Set label at runtime: ros2 param set /recorder_node current_label <label>")

    def _on_image(self, msg: Image):
        self._latest_frame = msg

    def _crop_roi(self, frame_bgr):
        x, y, w, h = self.roi
        H, W = frame_bgr.shape[:2]
        x2, y2 = min(x + w, W), min(y + h, H)
        x, y = max(0, x), max(0, y)
        if x2 <= x or y2 <= y:
            return None
        return frame_bgr[y:y2, x:x2].copy()

    def _tick(self):
        msg = self._latest_frame
        if msg is None:
            return
        label = str(self.get_parameter('current_label').value).strip()
        if not label:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"cv_bridge failed: {e}")
            return
        patch = self._crop_roi(frame)
        if patch is None:
            return

        label_dir = os.path.join(self.output_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        ts = time.time()
        out_path = os.path.join(label_dir, f"{ts:.3f}.png")
        cv2.imwrite(out_path, patch)


def main(args=None):
    rclpy.init(args=args)
    node = RecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

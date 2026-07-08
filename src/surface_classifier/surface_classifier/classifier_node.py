#!/usr/bin/env python3
"""ROS 2 node: crop a fixed ground ROI, run TensorRT classifier, publish material + friction."""

from collections import deque, Counter
import os

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32

from surface_classifier.trt_classifier import TRTClassifier
from surface_classifier.friction_map import FrictionMap


class ClassifierNode(Node):
    def __init__(self):
        super().__init__('classifier_node')

        self.declare_parameter('image_topic', '/camera/rgb/raw')
        self.declare_parameter('material_topic', '/surface/material')
        self.declare_parameter('friction_topic', '/surface/friction')
        self.declare_parameter('confidence_topic', '/surface/confidence')
        self.declare_parameter('debug_image_topic', '/surface/debug_image')
        self.declare_parameter('engine_path', '')
        self.declare_parameter('friction_map_path', '')
        self.declare_parameter('input_size', 224)
        self.declare_parameter('inference_hz', 10.0)
        self.declare_parameter('class_names', ['carpet', 'grass', 'ice', 'road', 'tile'])
        self.declare_parameter('roi_x', 220)
        self.declare_parameter('roi_y', 320)
        self.declare_parameter('roi_w', 200)
        self.declare_parameter('roi_h', 160)
        self.declare_parameter('smoothing_window', 5)
        self.declare_parameter('publish_debug', True)

        gp = self.get_parameter
        self.engine_path = os.path.expanduser(gp('engine_path').value)
        self.friction_map_path = os.path.expanduser(gp('friction_map_path').value)
        self.input_size = int(gp('input_size').value)
        self.inference_hz = float(gp('inference_hz').value)
        self.class_names = list(gp('class_names').value)
        self.roi = (int(gp('roi_x').value), int(gp('roi_y').value),
                    int(gp('roi_w').value), int(gp('roi_h').value))
        self.window = max(1, int(gp('smoothing_window').value))
        self.publish_debug = bool(gp('publish_debug').value)

        if not self.engine_path or not os.path.isfile(self.engine_path):
            self.get_logger().error(f"engine_path not found: {self.engine_path}")
            raise FileNotFoundError(self.engine_path)
        if not self.friction_map_path or not os.path.isfile(self.friction_map_path):
            self.get_logger().error(f"friction_map_path not found: {self.friction_map_path}")
            raise FileNotFoundError(self.friction_map_path)

        self.bridge = CvBridge()
        self.classifier = TRTClassifier(self.engine_path, input_size=self.input_size)
        self.friction = FrictionMap(self.friction_map_path)
        self.history = deque(maxlen=self.window)

        cam_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub = self.create_subscription(
            Image, gp('image_topic').value, self._on_image, cam_qos)

        self.material_pub = self.create_publisher(String, gp('material_topic').value, 10)
        self.friction_pub = self.create_publisher(String, gp('friction_topic').value, 10)
        self.conf_pub = self.create_publisher(Float32, gp('confidence_topic').value, 10)
        self.debug_pub = (
            self.create_publisher(Image, gp('debug_image_topic').value, 1)
            if self.publish_debug else None
        )

        self._latest_frame = None
        period = 1.0 / max(0.1, self.inference_hz)
        self.timer = self.create_timer(period, self._tick)

        self.get_logger().info(
            f"classifier_node ready | engine={self.engine_path} | classes={self.class_names} "
            f"| roi(xywh)={self.roi} | hz={self.inference_hz}")

    def destroy_node(self):
        try:
            self.classifier.close()
        except Exception:
            pass
        super().destroy_node()

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
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"cv_bridge failed: {e}")
            return

        patch = self._crop_roi(frame)
        if patch is None:
            self.get_logger().warn("ROI is outside the image; check classifier.yaml")
            return

        idx, conf, _probs = self.classifier.classify(patch)
        if idx >= len(self.class_names):
            self.get_logger().warn(
                f"class index {idx} out of range for class_names (len={len(self.class_names)})")
            return
        material_raw = self.class_names[idx]

        self.history.append(material_raw)
        material_smoothed = Counter(self.history).most_common(1)[0][0]
        friction_tier = self.friction.to_friction(material_smoothed)

        self.material_pub.publish(String(data=material_smoothed))
        self.friction_pub.publish(String(data=friction_tier))
        self.conf_pub.publish(Float32(data=conf))

        if self.debug_pub is not None:
            dbg = patch.copy()
            label = f"{material_smoothed} ({friction_tier}) {conf:.2f}"
            cv2.putText(dbg, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2, cv2.LINE_AA)
            dbg_msg = self.bridge.cv2_to_imgmsg(dbg, encoding='bgr8')
            dbg_msg.header = msg.header
            self.debug_pub.publish(dbg_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ClassifierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

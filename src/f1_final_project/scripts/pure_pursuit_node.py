#!/usr/bin/env python3
import rclpy, os, csv, yaml
import numpy as np
from rclpy.node import Node
from scipy.interpolate import splprep, splev
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, PointStamped
from ackermann_msgs.msg import AckermannDriveStamped
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Float32MultiArray, String

class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit')
        
        # --- PARAMETERS ---
        self.declare_parameters(namespace='', parameters=[
            ('waypoint_file', 'ice_waypoints.csv'), 
            ('spacing', 0.15), ('spline_smooth', 0.05),
            ('max_v', 1.5), ('min_v', 0.4), 
            ('max_lat_accel', 0.8), ('max_lon_accel', 0.5), 
            ('lookahead_min', 0.7), ('lookahead_max', 2.5), ('lookahead_gain', 1.5), 
            ('k_sensitivity', 0.5), ('k_lookahead_window', 14),
            ('turn_p', 1.0), ('friction_level', 0) # 0=Use ros params, 1=low, 2=medium, 3=high 
        ])

        self.waypoints = [] # Array of [x, y, velocity, curvature]
        self.pts = [] # Raw waypoints from CSV
        
        # Load friction profiles from YAML
        self.friction_profiles = self.load_friction_profiles()
        
        # --- ROS INFRASTRUCTURE ---
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.target_pub = self.create_publisher(PointStamped, '/debug/target_wp', 10)
        self.path_pub = self.create_publisher(Path, '/debug/waypoints_path', 10)
        self.debug_pub = self.create_publisher(Float32MultiArray, '/debug/pure_pursuit_stats', 10)
        self.pose_sub = self.create_subscription(Odometry, '/pf/pose/odom', self.odom_callback, 10)
        self.friction_sub = self.create_subscription(String, '/surface/friction', self.friction_callback, 10)
        
        self.add_on_set_parameters_callback(self.param_callback)
        self.load_and_process()
        self.create_timer(2.0, self.publish_path) # Static path viz

    def friction_callback(self, msg):
        """Updates friction level based on incoming surface condition messages."""
        level_map = {'low': 1, 'medium': 2, 'high': 3}
        new_level = level_map.get(msg.data.lower(), 0)
        
        if new_level != self.get_parameter('friction_level').value:
            self.get_logger().info(f"Friction level changed to: {msg.data} (Level {new_level})")
            self.set_parameters([rclpy.parameter.Parameter('friction_level', rclpy.Parameter.Type.INTEGER, new_level)])

    def load_friction_profiles(self):
        """Loads the friction profiles from the config directory."""
        pkg_path = os.path.join(os.environ.get('AMENT_PREFIX_PATH').split(':')[0], '..', '..', 
                                'src', 'f1_final_project')
        yaml_path = os.path.join(pkg_path, 'config', 'friction_profiles.yaml')
        
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                return yaml.safe_load(f) or {}
        else:
            self.get_logger().warn(f"Friction profiles YAML not found at: {yaml_path}")
            return {}

    def p(self, name):
        """
        Parameter wrapper: Returns YAML value if friction_level > 0, 
        otherwise returns standard ROS parameter value.
        """
        level = self.get_parameter('friction_level').value
        
        # If level is 0, or we failed to load the YAML, use standard ROS param
        if level == 0 or not self.friction_profiles:
            return self.get_parameter(name).value
        
        profile_key = f'level_{level}'
        
        # If the parameter exists in the specific level override, use it.
        # Otherwise, fall back to the ROS parameter.
        if profile_key in self.friction_profiles and name in self.friction_profiles[profile_key]:
            return self.friction_profiles[profile_key][name]
            
        return self.get_parameter(name).value

    def load_and_process(self):
        """Pipeline: Load CSV -> Interpolate -> Spline -> Physics Profile."""
        pkg_path = os.path.join(os.environ.get('AMENT_PREFIX_PATH').split(':')[0], '..', '..', 
                                'src', 'f1_final_project', 'waypoints')
        file_path = os.path.join(pkg_path, self.p('waypoint_file'))
        
        if not os.path.exists(file_path):
            self.get_logger().error(f"File not found: {file_path}")
            return

        with open(file_path, 'r') as f:
            self.pts = np.array([list(map(float, r)) for r in list(csv.reader(f))[1:] if r])
        # 1. GEOMETRY: Linear density followed by spline smoothing
        spacing = self.p('spacing')
        lin_pts = []
        for i in range(len(self.pts)):
            p0, p1 = self.pts[i], self.pts[(i+1) % len(self.pts)]
            vec, dist = p1 - p0, np.linalg.norm(p1 - p0)
            if dist < 0.01: continue
            for j in range(max(int(dist/spacing), 1)): 
                lin_pts.append(p0 + vec * (j/max(int(dist/spacing), 1)))
        
        try:
            tck, _ = splprep(np.array(lin_pts).T, s=self.p('spline_smooth'), per=True)
            smooth_pts = np.array(splev(np.linspace(0, 1, len(lin_pts)), tck)).T
        except: smooth_pts = np.array(lin_pts)

        # STAGE 2: PHYSICS (Circular Curvature Calculation)
        n = len(smooth_pts)
        k, v = np.zeros(n), np.zeros(n)
        win = 3 

        for i in range(n):
            idx_p = (i - win) % n
            idx_n = (i + win) % n
            
            p0, p1, p2 = smooth_pts[idx_p], smooth_pts[i], smooth_pts[idx_n]
            va, vb = p1 - p0, p2 - p1
            la, lb = np.linalg.norm(va), np.linalg.norm(vb)
            
            k[i] = np.arccos(np.clip(np.dot(va,vb)/(la*lb + 1e-6), -1, 1)) / ((la+lb)/2 + 1e-6)
            v[i] = np.sqrt(self.p('max_lat_accel') / (k[i] + 1e-6))
        
        v = np.clip(v, self.p('min_v'), self.p('max_v'))
        
        # Forward/Backward passes to respect longitudinal acceleration limits
        alon = self.p('max_lon_accel')
        for _ in range(2):
            for i in reversed(range(n)):
                nxt = (i+1)%n
                ds = np.linalg.norm(smooth_pts[nxt]-smooth_pts[i])
                v[i] = min(v[i], np.sqrt(v[nxt]**2 + 2*alon*ds))
            for i in range(n):
                prv = (i-1)%n
                ds = np.linalg.norm(smooth_pts[i]-smooth_pts[prv])
                v[i] = min(v[i], np.sqrt(v[prv]**2 + 2*alon*ds))

        self.waypoints = np.column_stack((smooth_pts, v, k))

    def odom_callback(self, msg):
        if not len(self.waypoints): return
        
        curr = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        idx = np.argmin(np.linalg.norm(self.waypoints[:,:2] - curr, axis=1))
        
        # 3. ADAPTIVE LOOKAHEAD: Shrink Ld if high curvature is detected ahead
        win = self.p('k_lookahead_window')
        ahead_k = np.max(self.waypoints[[(idx + j - 2) % len(self.waypoints) for j in range(win)], 3])
        
        Ld = (self.p('lookahead_gain') * self.waypoints[idx, 2]) / \
             (1.0 + (ahead_k * self.p('k_sensitivity')))
        Ld = np.clip(Ld, self.p('lookahead_min'), self.p('lookahead_max'))

        # 4. PURSUIT: Find target point and calculate steering
        target = self.waypoints[idx]
        for i in range(len(self.waypoints)):
            wp = self.waypoints[(idx + i) % len(self.waypoints)]
            if np.linalg.norm(wp[:2] - curr) > Ld:
                target = wp
                break

        # Transform target to local frame for lateral error (ly)
        q = msg.pose.pose.orientation
        yaw = np.arctan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y + q.z*q.z))
        dx, dy = target[0]-curr[0], target[1]-curr[1]
        ly = -dx * np.sin(yaw) + dy * np.cos(yaw) 
        
        # Command Drive
        drive = AckermannDriveStamped()
        drive.header.stamp = self.get_clock().now().to_msg()
        drive.drive.speed = float(target[2])
        drive.drive.steering_angle = np.clip(float((self.p('turn_p') * ly) / (Ld**2)), -0.4, 0.4)
        self.drive_pub.publish(drive)

        # Debug Visuals
        tp = PointStamped()
        tp.header.frame_id, tp.header.stamp = "map", self.get_clock().now().to_msg()
        tp.point.x, tp.point.y = target[0], target[1]
        self.target_pub.publish(tp)

        # Create debug message
        debug_msg = Float32MultiArray()
        debug_msg.data = [float(Ld), float(target[2]), float(ahead_k), float(ly)]
        self.debug_pub.publish(debug_msg)

    def param_callback(self, params):
        # Deferred execution via one-shot timer ensures params are fully saved in ROS map
        self.update_timer = self.create_timer(0.1, self.delayed_load)
        return SetParametersResult(successful=True)

    def delayed_load(self):
        self.update_timer.cancel()
        self.destroy_timer(self.update_timer)
        self.load_and_process()

    def publish_path(self):
        if not len(self.waypoints): return
        msg = Path()
        msg.header.frame_id, msg.header.stamp = "map", self.get_clock().now().to_msg()
        for wp in self.waypoints:
            p = PoseStamped()
            p.pose.position.x, p.pose.position.y = wp[0], wp[1]
            msg.poses.append(p)
        self.path_pub.publish(msg)

def main():
    rclpy.init(); rclpy.spin(PurePursuit()); rclpy.shutdown()

if __name__ == '__main__': main()
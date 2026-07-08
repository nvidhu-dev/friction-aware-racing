#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>

struct WP {
    double x, y, v, k;
};

class PurePursuit : public rclcpp::Node {
public:
    PurePursuit() : Node("pure_pursuit") {
        // --- PARAMETERS ---
        this->declare_parameter("waypoint_file", "my_map_wp.csv");
        this->declare_parameter("spacing", 0.15);
        this->declare_parameter("spline_smooth", 0.05);
        this->declare_parameter("max_v", 3.0);
        this->declare_parameter("min_v", 0.5);
        this->declare_parameter("max_lat_accel", 1.5);
        this->declare_parameter("max_lon_accel", 1.0);
        this->declare_parameter("lookahead_min", 0.7);
        this->declare_parameter("lookahead_max", 2.5);
        this->declare_parameter("lookahead_gain", 1.5);
        this->declare_parameter("k_sensitivity", 1.2);
        this->declare_parameter("k_lookahead_window", 14);
        this->declare_parameter("turn_p", 1.5);

        // --- ROS INFRASTRUCTURE ---
        drive_pub = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);
        target_pub = this->create_publisher<geometry_msgs::msg::PointStamped>("/debug/target_wp", 10);
        path_pub = this->create_publisher<nav_msgs::msg::Path>("/debug/waypoints_path", 10);
        debug_pub = this->create_publisher<std_msgs::msg::Float32MultiArray>("/debug/pure_pursuit_stats", 10);
        
        pose_sub = this->create_subscription<nav_msgs::msg::Odometry>(
            "/pf/pose/odom", 10, std::bind(&PurePursuit::odom_callback, this, std::placeholders::_1));

        param_handler = this->add_on_set_parameters_callback(
            std::bind(&PurePursuit::param_callback, this, std::placeholders::_1));

        load_and_process();
        path_timer = this->create_wall_timer(std::chrono::milliseconds(2000), std::bind(&PurePursuit::publish_path, this));
    }

private:
    void load_and_process() {
        std::string filename = this->get_parameter("waypoint_file").as_string();
        std::string file_path = "../../src/f1_final_project/waypoints/" + filename;
        
        std::ifstream file(file_path);
        if (!file.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "File not found: %s", file_path.c_str());
            return;
        }

        std::vector<std::pair<double, double>> pts;
        std::string line;
        std::getline(file, line); 
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string item;
            std::vector<double> row;
            while (std::getline(ss, item, ',')) row.push_back(std::stod(item));
            if (row.size() >= 2) pts.push_back({row[0], row[1]});
        }

        double spacing = this->get_parameter("spacing").as_double();
        std::vector<std::pair<double, double>> smooth_pts;
        for (size_t i = 0; i < pts.size(); ++i) {
            auto p0 = pts[i];
            auto p1 = pts[(i + 1) % pts.size()];
            double dx = p1.first - p0.first, dy = p1.second - p0.second;
            double dist = std::sqrt(dx*dx + dy*dy);
            if (dist < 0.01) continue;
            int steps = std::max((int)(dist / spacing), 1);
            for (int j = 0; j < steps; ++j) {
                smooth_pts.push_back({p0.first + dx * (j / (double)steps), p0.second + dy * (j / (double)steps)});
            }
        }

        size_t n = smooth_pts.size();
        waypoints.clear();
        waypoints.resize(n);
        int win = 3;
        double max_lat = this->get_parameter("max_lat_accel").as_double();
        double min_v = this->get_parameter("min_v").as_double();
        double max_v = this->get_parameter("max_v").as_double();

        for (size_t i = 0; i < n; ++i) {
            int ip = (i - win + n) % n, in = (i + win) % n;
            auto p0 = smooth_pts[ip], p1 = smooth_pts[i], p2 = smooth_pts[in];
            double vax = p1.first - p0.first, vay = p1.second - p0.second;
            double vbx = p2.first - p1.first, vby = p2.second - p1.second;
            double la = std::sqrt(vax*vax + vay*vay), lb = std::sqrt(vbx*vbx + vby*vby);
            double dot = vax*vbx + vay*vby;
            double k = std::acos(std::clamp(dot / (la * lb + 1e-6), -1.0, 1.0)) / ((la + lb) / 2.0 + 1e-6);
            double v = std::sqrt(max_lat / (k + 1e-6));
            waypoints[i] = {p1.first, p1.second, std::clamp(v, min_v, max_v), k};
        }

        double alon = this->get_parameter("max_lon_accel").as_double();
        for (int pass = 0; pass < 2; ++pass) {
            for (int i = (int)n - 1; i >= 0; --i) {
                int nxt = (i + 1) % n;
                double ds = std::sqrt(std::pow(waypoints[nxt].x - waypoints[i].x, 2) + std::pow(waypoints[nxt].y - waypoints[i].y, 2));
                waypoints[i].v = std::min(waypoints[i].v, std::sqrt(std::pow(waypoints[nxt].v, 2) + 2 * alon * ds));
            }
            for (size_t i = 0; i < n; ++i) {
                int prv = (i - 1 + n) % n;
                double ds = std::sqrt(std::pow(waypoints[i].x - waypoints[prv].x, 2) + std::pow(waypoints[i].y - waypoints[prv].y, 2));
                waypoints[i].v = std::min(waypoints[i].v, std::sqrt(std::pow(waypoints[prv].v, 2) + 2 * alon * ds));
            }
        }
    }

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        if (waypoints.empty()) return;
        double cx = msg->pose.pose.position.x, cy = msg->pose.pose.position.y;
        size_t idx = 0;
        double min_d = 1e18;
        for (size_t i = 0; i < waypoints.size(); ++i) {
            double d = std::pow(waypoints[i].x - cx, 2) + std::pow(waypoints[i].y - cy, 2);
            if (d < min_d) { min_d = d; idx = i; }
        }

        int win = this->get_parameter("k_lookahead_window").as_int();
        double ahead_k = 0;
        for (int j = 0; j < win; ++j) ahead_k = std::max(ahead_k, waypoints[(idx + j) % waypoints.size()].k);

        double Ld = (this->get_parameter("lookahead_gain").as_double() * waypoints[idx].v) / 
                    (1.0 + (ahead_k * this->get_parameter("k_sensitivity").as_double()));
        Ld = std::clamp(Ld, this->get_parameter("lookahead_min").as_double(), this->get_parameter("lookahead_max").as_double());

        WP target = waypoints[idx];
        for (size_t i = 0; i < waypoints.size(); ++i) {
            const auto& wp = waypoints[(idx + i) % waypoints.size()];
            if (std::sqrt(std::pow(wp.x - cx, 2) + std::pow(wp.y - cy, 2)) > Ld) { target = wp; break; }
        }

        auto q = msg->pose.pose.orientation;
        double yaw = std::atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z));
        double dx = target.x - cx, dy = target.y - cy;
        double ly = -dx * std::sin(yaw) + dy * std::cos(yaw);

        auto drive = ackermann_msgs::msg::AckermannDriveStamped();
        drive.header.stamp = this->now();
        drive.drive.speed = (float)target.v;
        drive.drive.steering_angle = std::clamp((float)((this->get_parameter("turn_p").as_double() * ly) / (Ld * Ld)), -0.4f, 0.4f);
        drive_pub->publish(drive);

        std_msgs::msg::Float32MultiArray dbg;
        dbg.data = {(float)Ld, (float)target.v, (float)ahead_k, (float)ly};
        debug_pub->publish(dbg);
    }

    rcl_interfaces::msg::SetParametersResult param_callback(const std::vector<rclcpp::Parameter> &params) {
        (void)params;
        this->create_wall_timer(std::chrono::milliseconds(100), [this]() { this->load_and_process(); });
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        result.reason = "OK";
        return result;
    }

    void publish_path() {
        if (waypoints.empty()) return;
        auto msg = nav_msgs::msg::Path();
        msg.header.frame_id = "map";
        msg.header.stamp = this->now();
        for (const auto& wp : waypoints) {
            geometry_msgs::msg::PoseStamped p;
            p.pose.position.x = wp.x; p.pose.position.y = wp.y;
            msg.poses.push_back(p); // CHANGED FROM append TO push_back
        }
        path_pub->publish(msg);
    }

    std::vector<WP> waypoints;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr target_pub;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr debug_pub;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr pose_sub;
    rclcpp::TimerBase::SharedPtr path_timer;
    OnSetParametersCallbackHandle::SharedPtr param_handler;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PurePursuit>());
    rclcpp::shutdown();
    return 0;
}
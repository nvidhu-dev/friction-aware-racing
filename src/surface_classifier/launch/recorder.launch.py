from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('surface_classifier')
    config = os.path.join(pkg_share, 'config', 'classifier.yaml')

    current_label = LaunchConfiguration('current_label')

    return LaunchDescription([
        DeclareLaunchArgument(
            'current_label',
            default_value='unlabeled',
            description='Active class label for recorded patches '
                        '(change at runtime: ros2 param set /recorder_node current_label <label>).',
        ),
        Node(
            package='surface_classifier',
            executable='recorder_node',
            name='recorder_node',
            output='screen',
            parameters=[config, {'current_label': current_label}],
        ),
    ])

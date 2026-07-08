from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('surface_classifier')
    config = os.path.join(pkg_share, 'config', 'classifier.yaml')

    return LaunchDescription([
        Node(
            package='surface_classifier',
            executable='classifier_node',
            name='classifier_node',
            output='screen',
            parameters=[config],
        ),
    ])

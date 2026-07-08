from setuptools import setup
from glob import glob
import os

package_name = 'surface_classifier'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'models'), glob('models/*.trt') + glob('models/*.onnx')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nandgopalv',
    maintainer_email='nandgopalv@gmail.com',
    description='F1Tenth ground-surface material/friction classifier (TensorRT MobileNetV3).',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'classifier_node = surface_classifier.classifier_node:main',
            'recorder_node = surface_classifier.recorder_node:main',
        ],
    },
)

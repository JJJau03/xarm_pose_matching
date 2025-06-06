from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'xarm_pose_matching'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jjj',
    maintainer_email='jjjau03@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': ['detector_node = xarm_pose_matching.detector_node:main',
                            'pointcloud_cropper_node = xarm_pose_matching.pointcloud_cropper_node:main',    
                            'pose_init_node = xarm_pose_matching.pose_init_node:main',       
                            'pose_match_node = xarm_pose_matching.pose_match_node:main',    
        ],
    },
)

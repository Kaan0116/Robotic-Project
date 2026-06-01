from setuptools import setup
import os
from glob import glob

package_name = 'pf_localization'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        # ament index
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        # package.xml
        ('share/' + package_name, ['package.xml']),
        # worlds
        (os.path.join('share', package_name, 'worlds'),
         glob('worlds/*.sdf')),
        # AR tag textures
        (os.path.join('share', package_name, 'worlds', 'materials', 'textures'),
         glob('worlds/materials/textures/*')),
        # launch
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.py')),
        # config
        (os.path.join('share', package_name, 'config'),
         glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yuksel Kaan Bolukbas',
    maintainer_email='kaan@example.com',
    description='Particle Filter Localization with AR Tags',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'particle_filter  = pf_localization.particle_filter_node:main',
            'aruco_detector   = pf_localization.aruco_detector_node:main',
            'visualizer       = pf_localization.visualizer_node:main',
        ],
    },
)

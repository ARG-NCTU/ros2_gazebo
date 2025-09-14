import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'js5_sitl'

# Helper function to recursively get all files in a directory
def get_model_files(model_dir):
    files = []
    for root, dirs, filenames in os.walk(model_dir):
        for filename in filenames:
            files.append((
                os.path.join('share', package_name, root),
                [os.path.join(root, filename)]
            ))
    return files

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # Launch files
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'launch/robots'),
            glob(os.path.join('launch/robots', '*.launch.py'))),

        # World files
        (os.path.join('share', package_name, 'worlds'),
            glob(os.path.join('worlds', '*.sdf'))),

        # Bridge configurations - FIXED: Should be 'config' not 'gz_bridges'
        (os.path.join('share', package_name, 'gz_bridges'),
            glob(os.path.join('gz_bridges', '*.yaml'))),

        # RViz configurations
        (os.path.join('share', package_name, 'rviz'),
            glob(os.path.join('rviz', '*.rviz'))),

        # Model files - FIXED: Properly recursive installation
        # JS5 model files
        *get_model_files('models/js5'),
        
        # 3D Lidar model files  
        *get_model_files('models/3d_lidar'),
        
        # Sensor post model files
        *get_model_files('models/sensor_post'),

        *get_model_files('models/gps'),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)

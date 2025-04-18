import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'blueboat_sitl'

model_data_files = [
    (os.path.join('share', package_name, root), [os.path.join(root, f) for f in files])
    for root, dirs, files in os.walk('models')
]

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py'))),

        (os.path.join('share', package_name, 'launch/robots'),
            glob(os.path.join('launch/robots', '*.launch.py'))),

        (os.path.join('share', package_name, 'worlds'),
            glob(os.path.join('worlds', '*.sdf'))),

        (os.path.join('share', package_name, 'gz_bridges'),
            glob(os.path.join('gz_bridges', '*.yaml'))),

        (os.path.join('share', package_name, 'rviz'),
            glob(os.path.join('rviz', '*.rviz'))),
    
    ] + model_data_files,
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

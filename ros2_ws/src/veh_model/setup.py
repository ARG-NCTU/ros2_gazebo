import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'veh_model'

model_data_files = [
    (os.path.join('share', package_name, root), [os.path.join(root, f) for f in files])
    for root, dirs, files in os.walk('models')
]
launch_data_files = [
    (os.path.join('share', package_name, root), [os.path.join(root, f) for f in files if f.endswith('.launch.py')])
    for root, dirs, files in os.walk('launch')
]
setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        (os.path.join('share', package_name, 'worlds'),
            glob(os.path.join('worlds', '*.sdf'))),
        
    ] + model_data_files + launch_data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'joy2cmdvel = veh_model.joy2cmdvel:main',
            'bb_twist2thrust = veh_model.blueboat.twist2thrust:main',
            'bb_gz_reset_node = veh_model.blueboat.gz_reset_node:main',
            'wamv_alpha_twist2thrust = veh_model.wamv_alpha.twist2thrust:main',
            'wamv_v1_twist2thrust = veh_model.wamv_v1.twist2thrust:main',
            'wamv_v2_twist2thrust = veh_model.wamv_v2.twist2thrust:main',
            'wamv_v1_sb3_dp = veh_model.wamv_v1.sb3_dp:main',
            'wamv_v1_acme_dp = veh_model.wamv_v1.acme_dp:main',
            'wamv_v2_acme_dp = veh_model.wamv_v2.acme_dp:main',
        ],
    },
)

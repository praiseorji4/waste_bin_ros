import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            RegisterEventHandler, SetEnvironmentVariable)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    pkg_description = get_package_share_directory('bin_description')
    # Parent of the package share dir: lets Gazebo resolve model://bin_description/meshes/...
    gz_resource_path = os.path.dirname(pkg_description)

    # 1. Process URDF with use_gazebo:=true
    robot_description_config = ParameterValue(
        Command([
            'xacro ',
            os.path.join(pkg_description, 'urdf', 'bin_robot.urdf.xacro'),
            ' use_gazebo:=true'
        ]),
        value_type=str
    )
    
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_config, 
            'use_sim_time': True
        }]
    )
    
    # 2. Gazebo
    world_file = os.path.join(
        get_package_share_directory('bin_bringup'),
        'worlds',
        'basic.sdf'
        # 'warehouse_world.sdf'
    )

    # 2. Modify the gazebo launch description
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')]),
        launch_arguments={'gz_args': f'-r -v 4 {world_file}'}.items(),
    )


    # 3. ROS-Gazebo Bridge (Clock, Cmd_vel, Odom, TF, Lidar, Camera)
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            # '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
            '/lidar@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            
            # FIXED RGB-D BRIDGE MAPPING
            '/camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',

            # IMU BRIDGE
            '/imu@sensor_msgs/msg/Imu[gz.msgs.IMU'
        ],
        remappings=[
            # The lidar rides on the lid in sim too, so route the raw feed through
            # scan_gate: bridge publishes /lidar_raw, scan_gate republishes /lidar.
            ('/lidar', '/lidar_raw'),
            ('/camera/image', '/camera/rgb/image_raw'),
            ('/camera/depth_image', '/camera/depth/image_raw'),
            ('/camera/camera_info', '/camera/rgb/camera_info'),
            ('/imu', '/imu/data')
        ],
        output='screen',
        parameters=[{'use_sim_time': True}]
    )
    
    # 4. Spawn Robot Entity
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-topic', 'robot_description', 
            '-name', 'bin_bot',
            '-world', 'sensors', 
            '-z', '0.1'
        ],
    )
    
    # 5. Controller Spawners
    load_joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster", 
            "--param-file", 
            os.path.join(get_package_share_directory('bin_bringup'), 
                        'config', 'bin_controllers.yaml')
        ],
        parameters=[{'use_sim_time': True}]
    )
    
    load_diff_drive_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "diff_drive_controller",
            "--param-file",
            os.path.join(get_package_share_directory('bin_bringup'),
                        'config', 'bin_controllers.yaml')
        ],
        parameters=[{'use_sim_time': True}]
    )

    load_bin_cover_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "bin_cover_controller",
            "--param-file",
            os.path.join(get_package_share_directory('bin_bringup'),
                        'config', 'bin_controllers.yaml')
        ],
        parameters=[{'use_sim_time': True}]
    )
    
    # 6. Twist Stamper (for teleop compatibility)
    node_twist_stamper = Node(
        package='twist_stamper',
        executable='twist_stamper',
        parameters=[{
            'use_sim_time': True,
            'frame_id': 'base_footprint'
        }],
        remappings=[
            ('/cmd_vel_in', '/cmd_vel'),
            ('/cmd_vel_out', '/diff_drive_controller/cmd_vel'),
        ]
    )

    bin_bringup_dir = get_package_share_directory('bin_bringup')
    bin_description_dir = get_package_share_directory('bin_description')
    slam_toolbox_dir = get_package_share_directory('slam_toolbox')
    # Construction of the variable
    nav2_launch_dir = os.path.join(get_package_share_directory('nav2_bringup'), 'launch')

    # Path to SLAM parameters
    slam_params_file = os.path.join(bin_bringup_dir, 'config', 'mapper_params_online_async.yaml')
    
    # Path to RViz configuration
    rviz_config_file = os.path.join(bin_description_dir, 'rviz', 'slam.rviz')

    # 7. Include SLAM Toolbox
    slam_toolbox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_toolbox_dir, 'launch', 'online_async_launch.py')
        ),
        launch_arguments={
            'slam_params_file': slam_params_file,
            'use_sim_time': 'true'
        }.items()
    )

    # 8. RViz2 Node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': True}]
    )


    ekf_config_path = os.path.join(
        get_package_share_directory('bin_bringup'),
        'config', 'ekf.yaml'
    )

    # Disabled: bin_controllers.yaml now sets enable_odom_tf: true, so diff_drive_controller
    # owns odom->base_footprint. Running the EKF as well gives two publishers of that same
    # transform and the model ghosts between them. bin_real does not run an EKF either, and
    # the real robot has no IMU wired up yet (the <sensor> block in bin_ros2_control.xacro is
    # commented out and imu_sensor_broadcaster is never spawned), so fusing here would only
    # make sim diverge from hardware. To re-enable: set enable_odom_tf: false first.
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_config_path, {'use_sim_time': True}]
    )

    nav2_params_path = os.path.join(
            get_package_share_directory('bin_bringup'),
            'config',
            'nav2_params.yaml'
        )

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_launch_dir, 'navigation_launch.py')),
        launch_arguments={
            'use_sim_time': 'true',
            'params_file': nav2_params_path
        }.items(),

    )

    # Which trigger source opens the lid:
    #   peace : MediaPipe peace-sign gesture (bin_perception/peace_detector)
    #   color : original blue/green blob detector (bin_behavior/color_detector)
    #   none  : no automatic trigger; publish /bin_trigger by hand
    declare_detector = DeclareLaunchArgument(
        'detector', default_value='peace', choices=['peace', 'color', 'none'],
        description='Trigger source for opening the bin lid')
    detector = LaunchConfiguration('detector')

    def when(value):
        return IfCondition(PythonExpression(["'", detector, "' == '", value, "'"]))

    color_detector_node = Node(
        package='bin_behavior',
        executable='color_detector',
        name='color_detector',
        output='screen',
        condition=when('color'),
        parameters=[{'use_sim_time': True}]
    )

    # Blocks laser scans while the lid is open: the lidar is bolted to the lid, so an
    # open lid sweeps floor and ceiling and corrupts the SLAM map and both costmaps.
    scan_gate_node = Node(
        package='bin_perception',
        executable='scan_gate',
        name='scan_gate',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'input_topic': '/lidar_raw',
            'output_topic': '/lidar',
            'joint_name': 'bin_cover_joint',
            'closed_tolerance': 0.05,
            'settle_time': 0.5,
        }]
    )

    peace_detector_node = Node(
        package='bin_perception',
        executable='peace_detector',
        name='peace_detector',
        output='screen',
        condition=when('peace'),
        parameters=[{
            'use_sim_time': True,
            'input_topic': '/camera/rgb/image_raw',
            'use_compressed': False,
            'consecutive_frames': 4,
            'cooldown_s': 5.0,
            'process_every_n': 2,
        }]
    )

    behavior_manager_node = Node(
        package='bin_behavior',
        executable='behavior_manager',
        name='behavior_manager',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            # Nav2 runs in this launch file, so docking after the lid closes works here.
            # On the real robot Nav2 lives on the laptop, so bin_real leaves it off.
            'dock_after_close': True,
            'dwell_s': 3.0,
            'move_timeout_s': 6.0,
        }]
    )

    return LaunchDescription([
        declare_detector,
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', gz_resource_path),
        node_robot_state_publisher,
        gazebo,
        bridge,
        spawn_entity,
        node_twist_stamper,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_entity,
                on_exit=[load_joint_state_broadcaster, load_diff_drive_controller, load_bin_cover_controller],
            )
        ),
        # ekf_node,  # see note at its definition — would duplicate odom->base_footprint
        rviz_node,
        slam_toolbox,
        nav2,
        scan_gate_node,
        color_detector_node,
        peace_detector_node,
        behavior_manager_node,
    ])
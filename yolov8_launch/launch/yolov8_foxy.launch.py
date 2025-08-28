from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, LogInfo
from launch_ros.actions import Node


def generate_launch_description() :

    weight = LaunchConfiguration('weight')
    device = LaunchConfiguration('device')
    conf_threshold = LaunchConfiguration('conf_threshold')
    iou_threshold = LaunchConfiguration('iou_threshold')
    input_type = LaunchConfiguration('input_type')
    input_path = LaunchConfiguration('input_path')
    camera_id = LaunchConfiguration('camera_id')
    width = LaunchConfiguration('width')
    height = LaunchConfiguration('height')
    fps = LaunchConfiguration('fps')
    namespace = LaunchConfiguration('namespace')    

    return LaunchDescription([
        DeclareLaunchArgument(
            'weight',
            default_value='/home/ljq/ros2_ws/best.pt',
            description='weight model path or name'
        ),
        DeclareLaunchArgument(
            'device',
            default_value='cuda:0',
            description='Device type(GPU|CPU)'
        ),
        DeclareLaunchArgument(
            'conf_threshold',
            default_value='0.5',
            description='NMS confidence threshold '    
        ),
        DeclareLaunchArgument(
            'iou_threshold',
            default_value='0.7',
            description='NMS IoU threshold'
        ),
        # 输入类型和路径参数
        DeclareLaunchArgument(
            'input_type',
            default_value='camera',
            description='Input type: camera, video, image'
        ),
        DeclareLaunchArgument(
            'input_path',
            default_value='',
            description='Path to video or image file (only used for video or image mode)'
        ),
        DeclareLaunchArgument(
            'camera_id',
            default_value='0',
            description='Camera device ID, usually 0 or 1 (only used for camera mode)'
        ),
        DeclareLaunchArgument(
            'width',
            default_value='640',
            description='Camera width (pixels)'
        ),
        DeclareLaunchArgument(
            'height',
            default_value='480',
            description='Camera height (pixels)'
        ),
        DeclareLaunchArgument(
            'fps',
            default_value='30',
            description='Frame rate (camera mode) or target frame rate (video mode, 0 means original frame rate)'
        ),
        DeclareLaunchArgument(
            'namespace',
            default_value='yolov8',
            description='Node namespace'
        ),
        LogInfo(
            msg="\nUsage:\n" +
                "- Camera mode: ros2 launch yolov8_launch yolov8_foxy.launch.py input_type:=camera camera_id:=0\n" +
                "- Video mode: ros2 launch yolov8_launch yolov8_foxy.launch.py input_type:=video input_path:=/path/to/video.mp4\n" +
                "- Image mode: ros2 launch yolov8_launch yolov8_foxy.launch.py input_type:=image input_path:=/path/to/image.jpg\n"
        ),
        # YOLOv8节点
        Node(
            package='yolov8_main',  
            executable='yolov8_node',  
            name='yolov8_node',
            parameters=[
                {'weight': weight},
                {'device': device},
                {'conf_threshold': conf_threshold},  
                {'iou_threshold': iou_threshold},
                # Input type parameters
                {'input_type': input_type},
                {'input_path': input_path},
                {'camera_id': camera_id},
                {'width': width},
                {'height': height},
                {'fps': fps}
            ],
            output='screen'  # Display all output to screen
        )
    ])

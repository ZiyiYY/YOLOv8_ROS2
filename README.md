# yolov8_ros2

ROS2(foxy version) wrapper for [YOLOv8](https://github.com/ultralytics/ultralytics). This package enables you to perform object detection and segmentation.

## Installation

### Prerequisites
- Ubuntu 20.04
- [CUDA](https://developer.nvidia.com/cuda-downloads)
- [CUDNN](https://developer.nvidia.com/cudnn-downloads)
- [ROS2 Foxy](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html)

### Build the package

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/ban2aru/yolov8-ros2.git
cd ..
sudo apt update
pip install -r ./src/yolov8-ros2/requirements.txt
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build
source ./install/setup.bash
```

## Usage

### Start the YOLOv8 node

```bash
# 默认使用相机模式
ros2 launch yolov8_launch yolov8_foxy.launch.py
```

### Input source

#### 1. Use camera

```bash
ros2 launch yolov8_launch yolov8_foxy.launch.py input_type:=camera


```

#### 2. Use video   

```bash
ros2 launch yolov8_launch yolov8_foxy.launch.py input_type:=video input_path:=/path/to/your/video.mp4
```


### Use Segmentation model

```bash
ros2 launch yolov8_launch yolov8_foxy.launch.py weight:=yolov8n-seg.pt
```

## Parameters

### Configuration parameters
- **input_type**:（camera, video, image）。默认为`camera`
- **weight**: YOLOv8 model weight. Default is `yolov8n.pt`
- **device**: device type (GPU/CUDA/CPU). Default is `cuda:0`
- **conf_threshold**: NMS confidence threshold. Default is `0.5`
- **iou_threshold**: NMS IOU threshold. Default is `0.5`
- **class_names**: Class names file path. Default is `coco.names`
- **model_type**: YOLOv8 model type (n, s, m, l, x). Default is `n`
- **segmentation**: Use segmentation model. Default is `False`
- **segmentation_threshold**: Segmentation threshold. Default is `0.5`



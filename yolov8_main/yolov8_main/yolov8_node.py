import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from cv_bridge import CvBridge

from ultralytics import YOLO

from sensor_msgs.msg import Image
from yolov8_msg.msg import Yolov8InferenceMsg
from yolov8_msg.msg import Yolov8Inference
from yolov8_msg.msg import BoundingBox
from yolov8_msg.msg import Mask
from yolov8_msg.msg import Point2D

import cv2
import random
import numpy as np
import signal
import sys
import threading
import time
import os

class Yolov8Node(Node) :
    
    def __init__(self) :
        super().__init__('yolov8_node')

        self._class_to_color = {}
        self._shutdown_requested = False

        # parameter default setting
        self.declare_parameter('weight', 'yolov8n.pt')
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('conf_threshold', '0.5')
        self.declare_parameter('iou_threshold', '0.7')
        
        # 输入源参数
        self.declare_parameter('input_type', 'camera')  # camera, video, image
        self.declare_parameter('input_path', '')  # 视频或图像文件路径，如果是相机则留空
        self.declare_parameter('camera_id', 0)  # 相机ID
        
        # 相机/视频参数
        self.declare_parameter('width', 1920)
        self.declare_parameter('height', 1080)
        self.declare_parameter('fps', 30)

        # get parameter value
        weight = self.get_parameter('weight').get_parameter_value().string_value
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.conf_threshold = self.get_parameter('conf_threshold').get_parameter_value().double_value
        self.iou_threshold = self.get_parameter('iou_threshold').get_parameter_value().double_value
        
        self.input_type = self.get_parameter('input_type').get_parameter_value().string_value
        self.input_path = self.get_parameter('input_path').get_parameter_value().string_value
        self.camera_id = self.get_parameter('camera_id').get_parameter_value().integer_value
        
        self.width = self.get_parameter('width').get_parameter_value().integer_value
        self.height = self.get_parameter('height').get_parameter_value().integer_value
        self.fps = self.get_parameter('fps').get_parameter_value().integer_value

        # parameter for image processing
        self.cv_bridge = CvBridge()
        self.model = YOLO(weight)

        # create publishers for results
        self.pub = self.create_publisher(Yolov8InferenceMsg, '/yolov8_inference', 1)
        self.img_pub = self.create_publisher(Image, '/yolov8_result', 1)
        
        # 初始化视频捕获
        self.cap = None
        self.camera_thread = None
        self.initialize_capture()
            
        self.get_logger().info('YOLOv8 node started')
    
    def initialize_capture(self):
        """初始化视频捕获（相机或视频文件）"""
        try:
            if self.input_type == 'camera':
                self.get_logger().info(f'初始化相机模式，相机ID: {self.camera_id}')
                self.cap = cv2.VideoCapture(self.camera_id)
                if not self.cap.isOpened():
                    self.get_logger().error(f'无法打开相机 ID: {self.camera_id}')
                    return
                
                # 设置相机参数
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                
            elif self.input_type == 'video':
                if not os.path.exists(self.input_path):
                    self.get_logger().error(f'视频文件不存在: {self.input_path}')
                    return
                
                self.get_logger().info(f'初始化视频模式，文件路径: {self.input_path}')
                self.cap = cv2.VideoCapture(self.input_path)
                if not self.cap.isOpened():
                    self.get_logger().error(f'无法打开视频文件: {self.input_path}')
                    return
                
                # 获取视频属性
                self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                self.get_logger().info(f'视频信息: {self.width}x{self.height}, {self.fps}fps, 共{total_frames}帧')
                
            elif self.input_type == 'image':
                if not os.path.exists(self.input_path):
                    self.get_logger().error(f'图像文件不存在: {self.input_path}')
                    return
                
                self.get_logger().info(f'初始化单张图像模式，文件路径: {self.input_path}')
                # 单张图像模式不需要视频捕获，直接读取并处理图像
                img = cv2.imread(self.input_path)
                if img is None:
                    self.get_logger().error(f'无法读取图像文件: {self.input_path}')
                    return
                
                self.width = img.shape[1]
                self.height = img.shape[0]
                self.process_frame(img)
                self.get_logger().info('单张图像处理完成')
                return
            else:
                self.get_logger().error(f'不支持的输入类型: {self.input_type}')
                return
            
            # 启动视频处理线程（相机和视频模式）
            self.camera_thread = threading.Thread(target=self.capture_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            self.get_logger().info('视频处理线程已启动')
            
        except Exception as e:
            self.get_logger().error(f'初始化视频捕获时出错: {str(e)}')
    
    def capture_loop(self):
        """视频捕获和处理循环"""
        frame_time = 1.0 / self.fps
        
        while not self._shutdown_requested and rclpy.ok():
            start_time = time.time()
            
            # 读取帧
            ret, frame = self.cap.read()
            if not ret:
                # 如果是视频文件，可以选择从头开始播放或停止
                if self.input_type == 'video':
                    self.get_logger().info('视频播放完毕，正在重新开始...')
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到视频开头
                    continue
                else:
                    self.get_logger().warn('无法从相机读取帧')
                    time.sleep(0.1)
                    continue
            
            # 处理帧
            self.process_frame(frame)
            
            # 控制帧率
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def process_frame(self, img):
        """处理图像帧"""
        self.get_logger().info(f'处理图像，形状: {img.shape}')

        # yolov8 results
        results = self.model.predict(
            source = img,
            conf = self.conf_threshold,
            iou = self.iou_threshold,
            device = self.device)
        self.get_logger().info(f'YOLOv8 prediction completed. Found {len(results[0])} objects')
        results = results[0].cpu()

        inference_msg = Yolov8InferenceMsg()
        
        # 创建自定义的消息头
        from std_msgs.msg import Header
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "camera"
        inference_msg.header = header

        if not results.boxes:
            self.get_logger().info('No objects detected')
            cv2.imshow("YOLOv8 Result", img)
            cv2.waitKey(1)
            # 即使没有检测到物体，也发布处理后的图像
            self.img_pub.publish(self.cv_bridge.cv2_to_imgmsg(img, 'bgr8'))
            return
        
        for i in range(len(results)):
            inference_result = Yolov8Inference()

            if results.boxes:
                bbox_info = results.boxes[i]
                inference_result.class_id = int(bbox_info.cls)
                inference_result.class_name = self.model.names[int(bbox_info.cls)]
                inference_result.score = float(bbox_info.conf)

                bbox_msg = BoundingBox()

                bbox = bbox_info.xywh[0]
                bbox_msg.center.x = float(bbox[0])
                bbox_msg.center.y = float(bbox[1])
                bbox_msg.size.x = float(bbox[2])
                bbox_msg.size.y = float(bbox[3])
                
                inference_result.bbox = bbox_msg

                label = inference_result.class_name

                if label not in self._class_to_color:
                    r = random.randint(0, 255)
                    g = random.randint(0, 255)
                    b = random.randint(0, 255)
                    self._class_to_color[label] = (r, g, b)
                
                color = self._class_to_color[label]

                pt1 = (round(bbox_msg.center.x - bbox_msg.size.x / 2.0),
                        round(bbox_msg.center.y - bbox_msg.size.y / 2.0))
                pt2 = (round(bbox_msg.center.x + bbox_msg.size.x / 2.0),
                        round(bbox_msg.center.y + bbox_msg.size.y / 2.0))
                cv2.rectangle(img, pt1, pt2, color, 2)
                cv2.putText(img, str(inference_result.class_name), ((pt1[0]+pt2[0])//2-5, pt1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2)

            if results.masks:
                mask_info = results.masks[i]
                mask_msg = Mask()
                mask_msg.data = [Point2D(x=float(point[0]), y=float(point[1])) for point in mask_info.xy[0].tolist()]
                mask_msg.height = results.orig_img.shape[0]
                mask_msg.width = results.orig_img.shape[1]

                inference_result.mask = mask_msg

                mask_array = np.array([[int(point.x), int(point.y)] for point in mask_msg.data])

                temp = img.copy()
                temp = cv2.fillPoly(temp, [mask_array], color)
                cv2.addWeighted(img, 0.4, temp, 0.6, 0, img)
                img = cv2.polylines(img, [mask_array], True, color, 2)

            inference_msg.yolov8_inference.append(inference_result)

        self.img_pub.publish(self.cv_bridge.cv2_to_imgmsg(img, 'bgr8'))
        self.pub.publish(inference_msg)
        
        # 检查是否请求关闭
        if self._shutdown_requested:
            return
            
        # Display the result
        cv2.imshow("YOLOv8 Result", img)
        
        # 单图像模式等待按键
        if self.input_type == 'image':
            self.get_logger().info('按任意键关闭图像窗口')
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            cv2.waitKey(1)

    def on_shutdown(self):
        """处理节点关闭时的清理工作"""
        self.get_logger().info('Shutting down YOLOv8 node...')
        self._shutdown_requested = True
        
        # 关闭视频捕获
        if self.cap is not None:
            self.cap.release()
            self.get_logger().info('视频捕获已关闭')
        
        # 等待线程结束
        if self.camera_thread is not None:
            self.camera_thread.join(timeout=1.0)
            self.get_logger().info('视频处理线程已关闭')
            
        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()
        
        # 如果有必要，释放CUDA资源
        if hasattr(self, 'model'):
            try:
                del self.model
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.get_logger().info('YOLO model resources released')
            except Exception as e:
                self.get_logger().error(f'Error releasing YOLO model: {str(e)}')
        
        self.get_logger().info('YOLOv8 node shutdown complete')

def main():
    rclpy.init(args=None)
    node = Yolov8Node()
    
    # 添加一个事件标志来控制主循环退出
    running = True
    
    # 注册信号处理器以更好地处理Ctrl+C
    def signal_handler(sig, frame):
        nonlocal running
        node.get_logger().info('Received shutdown signal')
        node.on_shutdown()
        running = False
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 使用自定义循环而不是rclpy.spin()
        while running and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        # 如果通过KeyboardInterrupt捕获到Ctrl+C
        if running:  # 如果信号处理器尚未处理
            node.get_logger().info('Keyboard interrupt detected')
            node.on_shutdown()
    finally:
        # 确保总是清理资源
        node.destroy_node()
        rclpy.shutdown()
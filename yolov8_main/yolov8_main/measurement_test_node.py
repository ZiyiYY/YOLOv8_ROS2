import rclpy
from rclpy.node import Node
import os
import time
import copy
import numpy as np
import threading
from yolov8_msg.msg import Yolov8InferenceMsg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def camera2bodyFRDrotationMatrix(cameraPitchRad):
    # camera lower than body, then cameraPitch is positive
    R_y = np.array([
            [np.cos(cameraPitchRad), 0, -np.sin(cameraPitchRad)],
            [0, 1, 0],
            [np.sin(cameraPitchRad), 0, np.cos(cameraPitchRad)]
        ])
    return R_y

def quaternion2euler(q):
    [q0, q1, q2, q3] = q
    roll = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
    pitch = np.arcsin(2 * (q0 * q2 - q3 * q1))
    yaw = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    return np.array([roll, pitch, yaw])

def ned2frdRotationMatrix(rpyRadNED):
    rollAngle = rpyRadNED[0]
    pitchAngle = rpyRadNED[1]
    yawAngle = rpyRadNED[2]
    R_z = np.array([
            [np.cos(yawAngle), np.sin(yawAngle), 0],
            [-np.sin(yawAngle), np.cos(yawAngle), 0],
            [0, 0, 1]
        ])
        
    R_y = np.array([
            [np.cos(pitchAngle), 0, -np.sin(pitchAngle)],
            [0, 1, 0],
            [np.sin(pitchAngle), 0, np.cos(pitchAngle)]
        ])
        
    R_x = np.array([
            [1, 0, 0],
            [0, np.cos(rollAngle), np.sin(rollAngle)],
            [0, -np.sin(rollAngle), np.cos(rollAngle)]
        ])

    ned2frdRotationMatrix = R_x @ R_y @ R_z
    return ned2frdRotationMatrix

def frd2nedRotationMatrix(rpyRadNED):
    R = np.linalg.inv(ned2frdRotationMatrix(rpyRadNED))
    return R

def ned2enu(vec):
    CHECK(vec)
    T = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, -1]
    ])
    return T @ vec

class MeasurementTestNode(Node):
    def __init__(self):
        super().__init__('measurement_test_node')
        self.startTime = time.time()
        self.cameraSub = self.create_subscription(Yolov8InferenceMsg, '/yolov8_inference', self.camera_callback, 10)
        # self.targetPositionSub = self.create_subscription(, '/Target/pose', self.targetPosition_callback, 10)
        # self.uavPositionSub = self.create_subscription(, '/UAV/pose', self.uavPosition_callback, 10)
        self.targetPosition = np.array([0, 0, 0])
        self.uavPosition = np.array([0, 0, 0])
        self.losTruth = np.array([0.0, 0.0])
        self.lookAngleFRD = None
        self.losMeasureENU = None
        self.cameraPitchRAD = None
        self.imageSizePixelXY= None
        self.targetCenterPixelXY = None
        self.targetSizePixelXY = None
        self.data = []
        self.cameraFOV = [np.deg2rad(69.0), np.deg2rad(42.0)]

        self.fig, self.ax = plt.subplots(2, 1, figsize=(8, 10))
        # self.losTruth_lines = [self.ax[0].plot([], [], label='Real elevation angle')[0], self.ax[1].plot([], [], label='Real azimuth angle')[0]]
        # self.losMeasure_lines = [self.ax[0].plot([], [], label='Measurement of elevation angle')[0], self.ax[1].plot([], [], label='Measurement of azimuth angle')[0]]
        self.lookAngle_lines = [self.ax[0].plot([], [], label='elevation look angle')[0], self.ax[1].plot([], [], label='azimuth look angle')[0]]

        for num, direction in enumerate(['elevation angle', 'azimuth angle']):
            # self.ax[num].set_xlim(0, 20)
            self.ax[num].set_ylim(-90, 90) 
            self.ax[num].set_xlabel('Time (s)')
            self.ax[num].set_ylabel('Angle (deg)')
            self.ax[num].legend(loc='upper right')
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=100, save_count=100)
        

    def camera_callback(self, msg):
        self.imageSizePixelXY = np.array([msg.yolov8_inference[0].img_size.x, msg.yolov8_inference[0].img_size.y])
        self.targetCenterPixelXY = np.array([msg.yolov8_inference[0].bbox.center.x, msg.yolov8_inference[0].bbox.center.y])
        self.targetSizePixelXY = np.array([msg.yolov8_inference[0].bbox.size.x, msg.yolov8_inference[0].bbox.size.y])
        self.getLosMeasure()

    # def targetPosition_callback(self, msg):
    #     self.targetPosition = np.array([msg.position.x, msg.position.y, msg.position.z])
    
    # def uavPosition_callback(self, msg):
    #     self.uavPosition = np.array([msg.position.x, msg.position.y, msg.position.z])
    #     self.uavQuaternion =
    #   self.uavRPY =  quaternion2euler(self.uavQuaternion)
        
    def getLosTruth(self):
        relativePosition = self.targetPosition - self.uavPosition
        self.losTruth = np.array([
                np.arctan2(relativePosition[2], np.sqrt(relativePosition[0] ** 2 + relativePosition[1] ** 2)),
                np.arctan2(relativePosition[1], relativePosition[0])
            ])
    
    def getLosMeasure(self):
        imageCenterPixel = np.array([self.imageSizePixelXY[0]/2, self.imageSizePixelXY[1]/2])
        targetPixel = self.targetCenterPixelXY - imageCenterPixel
        self.lookAngleFRD = np.array([np.arctan(targetPixel[1]/(self.imageSizePixelXY[1]/2) * np.tan(self.cameraFOV[1]/2)),
                               np.arctan(targetPixel[0]/(self.imageSizePixelXY[0]/2) * np.tan(self.cameraFOV[0]/2))])
        # LOSdirectionCameraFRD = np.array([1, np.tan(self.lookAngleFRD[1]), np.tan(self.lookAngleFRD[0])])/np.linalg.norm(np.array([1, np.tan(self.lookAngleFRD[1]), np.tan(self.lookAngleFRD[0])]))
        # LOSdirectionBodyFRD = camera2bodyFRDrotationMatrix(self.cameraPitchRAD) @ LOSdirectionCameraFRD
        # LOSdirectionNED = frd2nedRotationMatrix(self.uavRPY) @ LOSdirectionBodyFRD
        # LOSdirectionENU = ned2enu(LOSdirectionNED)
        # self.losMeasureENU = np.array([np.arctan2(LOSdirectionENU[2], np.sqrt(LOSdirectionENU[0] ** 2 + LOSdirectionENU[1] ** 2)),
        #                     np.arctan2(LOSdirectionENU[1], LOSdirectionENU[0])])
        self.log()
    
    def log(self):
        currentData = {
            't': copy.copy(time.time() - self.startTime),
            # 'losTruth': copy.copy(self.losTruth),
            # 'losMeasure':copy.copy(self.losMeasureENU),
            'lookAngle': copy.copy(self.lookAngleFRD)
        }
        print(f"{currentData = }")
        self.data.append(currentData)

    
    def update_plot(self, frame):

        if len(self.data) > 0:
            time_data = [d['t'] for d in self.data]
            # losTruth = np.array([d['losTruth'] for d in self.data]).T
            # losMeasure = np.array([d['losMeasure'] for d in self.data]).T
            lookAngle = np.array([d['lookAngle'] for d in self.data]).T
            
            # self.losTruth_lines[0].set_data(time_data, np.rad2deg(losTruth[0]))
            # self.losTruth_lines[1].set_data(time_data, np.rad2deg(losTruth[1]))
            # self.losMeasure_lines[0].set_data(time_data, np.rad2deg(losMeasure[0]))
            # self.losMeasure_lines[1].set_data(time_data, np.rad2deg(losMeasure[1]))
            self.lookAngle_lines[0].set_data(time_data, np.rad2deg(lookAngle[0]))
            self.lookAngle_lines[1].set_data(time_data, np.rad2deg(lookAngle[1]))
            
            self.ax[0].relim()
            self.ax[0].autoscale_view()
            self.ax[1].relim()
            self.ax[1].autoscale_view()
            self.fig.canvas.draw()

def main(args=None):
    rclpy.init(args=args)
    measurementTestNode = MeasurementTestNode()
    spinThread = threading.Thread(target=lambda: rclpy.spin(measurementTestNode))
    spinThread.start()

    try:
        measurementTestNode.update_plot(0)
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        measurementTestNode.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
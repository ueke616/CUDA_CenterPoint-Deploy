import rospy
import ros_numpy
import numpy as np
import copy
import json
import os
import sys
import torch
import time 

from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from pyquaternion import Quaternion

sys.path.insert(0, "/data/nuscenes/BEV/mit-bevfusion/NVIDIA-AI-IOT/Lidar_AI_Solution/CUDA-CenterPoint/CenterPoint")
from det3d import torchie
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.core.input.voxel_generator import VoxelGenerator

# import debugpy
# debugpy.listen(11123)
# print("wait for debugger")
# debugpy.wait_for_client()
# print("debugger attached")

def yaw2quaternion(yaw: float) -> Quaternion:  # 将偏航角转换为四元数，用于表示物体的方向
    return Quaternion(axis=[0,0,1], radians=yaw)

def get_annotations_indices(types, thresh, label_preds, scores):  # 用于过滤低分数的检测结果
    indexs = []
    annotation_indices = []
    for i in range(label_preds.shape[0]):
        if label_preds[i] == types:
            indexs.append(i)
    for index in indexs:
        if scores[index] >= thresh:
            annotation_indices.append(index)
    return annotation_indices  


def remove_low_score_nu(image_anno, thresh):  # 用于过滤低分数的检测结果
    img_filtered_annotations = {}
    label_preds_ = image_anno["label_preds"].detach().cpu().numpy()
    scores_ = image_anno["scores"].detach().cpu().numpy()
    
    car_indices =                  get_annotations_indices(0, 0.4, label_preds_, scores_)
    truck_indices =                get_annotations_indices(1, 0.4, label_preds_, scores_)
    construction_vehicle_indices = get_annotations_indices(2, 0.4, label_preds_, scores_)
    bus_indices =                  get_annotations_indices(3, 0.3, label_preds_, scores_)
    trailer_indices =              get_annotations_indices(4, 0.4, label_preds_, scores_)
    barrier_indices =              get_annotations_indices(5, 0.4, label_preds_, scores_)
    motorcycle_indices =           get_annotations_indices(6, 0.15, label_preds_, scores_)
    bicycle_indices =              get_annotations_indices(7, 0.15, label_preds_, scores_)
    pedestrain_indices =           get_annotations_indices(8, 0.1, label_preds_, scores_)
    traffic_cone_indices =         get_annotations_indices(9, 0.1, label_preds_, scores_)
    
    for key in image_anno.keys():
        if key == 'metadata':
            continue
        img_filtered_annotations[key] = (
            image_anno[key][car_indices +
                            pedestrain_indices + 
                            bicycle_indices +
                            bus_indices +
                            construction_vehicle_indices +
                            traffic_cone_indices +
                            trailer_indices +
                            barrier_indices +
                            truck_indices
                            ])

    return img_filtered_annotations


class Processor_ROS:
    """
    这个类负责管理与CenterPoint模型相关的所有操作。它包括初始化模型、读取配置文件、运行模型以及辅助函数。
    """
    def __init__(self, config_path, model_path):  
        # 初始化处理器实例，设置配置文件路径和模型路径
        self.points = None
        self.config_path = config_path
        self.model_path = model_path
        self.device = None
        self.net = None
        self.voxel_generator = None
        self.inputs = None
        
    def initialize(self):
        # 用于读取配置文件并初始化模型
        self.read_config()
        
    def read_config(self):  # 从配置文件加载模型参数，并创建模型实例和体素生成器
        config_path = self.config_path
        cfg = Config.fromfile(self.config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        state_dict = torch.load(self.model_path)["state_dict"]
        state_dict['backbone.conv_input.0.weight'] = state_dict['backbone.conv_input.0.weight'].reshape(16, 3, 3, 3, 5)
        state_dict['backbone.conv1.0.conv1.weight'] = state_dict['backbone.conv1.0.conv1.weight'].reshape(16, 3, 3, 3, 16)
        state_dict['backbone.conv1.0.conv2.weight'] = state_dict['backbone.conv1.0.conv2.weight'].reshape(16, 3, 3, 3, 16)
        state_dict['backbone.conv1.1.conv1.weight'] = state_dict['backbone.conv1.1.conv1.weight'].reshape(16, 3, 3, 3, 16)
        state_dict['backbone.conv1.1.conv2.weight'] = state_dict['backbone.conv1.1.conv2.weight'].reshape(16, 3, 3, 3, 16)
        state_dict['backbone.conv2.0.weight'] = state_dict['backbone.conv2.0.weight'].reshape(32, 3, 3, 3, 16)
        state_dict['backbone.conv2.3.conv1.weight'] = state_dict['backbone.conv2.3.conv1.weight'].reshape(32, 3, 3, 3, 32)
        state_dict['backbone.conv2.3.conv2.weight'] = state_dict['backbone.conv2.3.conv2.weight'].reshape(32, 3, 3, 3, 32)
        state_dict['backbone.conv2.4.conv1.weight'] = state_dict['backbone.conv2.4.conv1.weight'].reshape(32, 3, 3, 3, 32)
        state_dict['backbone.conv2.4.conv2.weight'] = state_dict['backbone.conv2.4.conv2.weight'].reshape(32, 3, 3, 3, 32)
        state_dict['backbone.conv3.0.weight'] = state_dict['backbone.conv3.0.weight'].reshape(64, 3, 3, 3, 32)
        state_dict['backbone.conv3.3.conv1.weight'] = state_dict['backbone.conv3.3.conv1.weight'].reshape(64, 3, 3, 3, 64)
        state_dict['backbone.conv3.3.conv2.weight'] = state_dict['backbone.conv3.3.conv2.weight'].reshape(64, 3, 3, 3, 64)
        state_dict['backbone.conv3.4.conv1.weight'] = state_dict['backbone.conv3.4.conv1.weight'].reshape(64, 3, 3, 3, 64)
        state_dict['backbone.conv3.4.conv2.weight'] = state_dict['backbone.conv3.4.conv2.weight'].reshape(64, 3, 3, 3, 64)
        state_dict['backbone.conv4.0.weight'] = state_dict['backbone.conv4.0.weight'].reshape(128, 3, 3, 3, 64)
        state_dict['backbone.conv4.3.conv1.weight'] = state_dict['backbone.conv4.3.conv1.weight'].reshape(128, 3, 3, 3, 128)
        state_dict['backbone.conv4.3.conv2.weight'] = state_dict['backbone.conv4.3.conv2.weight'].reshape(128, 3, 3, 3, 128)
        state_dict['backbone.conv4.4.conv1.weight'] = state_dict['backbone.conv4.4.conv1.weight'].reshape(128, 3, 3, 3, 128)
        state_dict['backbone.conv4.4.conv2.weight'] = state_dict['backbone.conv4.4.conv2.weight'].reshape(128, 3, 3, 3, 128)
        state_dict['backbone.extra_conv.0.weight'] = state_dict['backbone.extra_conv.0.weight'].reshape(128, 3, 1, 1, 128)
        
        self.net.load_state_dict(state_dict)
        self.net = self.net.to(self.device).eval()

        self.range = cfg.voxel_generator.range
        self.voxel_size = cfg.voxel_generator.voxel_size
        self.max_points_in_voxel = cfg.voxel_generator.max_points_in_voxel
        self.max_voxel_num = cfg.voxel_generator.max_voxel_num
        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.range,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num,
        )

    def run(self, points):  # 接收原始点云数据，通过体素生成器处理后输入到CenterPoint模型中，并返回检测结果
        t_t = time.time()
        print(f"input points shape: {points.shape}")
        num_features = 5        
        self.points = points.reshape([-1, num_features])
        self.points[:, 4] = 0 # timestamp value 
        
        voxels, coords, num_points = self.voxel_generator.generate(self.points)
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)
        grid_size = self.voxel_generator.grid_size
        coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values = 0)
        
        voxels = torch.tensor(voxels, dtype=torch.float32, device=self.device)
        coords = torch.tensor(coords, dtype=torch.int32, device=self.device)
        num_points = torch.tensor(num_points, dtype=torch.int32, device=self.device)
        num_voxels = torch.tensor(num_voxels, dtype=torch.int32, device=self.device)
        
        self.inputs = dict(
            voxels = voxels,
            num_points = num_points,
            num_voxels = num_voxels,
            coordinates = coords,
            shape = [grid_size]
        )
        torch.cuda.synchronize()
        t = time.time()

        with torch.no_grad():
            outputs = self.net(self.inputs, return_loss=False)[0]
    
        # print(f"output: {outputs}")
        
        torch.cuda.synchronize()
        print("  network predict time cost:", time.time() - t)

        outputs = remove_low_score_nu(outputs, 0.45)

        boxes_lidar = outputs["box3d_lidar"].detach().cpu().numpy()
        print("  predict boxes:", boxes_lidar.shape)

        scores = outputs["scores"].detach().cpu().numpy()
        types = outputs["label_preds"].detach().cpu().numpy()

        boxes_lidar[:, -1] = -boxes_lidar[:, -1] - np.pi / 2

        print(f"  total cost time: {time.time() - t_t}")

        return scores, boxes_lidar, types

def get_xyz_points(cloud_array, remove_nans=True, dtype=np.float64):
    """
    被设计用来从ROS点云消息的结构化数组中提取XYZ坐标，并可选地移除包含NaN值的点。
    """
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]

    points = np.zeros(cloud_array.shape + (5,), dtype=dtype)
    points[...,0] = cloud_array['x']
    points[...,1] = cloud_array['y']
    points[...,2] = cloud_array['z']
    return points

def xyz_array_to_pointcloud2(points_sum, stamp=None, frame_id=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array of points.
    '''
    msg = PointCloud2()
    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = points_sum.shape[0]
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)
        # PointField('i', 12, PointField.FLOAT32, 1)
        ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = points_sum.shape[0]
    msg.is_dense = int(np.isfinite(points_sum).all())
    msg.data = np.asarray(points_sum, np.float32).tostring()
    return msg

def rslidar_callback(msg):
    """
    是一个ROS回调函数，用于接收点云数据，调用Processor_ROS类处理这些数据，然后发布检测到的物体的边界框信息
    """
    t_t = time.time()
    arr_bbox = BoundingBoxArray()

    msg_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
    np_p = get_xyz_points(msg_cloud, True)
    print("  ")
    scores, dt_box_lidar, types = proc_1.run(np_p)

    if scores.size != 0:
        for i in range(scores.size):
            bbox = BoundingBox()
            bbox.header.frame_id = msg.header.frame_id
            bbox.header.stamp = rospy.Time.now()
            q = yaw2quaternion(float(dt_box_lidar[i][8]))
            bbox.pose.orientation.x = q[1]
            bbox.pose.orientation.y = q[2]
            bbox.pose.orientation.z = q[3]
            bbox.pose.orientation.w = q[0]           
            bbox.pose.position.x = float(dt_box_lidar[i][0])
            bbox.pose.position.y = float(dt_box_lidar[i][1])
            bbox.pose.position.z = float(dt_box_lidar[i][2])
            bbox.dimensions.x = float(dt_box_lidar[i][4])
            bbox.dimensions.y = float(dt_box_lidar[i][3])
            bbox.dimensions.z = float(dt_box_lidar[i][5])
            bbox.value = scores[i]
            bbox.label = int(types[i])
            arr_bbox.boxes.append(bbox)
    print("total callback time: ", time.time() - t_t)
    arr_bbox.header.frame_id = msg.header.frame_id
    arr_bbox.header.stamp = msg.header.stamp
    if len(arr_bbox.boxes) is not 0:
        pub_arr_bbox.publish(arr_bbox)
        arr_bbox.boxes = []
    else:
        arr_bbox.boxes = []
        pub_arr_bbox.publish(arr_bbox)
   
if __name__ == "__main__":
    global proc
    ## CenterPoint
    config_path = 'configs/nusc/voxelnet/nusc_centerpoint_voxelnet_01voxel.py'
    model_path = 'work_dirs/nusc_centerpoint_voxelnet_01voxel/latest.pth'

    proc_1 = Processor_ROS(config_path, model_path)  # 创建了Processor_ROS的实例
    
    proc_1.initialize()  # 初始化ROS节点
    
    rospy.init_node('centerpoint_ros_node')
    sub_lidar_topic = [ "/velodyne_points", 
                        "/top/rslidar_points",
                        "/points_raw", 
                        "/lidar_protector/merged_cloud", 
                        "/merged_cloud",
                        "/lidar_top", 
                        "/roi_pclouds"]
    # 设置订阅者和发布者
    sub_ = rospy.Subscriber(sub_lidar_topic[5], PointCloud2, rslidar_callback, queue_size=1, buff_size=2**24)
    
    # 订阅者订阅激光雷达的点云主题，发布者发布检测到的物体的边界框信息
    pub_arr_bbox = rospy.Publisher("pp_boxes", BoundingBoxArray, queue_size=1)

    print("[+] CenterPoint ros_node has started!")    
    rospy.spin()


"""
代码流程
    创建处理器实例，读取配置和模型。
    初始化ROS节点。
    定义点云数据的订阅者和边界框信息的发布者。
    当接收到点云数据时，通过回调函数处理数据，并发布检测到的物体的边界框。
注意事项
    代码假定已经安装和配置了ROS环境以及CenterPoint模型所需的依赖。
    需要根据实际环境调整点云主题名称和路径设置。
    模型的配置文件和权重文件路径需要根据实际情况进行修改。
"""
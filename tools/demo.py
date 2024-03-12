import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import argparse
import copy
import json
import os
import sys

# 尝试导入 apex 库, 用于混合精度训练, 如果失败则提示
try:
    import apex
except:
    print("No APEX!")

# 导入科学计算和深度学习相关的库
import numpy as np
import torch
import yaml
# 导入 det3d 库相关模块，用于 3D 物体检测
sys.path.insert(0, "/data/nuscenes/BEV/mit-bevfusion/NVIDIA-AI-IOT/Lidar_AI_Solution/CUDA-CenterPoint/CenterPoint")
from det3d import torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    batch_processor,
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)
from det3d.torchie.trainer import load_checkpoint
import pickle 
import time 
from matplotlib import pyplot as plt 
from det3d.torchie.parallel import collate, collate_kitti
from torch.utils.data import DataLoader
import matplotlib.cm as cm
import subprocess
import cv2
from tools.demo_utils import visual 
from collections import defaultdict

import debugpy
debugpy.listen(11123)
print("wait for debugger")
debugpy.wait_for_client()
print("debugger attached")

def convert_box(info):
    """
    此函数用于将给定的标注信息(info)转换为检测算法可以使用的格式。它主要处理了gt_boxes（真实标注的3D盒模型）和gt_names（对应的类别名称），并确保了每个盒模型都有一个对应的类别名称
    """
    # 转换标注信息中的盒子坐标
    boxes =  info["gt_boxes"].astype(np.float32)
    names = info["gt_names"]

    # 确保盒子和名称数量一致
    assert len(boxes) == len(names)

    detection = {}
    detection['box3d_lidar'] = boxes
    # dummy value 哑变量，通常用于处理类别数据（如性别、国籍、种类等），是用来反映质量属性的一种手段
    detection['label_preds'] = np.zeros(len(boxes)) 
    detection['scores'] = np.ones(len(boxes))
    return detection 

def main():
    # 从配置文件中加载配置
    cfg = Config.fromfile('configs/nusc/voxelnet/nusc_centerpoint_voxelnet_01voxel.py')
    # 构建检测模型
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    # 构建数据集
    dataset = build_dataset(cfg.data.val)
    # 数据加载器设置
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=None,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_kitti,
        pin_memory=False,
    )
    # 加载模型检查点
    checkpoint = load_checkpoint(model, 'work_dirs/nusc_centerpoint_voxelnet_01voxel/latest.pth', map_location="cpu")
    model.eval()    # 设置为评估模式
    model = model.cuda()    # 将模型移到CUDA

    cpu_device = torch.device("cpu")
    # 初始化列表用于存储点云数据、真实标注和检测结果
    points_list = [] 
    gt_annos = [] 
    detections  = [] 
    # 遍历数据加载器
    for i, data_batch in enumerate(data_loader):
        info = dataset._nusc_infos[i]
        gt_annos.append(convert_box(info))

        points = data_batch['points'][0][:, 1:4].cpu().numpy()
        with torch.no_grad():  # 不计算梯度
            outputs = batch_processor(
                model, data_batch, train_mode=False, local_rank=0,
            )
        for output in outputs:
            for k, v in output.items():
                if k not in [
                    "metadata",
                ]:
                    output[k] = v.to(cpu_device)
            detections.append(output)

        points_list.append(points.T)
    
    print('完成模型推理，请稍候，matplotlib 绘图可能稍慢...')
    
    for i in range(len(points_list)):
        visual(points_list[i], gt_annos[i], detections[i], i)
        print("渲染图片 {}".format(i))
    
    # 视频保存设置
    image_folder = 'demo'
    video_name = 'video.avi'
    # 读取并排序图像文件
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=lambda img_name: int(img_name.split('.')[0][4:]))
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    # 创建视频写入器
    video = cv2.VideoWriter(video_name, 0, 1, (width,height))
    cv2_images = []  # 初始化图像列表

    for image in images:
        cv2_images.append(cv2.imread(os.path.join(image_folder, image)))

    for img in cv2_images:
        video.write(img)

    cv2.destroyAllWindows()
    video.release()

    print("成功保存视频到主文件夹")

if __name__ == "__main__":
    main()

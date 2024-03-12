export PATH=/usr/local/cuda-11.8/bin:$PATH
export CUDA_PATH=/usr/local/cuda-11.8
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

echo $LD_LIBRARY_PATH

# nuScenes
# python tools/create_data.py nuscenes_data_prep --root_path=nuScenes --version="v1.0-trainval" --nsweeps=10

# nuScenes
python tools/create_data.py nuscenes_data_prep --root_path=nuScenes --version="v1.0-trainval" --nsweeps=10

# https://blog.csdn.net/weixin_36354875/article/details/127618711
# 执行训练脚本
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh

# 现在，我们只支持使用 GPU 进行训练和评估。不支持仅 CPU 模式。
# 使用以下命令使用 4 个 GPU 启动分布式训练。模型和日志将保存到work_dirs/CONFIG_NAME
torchrun --nproc_per_node=4 ./tools/train.py CONFIG_PATH

# Download the centerpoint_voxel_1440_flip here, save it into work_dirs/nusc_0075_flip, then run the following commands in the main folder to get detection prediction
# python tools/dist_test.py configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z_flip.py --work_dir work_dirs/nusc_centerpoint_voxelnet_dcn_0075voxel_flip_testset  --checkpoint work_dirs/nusc_0075_flip/voxelnet_converted.pth  --testset 
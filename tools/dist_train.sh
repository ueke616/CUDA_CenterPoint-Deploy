# CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train.py configs/nusc/voxelnet/nusc_centerpoint_voxelnet_01voxel.py --gpus=4 --autoscale-lr
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train.py configs/nusc/voxelnet/nusc_centerpoint_voxelnet_01voxel.py --autoscale-lr
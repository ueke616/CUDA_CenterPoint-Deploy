export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=true
CUDA_VISIBLE_DEVICES=1,2,3 python tools/dist_test.py configs/nusc/voxelnet/nusc_centerpoint_voxelnet_01voxel.py --work_dir=test_visualize/test --checkpoint=work_dirs/nusc_centerpoint_voxelnet_01voxel/latest.pth --gpus=3 --launcher=pytorch --local_rank=0
# --eval-options='show=False' --out='./eval_visualize/nusc_test.pkl'
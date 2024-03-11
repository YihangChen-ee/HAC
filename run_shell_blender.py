import os

for lmbda in [0.004]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
    for cuda, scene in enumerate(['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']):
        one_cmd = f'CUDA_VISIBLE_DEVICES={0} python train.py -s data/nerf_synthetic/{scene} --eval --lod 0 --voxel_size 0.001 --update_init_factor 4 --iterations 30_000 -m outputs/nerf_synthetic/{scene}/{lmbda} --lmbda {lmbda}'
        os.system(one_cmd)

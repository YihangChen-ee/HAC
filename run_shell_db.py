import os

for lmbda in [0.004]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
    for cuda, scene in enumerate(['playroom', 'drjohnson']):
        one_cmd = f'CUDA_VISIBLE_DEVICES={0} python train.py -s data/blending/{scene} --eval --lod 0 --voxel_size 0.005 --update_init_factor 16 --iterations 30_000 -m outputs/blending/{scene}/{lmbda} --lmbda {lmbda}'
        os.system(one_cmd)

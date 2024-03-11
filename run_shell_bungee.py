import os

for lmbda in [0.004]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
    for cuda, scene in enumerate(['amsterdam', 'bilbao', 'hollywood', 'pompidou', 'quebec', 'rome']):
        one_cmd = f'CUDA_VISIBLE_DEVICES={0} python train.py -s data/bungeenerf/{scene} --eval --lod 30 --voxel_size 0 --update_init_factor 128 --iterations 30_000 -m outputs/bungeenerf/{scene}/{lmbda} --lmbda {lmbda}'
        os.system(one_cmd)

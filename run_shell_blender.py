import os

log2_2D = 15
log2 = 13
n_features = 4

for lmbda in [0.0005, 0.001, 0.002, 0.003, 0.004]:
    for cuda, scene in enumerate(['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']):
        one_cmd = f'CUDA_VISIBLE_DEVICES={0} python train.py -s data/nerf_synthetic/{scene} --eval --lod 0 --voxel_size 0.001 --update_init_factor 4 --iterations 30_000 -m outputs_evaluation/nerf_synthetic/{scene}/{lmbda}/HAC_{log2}_{log2_2D}_{n_features} --log2 {log2} --log2_2D {log2_2D} --n_features {n_features} --lmbda {lmbda}'
        os.system(one_cmd)

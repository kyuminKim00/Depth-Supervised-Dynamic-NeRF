import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import os
import cv2

def _calc_std(frame_path_root, std_path_root, frame_start=0, n_frame=300):
    # if frame_start != 0:
    #     std_path_root = std_path_root+str(frame_start)
    print("calc_std start")
    if os.path.exists(std_path_root):
        print("std already exist")
        return
    os.makedirs(std_path_root)
    print(frame_path_root)
    for child in os.listdir(frame_path_root):
        if not child.startswith('cam'):
            continue
        frame_path = os.path.join(frame_path_root, child)
        std_path = os.path.join(std_path_root, child)
        frame_paths = sorted([os.path.join(frame_path, fn) for fn in os.listdir(frame_path)])[frame_start:frame_start+n_frame]

        frames = []
        for fp in frame_paths:
            frame = Image.open(fp).convert('RGB')
            frame = np.array(frame, dtype=float) / 255.
            frames.append(frame)
        frame = np.stack(frames, axis=0)
        std_map = frame.std(axis=0).mean(axis=-1)
        std_map_blur = (cv2.GaussianBlur(std_map, (31, 31), 0)).astype(float)
        np.save(std_path + '_std.npy', std_map_blur)
        print("calc_std finish")
        print(frame_paths)

root_dir = "/data2/kkm/km/data/flame_steak_static_only_new"
downsample = 4.0
frame_start = 0
n_frames = 300
_calc_std(os.path.join(root_dir, 'frames'+('' if downsample == 1.0 else '_{}'.format(int(downsample)))),
                  os.path.join(root_dir, 'stds'+('' if frame_start==0 else str(frame_start))+('' if downsample == 1.0 else '_{}'.format(int(downsample)))),
                  frame_start=frame_start, n_frame=n_frames)

# cam5 = np.load("depthmap_cam5_all.npy")
# cam6 = np.load("depthmap_cam6_all.npy")
# cam5_depth = torch.Tensor(cam5)
# cam6_depth = torch.Tensor(cam6)
# print(cam5.shape)
# print(cam6.shape)
# # # 각 텐서에 새로운 차원 추가
# cam5_depth = cam5_depth.unsqueeze(2)
# cam6_depth = cam6_depth.unsqueeze(2)

# # # (342732, 1)에서 (342732, 300, 1)로 데이터 반복 확장
# # cam5_depth_expanded = cam5_depth.repeat(1, 300, 1)
# # cam6_depth_expanded = cam6_depth.repeat(1, 300, 1)

# all_depth = torch.cat((cam5_depth, cam6_depth), dim=0)
# print(all_depth.shape)
# all_depth = all_depth.numpy()
# np.save("all_depth_25000.npy", all_depth)
# print("save done")
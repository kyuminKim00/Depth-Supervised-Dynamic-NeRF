import numpy as np
import torch
import os
import cv2
from render_one_video.py import *



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
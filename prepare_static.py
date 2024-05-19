import numpy as np
import torch
from PIL import Image
import os
import shutil
import argparse
import cv2

def resize_image(image_path, output_path, scale=None, resolution=None):
    # Open the image
    img = Image.open(image_path)

    # Calculate new size
    if resolution:
        new_size = resolution
    elif scale:
        original_size = img.size
        new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
    else:
        raise ValueError("Either scale or resolution must be provided.")

    # Resize the image using Lanczos resampling (high-quality)
    resized_img = img.resize(new_size, Image.Resampling.LANCZOS)

    # Save the resized image
    resized_img.save(output_path)

def copy_image(directory_path, original_image_name, frame_num):
    original_image_path = os.path.join(directory_path, original_image_name)
    
    # 0001.png부터 frame_num+1.png까지 반복
    for i in range(1, frame_num+1):
        # 새 파일 이름 설정 (예: 0001.png, 0002.png, ..., 0300.png)
        new_image_name = f"{i:04d}.png"
        new_image_path = os.path.join(directory_path, new_image_name)
        
        # 원본 이미지를 새 파일 이름으로 복사
        shutil.copyfile(original_image_path, new_image_path)

        if i == frame_num+1:  # 마지막 복사가 완료되면 원본 파일 삭제
            os.remove(original_image_path)
        
        print(f"Copied {original_image_path} to {new_image_path}")

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




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--static_data_path', type=str, 
                        help='static data dir path')

    parser.add_argument('--downsample', type=int,
                        default= 4, help= 'downscaling factor')

    parser.add_argument('--frame_num', type=int,
                        default= 300, help= 'number of frame(dynamic model)')

    args = parser.parse_args()

    root_dir = args.static_data_path
    downsample = args.downsample
    frame_start = 0
    n_frames = args.frame_num

    frames_dir = args.static_data_path + "/frames"

    frames_resize_dir = root_dir + "/frames_"+str(args.downsample)
    depth_dir = root_dir +"/depth"
    if not os.path.exists(frames_resize_dir):
        os.makedirs(frames_resize_dir)
    if not os.path.exists(depth_dir):
            os.makedirs(depth_dir)
    
    destination_template = "/cam{:02}/cam{:02}.png"
    frames_path = frames_dir + destination_template

    frame_resize_dir = frames_resize_dir + "/cam{:02}"
    
    cam_dir_list = os.listdir(frames_dir)
    
    for i in range(len(cam_dir_list)):
        
        print("cam_dir_list :", cam_dir_list)
        cam_resize_dir = frames_resize_dir + "/cam{:02}"
        cam_resize_dir_path = cam_resize_dir.format(i)
        if not os.path.exists(cam_resize_dir_path):
            os.makedirs(cam_resize_dir_path) 
        frame_resize_path = cam_resize_dir_path +"/cam{:02}.png".format(i)
        frame_path = frames_path.format(i, i)
        resize_image(frame_path, frame_resize_path, scale=1/args.downsample)  #(image_path, output_path, scale=None, resolution=None)
        copy_image(cam_resize_dir_path, "cam{:02}.png".format(i), n_frames)

    _calc_std(os.path.join(root_dir, 'frames'+('' if downsample == 1.0 else '_{}'.format(int(downsample)))),
                  os.path.join(root_dir, 'stds'+('' if frame_start==0 else str(frame_start))+('' if downsample == 1.0 else '_{}'.format(int(downsample)))),
                  frame_start=frame_start, n_frame=n_frames)
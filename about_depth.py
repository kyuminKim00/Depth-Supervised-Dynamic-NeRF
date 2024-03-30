import numpy as np
import torch
from dataLoader.ray_utils import get_rays_by_coord_np
from dataLoader.ray_utils import ndc_rays_blender
from dataLoader.ray_utils import get_ray_directions_blender
from dataLoader.load_llff import load_colmap_depth

import pandas as pd
import cv2
from Gen_pose_bounds import printdatasetINFO
import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from dataLoader.llff_video import center_poses

import colmapUtils.colmap_read_model as read_model

def load_colmap_data(realdir):
    
    camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)
    
    # cam = camdata[camdata.keys()[0]]
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print( 'Cameras', len(cam))

    h, w, f = cam.height, cam.width, cam.params[0]
    # w, h, f = factor * w, factor * h, factor * f
    hwf = np.array([h,w,f]).reshape([3,1])
    
    imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
    imdata = read_model.read_images_binary(imagesfile)
    
    w2c_mats = []
    bottom = np.array([0,0,0,1.]).reshape([1,4])
    
    names = [imdata[k].name for k in imdata]
    print( 'Images #', len(names))
    perm = np.argsort(names)
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3,1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)
    
    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)
    
    poses = c2w_mats[:, :3, :4].transpose([1,2,0])
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)
    
    points3dfile = os.path.join(realdir, 'sparse/0/points3D.bin')
    pts3d = read_model.read_points3d_binary(points3dfile)
    
    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
    
    return poses, pts3d, perm

def gen_poses(basedir, match_type, factors=None):
    
    files_needed = ['{}.bin'.format(f) for f in ['cameras', 'images', 'points3D']]
    if os.path.exists(os.path.join(basedir, 'sparse/0')):
        files_had = os.listdir(os.path.join(basedir, 'sparse/0'))
    else:
        files_had = []
    if not all([f in files_had for f in files_needed]):
        print( 'Need to run COLMAP' )
        #run_colmap(basedir, match_type) #COLMAP은 로컬에서 돌려서 오기
    else:
        print('Don\'t need to run COLMAP')
        
    print( 'Post-colmap')
    
    poses, pts3d, perm = load_colmap_data(basedir)
    
    save_poses(basedir, poses, pts3d, perm)
    
    if factors is not None:
        print( 'Factors:', factors)
        minify(basedir, factors)
    
    print( 'Done with imgs2poses' )
    
    return True

def save_poses(basedir, poses, pts3d, perm):
    pts_arr = []
    vis_arr = []
    for k in pts3d:
        pts_arr.append(pts3d[k].xyz)
        cams = [0] * poses.shape[-1]
        for ind in pts3d[k].image_ids:
            if len(cams) < ind - 1:
                print('ERROR: the correct camera poses for current points cannot be accessed')
                return
            cams[ind-1] = 1
        vis_arr.append(cams)

    pts_arr = np.array(pts_arr)
    vis_arr = np.array(vis_arr)
    print( 'Points', pts_arr.shape, 'Visibility', vis_arr.shape )
    
    zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose([2,0,1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
    valid_z = zvals[vis_arr==1]
    print( 'Depth stats', valid_z.min(), valid_z.max(), valid_z.mean() )
    
    save_arr = []
    for i in perm:
        vis = vis_arr[:, i]
        zs = zvals[:, i]
        zs = zs[vis==1]
        close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9)
        # print( i, close_depth, inf_depth )
        
        save_arr.append(np.concatenate([poses[..., i].ravel(), np.array([close_depth, inf_depth])], 0))
    save_arr = np.array(save_arr)
    
    np.save(os.path.join(basedir, 'poses_bounds.npy'), save_arr)
    
def prepare_pose_etc(poses_bounds, downsample_factor=4):
    poses = poses_bounds[:, :15].reshape(-1, 3, 5)

    near_fars = poses_bounds[:, -2:]
    hwf = poses[:, :, -1]
    H, W, focal = poses[0, :, -1]
    img_wh = np.array([int(W / downsample_factor), int(H / downsample_factor)]) 
    focal = [focal * img_wh[0] / W, focal * img_wh[1] / H]

    poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    blender2opencv = np.eye(4)
    poses, pose_avg = center_poses(poses, blender2opencv)

    near_original = near_fars.min()
    scale_factor = near_original * 0.75

    near_fars /= scale_factor #bds_min, bds_max는 정규화 되는게 아니라 scale_factor에 따라서 스케일이 정해짐
    poses[..., 3] /= scale_factor

    W, H = img_wh
    return poses, near_fars, H, W, focal, H, W, scale_factor


def calculate_psnr(image_path1, image_path2):
    # 이미지를 컬러로 읽기
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    
    # 이미지가 제대로 로드되었는지 확인
    if img1 is None or img2 is None:
        print("Error: Could not load images.")
        return

    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
    # MSE 계산
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        # 두 이미지가 완벽하게 동일한 경우
        return float('inf')
    psnr = peak_signal_noise_ratio(img1, img2)
    # # PSNR 계산
    # max_pixel = 255.0
    # #psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    # psnr = -10.0*np.log(mse/np.log(10.0))
    # #PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))
    return psnr


def calculate_psnr_video(video_file1_path, video_file2_path):
    cap1 = cv2.VideoCapture(video_file1_path)
    cap2 = cv2.VideoCapture(video_file2_path)
    
    mse_values = []
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        mse = np.mean((frame1 - frame2) ** 2)
        mse_values.append(mse)
    
    cap1.release()
    cap2.release()
    
    mse_avg = np.mean(mse_values)
    if mse_avg == 0:
        return float('inf')
    max_pixel_value = 255.0
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse_avg))
    return psnr


    
def get_depth_ray(depth_gts_path, poses_bounds, poses_bounds_colmap):
    depth_gts = load_colmap_depth(depth_gts_path)
    print("load_colmap_depth OK")
    poses_bounds = np.load(poses_bounds)
    print("poses_bounds OK")

    poses_bounds_colmap = np.load(poses_bounds_colmap)

    poses, near_fars, H, W, focal, H, W, scale_factor = prepare_pose_etc(poses_bounds)

    directions = get_ray_directions_blender(H, W, focal)

    hold_id=[0,]
    i_test = np.array(hold_id)
    split = 'train'

    video_list = i_test if split != 'train' else list(set(np.arange(len(poses))) - set(i_test))

    all_rays_depth = []
    is_colmap_pose = 1

    for i in video_list:
        if is_colmap_pose:
            bds_max = poses_bounds_colmap[i, -1]
        else:
            bds_max = near_fars[i, -1]

        c2w = torch.FloatTensor(poses[i])
        c2w = c2w.numpy()

        rays_o_col, rays_d_col = get_rays_by_coord_np(H, W, focal[0], c2w, depth_gts[i]['coord'])
        rays_o_col = torch.tensor(rays_o_col)
        rays_d_col = torch.tensor(rays_d_col)
        rays_o_col, rays_d_col = ndc_rays_blender(H, W, focal[0], 1.0, rays_o_col, rays_d_col)
        rays_o_col = rays_o_col.float()
        rays_d_col = rays_d_col.float()

        depth_value = depth_gts[i]['depth'][:,None,None]
        weights = depth_gts[i]['error'][:,None,None]
        depth_value = torch.tensor(depth_value)
        depth_value = depth_value.squeeze(-1)
        depth_value = depth_value / bds_max
        weights = torch.tensor(weights)
        weights = weights.squeeze(-1)
        rays_depth = torch.cat([rays_o_col, rays_d_col], 1).half()
        rays_depth = torch.cat([rays_depth, depth_value, weights], axis=1)

        all_rays_depth += [rays_depth]
        print(bds_max)

    all_rays_depth = torch.cat(all_rays_depth, 0)

    frame = 300
    all_rays_depth = all_rays_depth.unsqueeze(0)
    all_rays_depth = all_rays_depth.expand(frame, -1, -1)
    all_rays_depth = all_rays_depth.reshape(-1, 8)
    

    return all_rays_depth


def save_video_frames(video_path, output_folder):
    # 비디오 캡쳐 객체 생성
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # 출력 폴더가 없다면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    frame_idx = 1
    while True:
        # 프레임별로 읽기
        ret, frame = cap.read()
        # 더 이상 읽을 프레임이 없으면 종료
        if not ret:
            break
        
        # 프레임 저장
        output_path = os.path.join(output_folder, f"frame_{frame_idx:04d}.png")
        cv2.imwrite(output_path, frame)
        print(f"Saved {output_path}")
        frame_idx += 1
    
    # 자원 해제
    cap.release()
    print("Finished saving frames.")


if __name__ == '__main__':
    depth_gts_path = '/data2/kkm/km/data/flame_steak'
    poses_bounds = '/data2/kkm/km/data/flame_steak/poses_bounds.npy'
    poses_bounds_colmap = '/data2/kkm/km/data/flame_steak/poses_bounds_colmap.npy'
    depth_ray = get_depth_ray(depth_gts_path, poses_bounds, poses_bounds_colmap)
    print(depth_ray)
import numpy as np
import torch
from dataLoader.ray_utils import ndc_rays_blender
from dataLoader.ray_utils import get_rays
from dataLoader.ray_utils import get_ray_directions_blender
from about_depth import prepare_pose_etc
from tqdm import tqdm
from opt import config_parser
import configargparse
import os
from models.tensoRF import raw2alpha, TensorVMSplit, AlphaGridMask
from renderer import OctreeRender_trilinear_fast
from torch.cuda.amp import autocast
from argparse import Namespace
from renderer import write_video
import multiprocessing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device=="cuda": torch.cuda.empty_cache()

torch.set_default_dtype(torch.float32)
device = torch.device("cuda")
print(device)
#print(torch.cuda.get_device_name(device))
print(torch.cuda.device_count())
args = config_parser()
ckpt = torch.load(args.ckpt, map_location=device)
kwargs = ckpt['kwargs']

tensorf = eval(args.model_name)(args, kwargs['aabb'], kwargs['gridSize'], device,
                                    density_n_comp=kwargs['density_n_comp'], appearance_n_comp=kwargs['appearance_n_comp'],
                                    app_dim=args.data_dim_color, near_far=kwargs['near_far'],
                                    shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre,
                                    density_shift=args.density_shift, distance_scale=args.distance_scale,
                                    rayMarch_weight_thres=args.rm_weight_mask_thre,
                                    rayMarch_weight_thres_static=args.rm_weight_mask_thre_static,
                                    pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe,
                                    featureC=args.featureC, step_ratio=kwargs['step_ratio'], fea2denseAct=args.fea2denseAct,
                                    den_dim=args.data_dim_density, densityMode=args.densityMode, featureD=args.featureD,
                                    rel_pos_pe=args.rel_pos_pe, n_frames=args.n_frames,
                                    amp=args.amp, temporal_variance_threshold=args.temporal_variance_threshold,
                                    n_frame_for_static=args.n_frame_for_static,
                                    dynamic_threshold=args.dynamic_threshold, n_time_embedding=args.n_time_embedding,
                                    static_dynamic_seperate=args.static_dynamic_seperate,
                                    zero_dynamic_sigma=args.zero_dynamic_sigma,
                                    zero_dynamic_sigma_thresh=args.zero_dynamic_sigma_thresh,
                                    sigma_static_thresh=args.sigma_static_thresh,
                                    n_train_frames=args.n_train_frames,
                                    net_layer_add=args.net_layer_add,
                                    density_n_comp_dynamic=args.n_lamb_sigma_dynamic,
                                    app_n_comp_dynamic=args.n_lamb_sh_dynamic,
                                    interpolation=args.interpolation,
                                    dynamic_granularity=args.dynamic_granularity,
                                    point_wise_dynamic_threshold=args.point_wise_dynamic_threshold,
                                    static_point_detach=args.static_point_detach,
                                    dynamic_pool_kernel_size=args.dynamic_pool_kernel_size,
                                    time_head=args.time_head, filter_thresh=args.filter_threshold,
                                    static_featureC=args.static_featureC,
                                    )
tensorf.load(ckpt)
print("ckpt loaded!")
savePath = args.basedir

renderer = OctreeRender_trilinear_fast

pose_path =  os.path.join(args.datadir, "poses_bounds.npy")
#pose_path = "~~" # poses_bounds.npy 위치 직접 지정
poses_bounds = np.load(pose_path)
poses, near_fars, H, W, focal, H, W, scale_factor = prepare_pose_etc(poses_bounds, 4)
c2w = poses[2, :, :] # 0, 1, 2 카메라 번호 명시, 0번째 카메라 위치에서 렌더링 진행, 
print("poses_bounds OK")

directions = get_ray_directions_blender(H, W, focal)
processings = []
nodepth = True
with torch.no_grad():
    with autocast(enabled=bool(args.amp)):
        
        temporal_indices = torch.arange(args.n_frames).long().cuda()
        c2w = torch.FloatTensor(c2w)
        print("render on here : \n", c2w)
        rays_o, rays_d = get_rays(directions, c2w)
        rays_o, rays_d = ndc_rays_blender(H, W, focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)

        print('pre-render')

        retva = renderer(rays, tensorf, std_train=None, chunk=args.batch_size*4, N_samples=-1,
                                ndc_ray=args.ndc_ray, white_bg = False, device=device, with_grad=False,
                                simplify=True, static_branch_only=False, temporal_indices=temporal_indices,
                                remove_foreground=False, diff_calc=False, render_path=True, nodepth=True)
        retva = Namespace(**retva)
        #print((retva.static_depth_map).shape)
        #retva.static_depth_map = retva.static_depth_map.cpu().numpy()
        #np.save('/data2/kkm/km/mixvoxels_depth/cam0_5', retva.static_depth_map)
        retva.comp_rgb_map = retva.comp_rgb_map.clamp(0.0, 1.0)
        
        

        proc = multiprocessing.Process(target=write_video, args=(retva.comp_rgb_map.cpu(), savePath, 1, (None if nodepth else retva.comp_depth_map.cpu()), 30, 10,
                                                                 H, W, args.n_train_frames, near_fars))
        processings.append(proc)
        proc.start()

        for proc in processings:
            proc.join()

    print("rendering video COMPLETED !")
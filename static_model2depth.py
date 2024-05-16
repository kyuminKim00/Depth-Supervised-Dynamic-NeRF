from render_one_video import *


# python static_model2depth.py --checkpoint_path /data2/kkm/km/log/cam21_static_only_changed/flame_steak_25000_static_only/flame_steak_25000_static_only.th


if __name__=='__main__':
    args = config_parser()
    data_dir = args.datadir
    depth_dir = data_dir +"/depth"
    print(depth_dir)
    print(args.cam_num_list)

    for i in args.cam_num_list:
        if i == " ":
            pass
        else:
            print("cam_num :", i )
            render_one_video(args, int(i), 0, 1)
    
    depth_npy_path = [os.path.join(depth_dir, file) for file in os.listdir(depth_dir) if file.endswith('.npy')]

    all_depth = []
    for i in depth_npy_path:
        depth = np.load(i)
        depth = torch.Tensor(depth)
        depth = depth.unsqueeze(2)
        all_depth.append(depth)

    all_depth = np.concatenate(all_depth, axis=0)
    all_depth = np.array(all_depth)
    print(all_depth.shape)

    savePath = depth_dir+"/all_depth.npy"
    np.save(savePath, all_depth)
    print("all_depth.npy save done")



import numpy as np
import math



def eul2rotm(eul, sequence='ZYX'): 
    if eul.shape != (3, 1): #[rx, ry, rz]
        
        raise ValueError("Input Euler angles must be a 3x1 numpy array.")

    rotm = np.zeros((3, 3))

    s_1 = np.sin(eul[0])
    c_1 = np.cos(eul[0])
    s_2 = np.sin(eul[1])
    c_2 = np.cos(eul[1])
    s_3 = np.sin(eul[2])
    c_3 = np.cos(eul[2])

    if sequence == 'ZYX':
        rotm[0, 0] = c_1 * c_2
        rotm[0, 1] = c_1 * s_2 * s_3 - s_1 * c_3
        rotm[0, 2] = c_1 * s_2 * c_3 + s_1 * s_3

        rotm[1, 0] = s_1 * c_2
        rotm[1, 1] = s_1 * s_2 * s_3 + c_1 * c_3
        rotm[1, 2] = s_1 * s_2 * c_3 - c_1 * s_3

        rotm[2, 0] = -s_2
        rotm[2, 1] = c_2 * s_3
        rotm[2, 2] = c_2 * c_3
    elif sequence == 'ZYZ':
        rotm[0, 0] = c_1 * c_2 * c_3 - s_1 * s_3
        rotm[0, 1] = -c_1 * c_2 * s_3 - s_1 * c_3
        rotm[0, 2] = c_1 * s_2

        rotm[1, 0] = s_1 * c_2 * c_3 + c_1 * s_3
        rotm[1, 1] = -s_1 * c_2 * s_3 + c_1 * c_3
        rotm[1, 2] = s_1 * s_2

        rotm[2, 0] = -s_2 * c_3
        rotm[2, 1] = s_2 * s_3
        rotm[2, 2] = c_2
    else:
        raise ValueError("Unknown axis sequence.")

    return rotm

def rotm2eul(rotm, sequence='ZYX'):
    if rotm.shape != (3, 3):
        raise ValueError("Input rotation matrix must be a 3x3 numpy array.")

    if sequence == 'ZYX':
        sy = np.sqrt(rotm[0, 0] ** 2 +  rotm[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(rotm[2, 1], rotm[2, 2])
            y = np.arctan2(-rotm[2, 0], sy)
            z = np.arctan2(rotm[1, 0], rotm[0, 0])
        else:
            x = np.arctan2(-rotm[1, 2], rotm[1, 1])
            y = np.arctan2(-rotm[2, 0], sy)
            z = 0
    elif sequence == 'ZYZ':
        sy = np.sqrt(rotm[2, 0] ** 2 +  rotm[2, 1] ** 2)
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(rotm[1, 2], rotm[0, 2])
            y = np.arctan2(sy, rotm[2, 2])
            z = np.arctan2(rotm[2, 1], -rotm[2, 0])
        else:
            x = np.arctan2(-rotm[0, 1], rotm[0, 0])
            y = np.arctan2(sy, rotm[2, 2])
            z = 0
    else:
        raise ValueError("Unknown axis sequence.")

    return np.array([z, y, x])
    
def printdatasetINFO(i, data_numpy):
     #N번째 데이터
    print("-------------------------")
    print(i, "번째 카메라")
    print("qx1", data_numpy[i, 0])
    print("qx2", data_numpy[i, 5])
    print("qx3", data_numpy[i, 10])
    print("qy1", data_numpy[i, 1])
    print("qy2", data_numpy[i, 6])
    print("qy3", data_numpy[i, 11])
    print("qz1", data_numpy[i, 2])
    print("qz2", data_numpy[i, 7])
    print("qz3", data_numpy[i, 12])
    print("tx", data_numpy[i, 3])
    print("ty", data_numpy[i, 8])
    print("tz", data_numpy[i, 13])
    print("H", data_numpy[i, 4])
    print("W", data_numpy[i, 9])
    print("Focal_length", data_numpy[i, 14])
    print("bds_min", data_numpy[i, 15])
    print("bds_max", data_numpy[i, 16])
    print("-------------------------")


if __name__ == '__main__':
# ###############################################
# ## 카메라 당 하나의 포즈 만들기 ##
# ###############################################

    tile_size = 1.3 #지하에 타일 사이즈 (m)
    tx, ty ,tz = 1.3, 0.6, 0
    H, W = 720, 1280
    Focal_length = 1011.4238041579493 #colmap으로 얻은 초점거리, 아이폰 이미지 정보로 보면 2.5mm
    bds_min, bds_max = tile_size*3, tile_size*5
    orientation = np.array([tx, ty, tz])
    orientation = orientation.reshape(-1,1)
    rx = math.radians(0) 
    ry = math.radians(0)
    rz = math.radians(0)
    image_param = np.array([H, W, Focal_length])
    image_param = image_param.reshape(-1,1)
    bds = np.array([bds_min, bds_max])
    rotm = eul2rotm(np.array([[rx], [ry], [rz]]))
    pose = np.hstack([rotm, orientation, image_param])
    pose = pose.reshape(1, -1)
    pose = np.concatenate([pose, bds.reshape(1, -1)], axis=1)
    #np.save('/data2/kkm/km/mixvoxels/pose_bound4.npy', pose) # 카메라 당 하나의 포즈 만들기


# ###############################################
# ## 포즈 만들어 놓고 하나의 넘파이 배열로 합치기 ##
# ###############################################

    # pose_list = ['/data2/kkm/km/mixvoxels/pose_bound1.npy', '/data2/kkm/km/mixvoxels/pose_bound2.npy',
    #  '/data2/kkm/km/mixvoxels/pose_bound3.npy', '/data2/kkm/km/mixvoxels/pose_bound4.npy']

    # pose_buff =[]
    # for pose_one in pose_list:
    #     pose_numpy = np.load(pose_one)
    #     pose_buff.append(pose_numpy)

    # pose_buff = np.vstack(pose_buff)
    # np.save('/data2/kkm/km/mixvoxels/poses_bounds.npy', pose_buff)
    flame_steak = np.load("/data2/kkm/km/mixvoxels/data/flame_steak/poses_bounds.npy")
    basement = np.load("/data2/kkm/km/mixvoxels/data/basement/poses_bounds.npy")
    BMW_dataset = np.load("/data2/kkm/km/NeuLF_dynamic/NeuLF_vec3_xyz_dynamic/dataset/240105_test/poses_bounds.npy")
    printdatasetINFO(0, flame_steak)


    